"""
Generate synthetic trade reconciliation data.

Creates in data/:
  fo_trades.csv        - front office trades (the "truth")
  bo_trades.csv        - custodian/back office copy with ~20 injected breaks
  injected_breaks.csv  - answer key: which trade got which break
  break_history.json   - past breaks with root cause + resolution (for RAG)
  recon_rules.json     - reconciliation rules (for RAG)

Run from the project root:  uv run python generate_data.py
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)  # same data every run
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_TRADES = 200
N_BREAKS = 20

# instrument, isin, currency, base price
INSTRUMENTS = [
    ("VOD.L",   "GB00BH4HKS39", "GBP", 0.72),
    ("HSBA.L",  "GB0005405286", "GBP", 6.80),
    ("BARC.L",  "GB0031348658", "GBP", 2.10),
    ("LLOY.L",  "GB0008706128", "GBP", 0.55),
    ("BP.L",    "GB0007980591", "GBP", 4.60),
    ("AAPL",    "US0378331005", "USD", 225.00),
    ("MSFT",    "US5949181045", "USD", 430.00),
    ("JPM",     "US46625H1005", "USD", 210.00),
    ("SAP.DE",  "DE0007164600", "EUR", 190.00),
    ("ASML.AS", "NL0010273215", "EUR", 680.00),
]
COUNTERPARTIES = [
    "Goldman Sachs International",
    "J.P. Morgan Securities plc",
    "Morgan Stanley & Co. International plc",
    "Barclays Bank PLC",
    "UBS AG London Branch",
    "Citigroup Global Markets Limited",
]
# alternative legal entity names used to create counterparty breaks
CPTY_ALIASES = {
    "Goldman Sachs International": "Goldman Sachs & Co. LLC",
    "J.P. Morgan Securities plc": "JPMorgan Chase Bank N.A.",
    "Morgan Stanley & Co. International plc": "Morgan Stanley & Co. LLC",
    "Barclays Bank PLC": "Barclays Capital Securities Ltd",
    "UBS AG London Branch": "UBS Europe SE",
    "Citigroup Global Markets Limited": "Citigroup Global Markets Inc.",
}
BOOKS = {"GBP": "EQ-UK-CASH", "USD": "EQ-US-CASH", "EUR": "EQ-EU-CASH"}
LOTS = [100, 250, 500, 1000, 2500, 5000, 10000]


def add_business_days(d, n):
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def make_trade(i):
    instrument, isin, ccy, base = random.choice(INSTRUMENTS)
    trade_date = add_business_days(date(2026, 9, 1), random.randint(0, 25))
    return {
        "trade_id": f"T{1000 + i}",
        "trade_date": trade_date.isoformat(),
        "settle_date": add_business_days(trade_date, 2).isoformat(),  # T+2
        "instrument": instrument,
        "isin": isin,
        "side": random.choice(["BUY", "SELL"]),
        "quantity": random.choice(LOTS),
        "price": round(base * random.uniform(0.98, 1.02), 4),
        "currency": ccy,
        "counterparty": random.choice(COUNTERPARTIES),
        "book": BOOKS[ccy],
    }


# ---------- 1. clean trades (front office = truth) ----------
fo = pd.DataFrame([make_trade(i) for i in range(N_TRADES)])

# ---------- 2. back office starts as an identical copy ----------
bo = fo.copy()

# ---------- 3. inject breaks on the back office side ----------
BREAK_TYPES = ["quantity", "price", "settle_date", "counterparty",
               "currency", "missing_in_bo", "missing_in_fo"]

break_ids = random.sample(list(fo["trade_id"]), N_BREAKS)
answer_key = []
extra_bo_rows = []

for n, tid in enumerate(break_ids):
    kind = BREAK_TYPES[n % len(BREAK_TYPES)]  # spread evenly across types
    idx = bo.index[bo["trade_id"] == tid][0]
    row = bo.loc[idx]

    if kind == "currency" and row["currency"] != "GBP":
        kind = "price"  # GBP/GBp confusion only makes sense for UK stocks

    if kind == "quantity":
        old = row["quantity"]
        new = old // 10 if old >= 1000 else old * 10  # extra/missing zero
        bo.at[idx, "quantity"] = new
        detail = f"quantity FO={old} BO={new}"

    elif kind == "price":
        old = row["price"]
        new = round(old * random.choice([1.005, 0.995, 1.01]), 4)
        bo.at[idx, "price"] = new
        detail = f"price FO={old} BO={new}"

    elif kind == "settle_date":
        old = row["settle_date"]
        new = add_business_days(date.fromisoformat(old), 1).isoformat()
        bo.at[idx, "settle_date"] = new
        detail = f"settle_date FO={old} BO={new}"

    elif kind == "counterparty":
        old = row["counterparty"]
        new = CPTY_ALIASES[old]
        bo.at[idx, "counterparty"] = new
        detail = f"counterparty FO={old} BO={new}"

    elif kind == "currency":
        # custodian reports UK stock in pence: GBp and price x100
        bo.at[idx, "currency"] = "GBp"
        bo.at[idx, "price"] = round(row["price"] * 100, 2)
        detail = f"currency FO=GBP BO=GBp, price FO={row['price']} BO={row['price'] * 100:.2f}"

    elif kind == "missing_in_bo":
        bo = bo.drop(idx)
        detail = "trade in FO but not in BO"

    elif kind == "missing_in_fo":
        # BO has a trade the FO never booked (e.g. unrecognised custodian trade)
        ghost = make_trade(9000 + n)
        ghost["trade_id"] = f"C{9000 + n}"
        extra_bo_rows.append(ghost)
        tid = ghost["trade_id"]
        detail = "trade in BO but not in FO"

    answer_key.append({"trade_id": tid, "break_type": kind, "detail": detail})

if extra_bo_rows:
    bo = pd.concat([bo, pd.DataFrame(extra_bo_rows)], ignore_index=True)

fo.to_csv(DATA_DIR / "fo_trades.csv", index=False)
bo.to_csv(DATA_DIR / "bo_trades.csv", index=False)
pd.DataFrame(answer_key).to_csv(DATA_DIR / "injected_breaks.csv", index=False)

# ---------- 4. knowledge base for RAG ----------
break_history = [
    # quantity
    {"break_type": "quantity", "field": "quantity",
     "description": "FO quantity 10000, custodian 1000. Factor of 10 difference.",
     "root_cause": "Trader fat-finger: extra zero keyed in the order management system; custodian received the broker confirm with the correct quantity.",
     "resolution": "Checked broker confirmation, which matched custodian. FO trade amended by desk; amendment approved by supervisor.",
     "team": "Trade Support"},
    {"break_type": "quantity", "field": "quantity",
     "description": "FO shows full order quantity, custodian shows a smaller amount on the same day.",
     "root_cause": "Partial fill: order executed in several fills, only the first fill settled to custodian by cut-off.",
     "resolution": "Matched sum of fills from execution report. Remaining fills arrived next day; break auto-cleared.",
     "team": "Middle Office"},
    {"break_type": "quantity", "field": "quantity",
     "description": "Quantity differs by a small odd amount between FO and custodian.",
     "root_cause": "Corporate action (stock split / scrip dividend) applied at custodian but not yet in FO static data.",
     "resolution": "Confirmed corporate action with custodian notice; FO position adjusted by corporate actions team.",
     "team": "Corporate Actions"},
    {"break_type": "quantity", "field": "quantity",
     "description": "Block trade quantity in FO, custodian shows several smaller trades.",
     "root_cause": "Block allocated to sub-accounts; custodian reports post-allocation, FO reports pre-allocation block.",
     "resolution": "Reconciled allocations to block total. Recon rule updated to compare at allocation level.",
     "team": "Middle Office"},
    {"break_type": "quantity", "field": "quantity",
     "description": "FO quantity is 10 times the custodian quantity for a UK equity.",
     "root_cause": "Booking entered in wrong lot size on legacy booking screen.",
     "resolution": "Desk corrected booking after confirming against broker confirm; added validation on lot size.",
     "team": "Trade Support"},
    # price
    {"break_type": "price", "field": "price",
     "description": "Price difference of about 0.5% between FO and custodian.",
     "root_cause": "FO booked average price across fills; custodian settled each fill at its execution price.",
     "resolution": "Recalculated volume-weighted average price from fills; difference within tolerance after aggregation.",
     "team": "Middle Office"},
    {"break_type": "price", "field": "price",
     "description": "Small price difference, consideration differs by the commission amount.",
     "root_cause": "Custodian price is gross of commission, FO price is net (or vice versa).",
     "resolution": "Compared net consideration instead of price; break closed. Recon config changed to compare net amounts.",
     "team": "Reconciliations"},
    {"break_type": "price", "field": "price",
     "description": "Price breaks above tolerance on a late-afternoon trade.",
     "root_cause": "Trader booked indicative price before execution; actual execution price differed.",
     "resolution": "Desk amended FO price to broker confirmation price.",
     "team": "Trade Support"},
    {"break_type": "price", "field": "price",
     "description": "Price mismatch on a US equity, around 1%.",
     "root_cause": "Price rounding: FO stores 4 decimals, custodian 2 decimals, combined with an FX-converted booking.",
     "resolution": "Confirmed economic amount matched; trade booked in wrong currency view, corrected to local currency.",
     "team": "Reconciliations"},
    # settle date
    {"break_type": "settle_date", "field": "settle_date",
     "description": "Custodian settlement date one business day after FO.",
     "root_cause": "Market holiday in settlement market not in FO holiday calendar, so FO calculated T+2 incorrectly.",
     "resolution": "Holiday calendar updated in FO static data; settle date amended.",
     "team": "Static Data"},
    {"break_type": "settle_date", "field": "settle_date",
     "description": "Settle date differs by one day, non-standard settlement.",
     "root_cause": "Trade agreed with non-standard settlement (T+3) with counterparty but FO defaulted to T+2.",
     "resolution": "Confirmed agreed settlement terms on broker confirm; FO amended.",
     "team": "Trade Support"},
    {"break_type": "settle_date", "field": "settle_date",
     "description": "Settlement date mismatch for a trade around a weekend.",
     "root_cause": "Booking system calculated settlement using calendar days rather than business days.",
     "resolution": "Bug raised with booking system team; trade amended manually.",
     "team": "Middle Office"},
    {"break_type": "settle_date", "field": "settle_date",
     "description": "Custodian shows later settle date after a failed settlement.",
     "root_cause": "Settlement failed due to insufficient securities from counterparty; custodian rolled the date.",
     "resolution": "Fail tracked with settlements team; FO expected settle date updated; buy-in monitored.",
     "team": "Settlements"},
    # counterparty
    {"break_type": "counterparty", "field": "counterparty",
     "description": "Counterparty names differ but belong to the same banking group.",
     "root_cause": "Trader booked against the group's US entity; trade actually executed with the UK entity.",
     "resolution": "Checked LEI on broker confirmation; FO counterparty amended to correct legal entity. Credit limits rechecked.",
     "team": "Trade Support"},
    {"break_type": "counterparty", "field": "counterparty",
     "description": "Counterparty mismatch after a counterparty entity restructure.",
     "root_cause": "Counterparty migrated business to new legal entity (Brexit-related); FO static data not updated.",
     "resolution": "Static data updated with new entity and SSIs; open trades re-pointed.",
     "team": "Static Data"},
    {"break_type": "counterparty", "field": "counterparty",
     "description": "Custodian shows a different counterparty name for the same LEI.",
     "root_cause": "Naming difference only; same legal entity identifier in both systems.",
     "resolution": "Recon matching switched from name to LEI; break was a false positive.",
     "team": "Reconciliations"},
    # currency
    {"break_type": "currency", "field": "currency",
     "description": "FO shows GBP, custodian shows GBp, price is 100 times higher at custodian.",
     "root_cause": "UK equities quoted in pence (GBp) by custodian and exchange; FO stores pounds (GBP). Unit difference, not economic break.",
     "resolution": "Normalised GBp to GBP (divide price by 100) before matching; break closed as presentation difference.",
     "team": "Reconciliations"},
    {"break_type": "currency", "field": "currency",
     "description": "Trade currency differs between systems for a cross-listed stock.",
     "root_cause": "Stock traded on secondary listing in different currency; FO static data pointed to primary listing.",
     "resolution": "Corrected instrument listing (exchange/ISIN + currency) in FO.",
     "team": "Static Data"},
    {"break_type": "currency", "field": "currency",
     "description": "Settlement currency differs from trade currency at custodian.",
     "root_cause": "Client requested settlement in a different currency with FX at custodian.",
     "resolution": "Compared on trade currency; FX leg reconciled separately.",
     "team": "Settlements"},
    # missing in BO
    {"break_type": "missing_in_bo", "field": "trade",
     "description": "Trade exists in FO but not at custodian.",
     "root_cause": "Late booking: trade booked after custodian instruction cut-off, instruction sent next day.",
     "resolution": "Confirmed instruction sent next morning; break cleared on next recon cycle.",
     "team": "Settlements"},
    {"break_type": "missing_in_bo", "field": "trade",
     "description": "FO trade never reached custodian.",
     "root_cause": "Settlement instruction rejected due to missing or wrong SSI (standing settlement instruction).",
     "resolution": "SSI corrected in static data; instruction resent; settlement date agreed with counterparty.",
     "team": "Settlements"},
    {"break_type": "missing_in_bo", "field": "trade",
     "description": "FO trade missing at custodian; trade was cancelled by desk.",
     "root_cause": "Trade cancelled in FO but cancellation not processed correctly, leaving a stale live trade in FO.",
     "resolution": "Confirmed cancellation with desk; FO trade status corrected to cancelled.",
     "team": "Trade Support"},
    # missing in FO
    {"break_type": "missing_in_fo", "field": "trade",
     "description": "Custodian reports a trade not found in FO.",
     "root_cause": "Trade executed by broker under a give-up arrangement but not booked by desk.",
     "resolution": "Desk confirmed trade; booked late in FO with supervisor approval. Escalated as unbooked trade.",
     "team": "Trade Support"},
    {"break_type": "missing_in_fo", "field": "trade",
     "description": "Unknown trade at custodian.",
     "root_cause": "Custodian booked trade to wrong account (belongs to another client).",
     "resolution": "Raised DK (don't know) with custodian; trade moved to correct account.",
     "team": "Settlements"},
    {"break_type": "missing_in_fo", "field": "trade",
     "description": "Custodian trade with no FO match, matches a corporate action.",
     "root_cause": "Custodian generated movement from a corporate action (e.g. scrip), not a real trade.",
     "resolution": "Classified as corporate action movement; excluded from trade recon.",
     "team": "Corporate Actions"},
]
for i, b in enumerate(break_history, start=1):
    b["break_id"] = f"B{i:03d}"
    b["doc_type"] = "past_break"

recon_rules = [
    {"rule_id": "R01", "title": "Matching key",
     "text": "Trades are matched on trade_id. If no match, try ISIN + side + quantity + trade_date."},
    {"rule_id": "R02", "title": "Price tolerance",
     "text": "Price differences within 0.1% are auto-matched. Above 0.1% is a price break and needs investigation."},
    {"rule_id": "R03", "title": "Quantity tolerance",
     "text": "Quantity must match exactly. Any quantity difference is a break. Factor-of-10 differences suggest booking error; check broker confirm first."},
    {"rule_id": "R04", "title": "Settlement date",
     "text": "Settlement date must match. Standard equity settlement is T+2 (business days). Check holiday calendars and agreed terms."},
    {"rule_id": "R05", "title": "Currency normalisation",
     "text": "Normalise GBp to GBP (divide price by 100) before comparing. A GBP/GBp difference alone is not an economic break."},
    {"rule_id": "R06", "title": "Counterparty matching",
     "text": "Counterparties should be compared by LEI where available. Name differences within the same group need legal entity check against broker confirm."},
    {"rule_id": "R07", "title": "Missing trades",
     "text": "Trades in FO but not BO: check instruction status and SSIs. Trades in BO but not FO: raise with desk as potential unbooked trade, same day."},
    {"rule_id": "R08", "title": "Escalation - ageing",
     "text": "Breaks open more than 1 business day escalate to team lead. More than 3 days escalate to Head of Operations."},
    {"rule_id": "R09", "title": "Escalation - value",
     "text": "Any break with notional above 1,000,000 in trade currency is escalated immediately to team lead and the desk."},
    {"rule_id": "R10", "title": "Evidence",
     "text": "Every break resolution must reference evidence: broker confirmation, custodian statement, or approved amendment."},
]
for r in recon_rules:
    r["doc_type"] = "rule"

with open(DATA_DIR / "break_history.json", "w") as f:
    json.dump(break_history, f, indent=2)
with open(DATA_DIR / "recon_rules.json", "w") as f:
    json.dump(recon_rules, f, indent=2)

print(f"FO trades: {len(fo)}, BO trades: {len(bo)}, injected breaks: {len(answer_key)}")
print(f"Knowledge base: {len(break_history)} past breaks, {len(recon_rules)} rules")
print(pd.DataFrame(answer_key).to_string(index=False))