# Your Project Name

A starter template for the AI Engineering Buildcamp capstone. Replace this README with a description of your own project.

## The Problem

When a bank's front-office trading system and its back-office or risk systems disagree about a trade, rule-based matching flags the break, but operations and data analysts still spend hours manually investigating the root cause across bookings, reference data, prices and timing. This investigation step is slow, repetitive and error-prone, and unresolved breaks create operational and regulatory risk.

## What It Does

ReconAgent is an AI agent that investigates trade reconciliation breaks. A deterministic SQL matching engine first compares front-office and back-office trade records and produces a list of breaks. The agent then takes each break and uses tools to query trade, reference and price data, reasoning step by step about why the records disagree.

**Typical interaction:**

- **User provides:** a break selected from the list (or a file of breaks to process in batch).
- **System returns:**
  - a diagnosed root cause (e.g. incorrect counterparty LEI mapping, stale price, missing leg, amendment not propagated)
  - the evidence the agent checked to reach that conclusion
  - a confidence level
  - a suggested resolution note for the analyst to approve or reject

The data combines real public reference data (GLEIF LEIs, ESMA instrument data, market prices) with synthetic trades containing seeded breaks. Because each seeded break is logged, the agent's diagnoses can be scored against a known answer key.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.
