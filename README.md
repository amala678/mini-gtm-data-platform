# GTM Outreach Agent — Modified Approach

This repository contains my modified version of the Mini GTM Data Platform assignment. The goal was to build an agent that dynamically gathers GTM context for a given account or prospect and drafts a personalized outreach email, using both LLM-generated and fallback SQL queries.

## Assignment

Build an agent that, given an account or prospect identifier, pulls together relevant internal context — deal history, product usage, call intelligence, marketing engagement, and key contacts — and drafts a personalized outreach email. The agent should dynamically discover the schema, avoid hardcoding table/column names, and generate valid DuckDB SQL. Fallback queries are used if the LLM fails.

## My Modifications

- Rewrote the agent in a modular Python CLI (`agent.py`) for account, account ID, or prospect email.
- Added LLM-assisted query generation using OpenAI GPT-4o-mini.
- Automatic schema discovery with `information_schema.columns` from DuckDB.
- Fallback manual queries to ensure context can always be fetched.
- Summarizes context in human-readable format before drafting email.
- Drafts a personalized outreach email grounded in real GTM data — no placeholders or invented facts.

## Possible Future Modifications

## If I Had More Time

- Add a caching layer to avoid regenerating SQL and re-fetching the same account context repeatedly.
- Use a vector database to store past account contexts, call summaries, and embeddings for better retrieval and personalization.
- Improve LLM reliability by breaking query generation into steps.
- Also, maybe use a feedback loop such as RLHF to verify if the email being generated is up to the mark.
- Add logging and evaluation to track SQL success rates.
- Additionally, I do think unit tests, dependency tests, etc should be written.

## Quick Setup

1. export OPENAI_API_KEY="YOUR_API_KEY"

### CLI Usage
Supports three identifiers:
```bash
--account "Account Name"
--account-id 123
--prospect-email "email@example.com"