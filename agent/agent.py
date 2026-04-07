import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_DB = Path(__file__).parent.parent / "warehouse" / "data.duckdb"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def llm(system: str, user: str, *, api_key: str, max_tokens: int = 1024) -> str:
    payload = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response shape: {body}") from exc


def discover_schema(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    rows = conn.execute("""
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        ORDER BY table_schema, table_name, ordinal_position
    """).fetchall()

    schema: dict[str, list[str]] = {}
    for tschema, tname, col in rows:
        key = f"{tschema}.{tname}"
        schema.setdefault(key, []).append(col)
    return schema


def schema_summary(schema: dict[str, list[str]]) -> str:
    return "\n".join(
        f"  {table}: {', '.join(cols)}"
        for table, cols in sorted(schema.items())
    )


SYSTEM_SCHEMA_ANALYST = """
You are a DuckDB SQL expert working with a GTM data warehouse.
Write a single DuckDB SQL query that returns ONE row with ONE column called
"context_json" containing a JSON object with all relevant GTM context for a
given account.

CRITICAL DuckDB JSON rules — do NOT use functions that don't exist:
- Use json_object('key', value, ...) to build objects
- Use json_array(v1, v2, ...) for fixed arrays
- For aggregating rows into a JSON array use:
    (SELECT json_group_array(json_object(...)) FROM ... WHERE ...)
  NOT json_array_agg (does not exist in DuckDB)
- Cast timestamps/dates to VARCHAR if needed: CAST(col AS VARCHAR)

The query must cover: account profile, opportunities, recent calls,
product usage, marketing funnel, key contacts.
Prefer marts schema tables. Use subqueries to avoid row explosion.
No writes. Use ? as the single bind parameter (account name or id).

Return ONLY raw SQL, no markdown fences, no explanation.
""".strip()

SYSTEM_EMAIL_DRAFTER = """
You are an expert B2B sales development rep writing a highly personalized
outreach email based on real internal data about an account.

RULES — every single one is mandatory:
1. Use the ACTUAL person's name from contacts data as the greeting (first name,
   primary contact or champion). Never write [Contact Name].
2. Reference at least 2 specific numbers from the data (e.g. ARR, active users,
   deal amount, call count, engagement score, usage trend %).
3. Name the actual competitor if one is mentioned in calls or opportunities.
4. Reference the actual deal stage or opportunity name if one exists.
5. Mention a specific product usage signal (feature adoption, engagement tier,
   usage trend) if product_usage data is present.
6. Subject line must reference something concrete from the data — company name,
   competitor, product metric, or deal stage. Under 10 words.
7. Body: 3 short paragraphs. No filler. No "I hope this finds you well."
8. End with one low-friction CTA — a specific question or 15-min call ask.
9. Sign off as "XYZ" (generic rep name).

If a field is missing or null in the data, skip it — do not invent facts.

Output format (exactly):
SUBJECT: <subject>

<paragraph 1>

<paragraph 2>

<paragraph 3>

XYZ
""".strip()


def strip_fences(text: str) -> str:
    if "```" in text:
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )
    return text.strip()


def build_and_run_context_query(
    conn: duckdb.DuckDBPyConnection,
    schema: dict[str, list[str]],
    identifier: str,
    id_type: str,
    api_key: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    user_prompt = (
        f"Warehouse schema:\n{schema_summary(schema)}\n\n"
        f"identifier_type='{id_type}', value='{identifier}'\n"
        f"Write the DuckDB SQL query."
    )

    sql = strip_fences(llm(
        system=SYSTEM_SCHEMA_ANALYST,
        user=user_prompt,
        api_key=api_key,
        max_tokens=1500,
    ))

    for attempt in range(max_retries + 1):
        try:
            rows = conn.execute(sql, [identifier]).fetchall()
            if not rows or rows[0][0] is None:
                raise ValueError(f"Query returned no data for '{identifier}'.")
            raw = rows[0][0]
            return json.loads(raw) if isinstance(raw, str) else raw

        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Query failed after {max_retries + 1} attempts: {exc}") from exc

            print(f"Query error on attempt {attempt + 1}, asking LLM to fix...")
            sql = strip_fences(llm(
                system=SYSTEM_SCHEMA_ANALYST,
                user=(
                    f"{user_prompt}\n\n"
                    f"Your previous query failed with this error:\n{exc}\n\n"
                    f"Previous query:\n{sql}\n\n"
                    f"Fix the query and return only the corrected SQL."
                ),
                api_key=api_key,
                max_tokens=1500,
            ))

    raise RuntimeError("Unreachable")


def fetch(conn: duckdb.DuckDBPyConnection, sql: str, params: list) -> list[dict]:
    res = conn.execute(sql, params)
    rows = res.fetchall()
    cols = [d[0] for d in res.description]
    return [dict(zip(cols, row)) for row in rows]


def fallback_context(
    conn: duckdb.DuckDBPyConnection,
    schema: dict[str, list[str]],
    identifier: str,
    id_type: str,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}

    account_id: int | None = None
    if id_type == "account_id":
        account_id = int(identifier)
    elif id_type == "account_name" and "marts.dim_accounts" in schema:
        rows = conn.execute(
            "SELECT account_id FROM marts.dim_accounts WHERE lower(name) = lower(?)",
            [identifier],
        ).fetchall()
        if rows:
            account_id = rows[0][0]
    elif id_type == "email":
        for tbl in ("raw.contacts", "raw.product_users"):
            if tbl in schema:
                rows = conn.execute(
                    f"SELECT account_id FROM {tbl} WHERE lower(email) = lower(?)",
                    [identifier],
                ).fetchall()
                if rows:
                    account_id = rows[0][0]
                    break

    if account_id is None:
        raise ValueError(f"Could not resolve account for '{identifier}' (type={id_type}).")

    ctx["account_id"] = account_id

    if "marts.dim_accounts" in schema:
        rows = fetch(conn, "SELECT * FROM marts.dim_accounts WHERE account_id = ?", [account_id])
        if rows:
            ctx["account"] = rows[0]

    if "marts.fct_opportunities" in schema:
        ctx["opportunities"] = fetch(conn, """
            SELECT opp_id, opp_name, stage, amount, CAST(close_date AS VARCHAR) close_date,
                   forecast_category, competitor, loss_reason,
                   total_calls, avg_talk_ratio, buying_signal_mentions,
                   objection_mentions, pricing_mentions
            FROM marts.fct_opportunities
            WHERE account_id = ?
            ORDER BY created_at DESC LIMIT 5
        """, [account_id])

    if "marts.fct_calls" in schema:
        ctx["recent_calls"] = fetch(conn, """
            SELECT CAST(call_date AS VARCHAR) call_date, duration_minutes,
                   opp_name, opp_stage, competitor_mentions, pricing_mentions,
                   buying_signal_mentions, objection_mentions, risk_mentions
            FROM marts.fct_calls
            WHERE account_id = ?
            ORDER BY call_date DESC LIMIT 5
        """, [account_id])

    if "marts.fct_product_usage" in schema:
        ctx["product_usage"] = fetch(conn, """
            SELECT CAST(usage_month AS VARCHAR) usage_month, active_users,
                   total_events, unique_features_used, engagement_tier, usage_trend_pct
            FROM marts.fct_product_usage
            WHERE account_id = ?
            ORDER BY usage_month DESC LIMIT 3
        """, [account_id])

    if "marts.fct_funnel" in schema:
        ctx["funnel"] = fetch(conn, """
            SELECT lead_source, first_campaign_name, first_campaign_channel,
                   total_activities, email_clicks, webinars_attended,
                   content_downloads, reached_mql, reached_sql, reached_closed_won
            FROM marts.fct_funnel
            WHERE converted_account_id = ?
            LIMIT 5
        """, [account_id])

    if "raw.contacts" in schema and "raw.contact_roles" in schema:
        ctx["contacts"] = fetch(conn, """
            SELECT c.first_name, c.last_name, c.title, c.role,
                   cr.role AS deal_role, cr.is_primary
            FROM raw.contacts c
            LEFT JOIN raw.contact_roles cr ON c.contact_id = cr.contact_id
            WHERE c.account_id = ?
            ORDER BY cr.is_primary DESC NULLS LAST LIMIT 5
        """, [account_id])

    return ctx


def summarise_context(ctx: dict[str, Any]) -> None:
    """Print a human-readable summary of what data was actually pulled."""
    print("\nContext summary:")
    acc = ctx.get("account", {})
    if acc:
        print(f"Account: {acc.get('name')} | {acc.get('segment')} | "
              f"ARR ${acc.get('arr', 0):,} | {acc.get('industry')} | {acc.get('region')}")

    opps = ctx.get("opportunities", [])
    print(f"Opps: {len(opps)} found", end="")
    if opps:
        o = opps[0]
        print(f" — latest: '{o.get('opp_name')}' [{o.get('stage')}] "
              f"${o.get('amount', 0):,} vs {o.get('competitor') or 'no competitor'}", end="")
    print()

    calls = ctx.get("recent_calls", [])
    print(f"Calls: {len(calls)} recent", end="")
    if calls:
        c = calls[0]
        print(f" — last: {c.get('call_date','')[:10]}, "
              f"buying_signals={c.get('buying_signal_mentions')}, "
              f"objections={c.get('objection_mentions')}", end="")
    print()

    usage = ctx.get("product_usage", [])
    if usage:
        u = usage[0]
        print(f"Usage: {u.get('active_users')} active users | "
              f"{u.get('unique_features_used')} features | "
              f"tier={u.get('engagement_tier')} | trend={u.get('usage_trend_pct')}%")

    contacts = ctx.get("contacts", [])
    if contacts:
        names = [f"{c.get('first_name')} {c.get('last_name')} ({c.get('deal_role') or c.get('role')})"
                 for c in contacts[:3]]
        print(f"Contacts: {', '.join(names)}")
    print()


def draft_email(context: dict[str, Any], api_key: str) -> str:
    return llm(
        system=SYSTEM_EMAIL_DRAFTER,
        user=(
            f"Account context (real data from our warehouse):\n```json\n"
            f"{json.dumps(context, indent=2, default=str)}\n```\n\n"
            "Draft the outreach email now. Every fact you mention must come "
            "directly from the JSON above — no placeholders, no invented details."
        ),
        api_key=api_key,
        max_tokens=1024,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GTM Outreach Agent — drafts personalized emails using OpenAI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--account", metavar="NAME", help="Account name")
    group.add_argument("--account-id", metavar="ID", type=int, help="Account ID")
    group.add_argument("--prospect-email", metavar="EMAIL", help="Prospect email")

    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help=f"Path to DuckDB file (default: {DEFAULT_DB})")
    parser.add_argument("--show-context", action="store_true",
                        help="Print full raw context JSON")
    parser.add_argument("--show-sql", action="store_true",
                        help="Print the LLM-generated SQL query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not set.")

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"Error: DuckDB file not found at '{db_path}'.")

    conn = duckdb.connect(str(db_path), read_only=True)

    print("Discovering schema...")
    schema = discover_schema(conn)
    schemas_found = sorted({k.split(".")[0] for k in schema})
    print(f"Found {len(schema)} tables across: {', '.join(schemas_found)}")

    if args.account:
        identifier, id_type = args.account, "account_name"
    elif args.account_id:
        identifier, id_type = str(args.account_id), "account_id"
    else:
        identifier, id_type = args.prospect_email, "email"

    print(f"\nGenerating context query for {id_type}='{identifier}'...")
    try:
        context = build_and_run_context_query(conn, schema, identifier, id_type, api_key)
        print("Context gathered via LLM-generated query.")
    except Exception as exc:
        print(f"LLM query failed ({exc}). Falling back to manual queries...")
        context = fallback_context(conn, schema, identifier, id_type)
        print("Context gathered via fallback queries.")

    summarise_context(context)

    if args.show_context:
        print("--- Full Context JSON ---")
        print(json.dumps(context, indent=2, default=str))
        print("--- End Context ---\n")

    print("Drafting personalized outreach email...")
    email = draft_email(context, api_key)

    print("\n" + "=" * 70)
    print(email)
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()