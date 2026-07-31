"""
modules/ai.py
==============
Generative AI layer powering the "AI Strategy Consultant" page:
1. Text-to-SQL — converts natural language questions into executable
   SQLite queries using Google Gemini.
2. AI Business Consultant — turns query results into executive-style
   narrative insights and strategic recommendations.

Together these implement the Question -> SQL -> Result -> Executive
Summary -> Strategic Recommendations workflow the UI presents as a
chat-style consultant.

Design decisions:
- The Gemini prompt is deliberately strict ("return ONLY SQL, no
  markdown, no explanation") because LLMs default to wrapping code in
  ``` fences and adding commentary — both would break naive execution.
  We still defensively strip fences in code, never trust the prompt alone.
- We pass the schema (column names + types) into the prompt rather than
  hoping Gemini "knows" the table structure. Without this, the model
  hallucinates plausible-sounding but wrong column names.
- SQL execution is read-only by construction: we reject anything that
  isn't a single SELECT statement, so a bad or adversarial generation
  can't mutate the warehouse or smuggle a second statement in behind a
  semicolon. This is a real safety consideration in a text-to-SQL
  feature, not just a nicety.
- The Gemini client is created lazily inside functions (not at import
  time) so the app can still boot and show a friendly error if the
  API key isn't configured yet, instead of crashing on import.

v2 changes (enterprise polish pass):
- _is_safe_select() now also rejects stacked/multiple statements (e.g.
  "SELECT 1; DROP TABLE sales;") as defense-in-depth. Python's sqlite3
  module already refuses to execute more than one statement per call,
  so this wasn't previously exploitable — but an explicit check here
  means the app reports a clear "not permitted" message instead of
  relying silently on a driver-level implementation detail, and it's a
  more complete answer if asked about SQL-injection handling in an
  interview.
"""

import os
import re
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "sales.db"

SCHEMA_DESCRIPTION = """
Table: sales
Columns:
  Transaction_ID TEXT
  Date TEXT (format YYYY-MM-DD)
  Region TEXT (North, South, East, West, Central)
  Country TEXT
  Drug_Name TEXT
  Therapy_Area TEXT (Oncology, Cardiology, Neurology, Diabetes, Vaccines)
  Hospital TEXT
  Doctor_Segment TEXT
  Sales_Representative TEXT
  Units_Sold INTEGER
  Revenue REAL
  Discount REAL
  Manufacturing_Cost REAL
  Profit REAL
  Marketing_Spend REAL
  Inventory INTEGER
  Customer_Type TEXT
  Quarter TEXT
  Month TEXT
"""

SQL_SYSTEM_PROMPT = """You are an expert SQLite database engineer.

You will be given a business question in plain English and a table schema.

Generate ONLY executable SQLite SQL that answers the question.
Return no explanation.
Return no markdown formatting or code fences.
Return only a single SQL statement, ending in a semicolon.
Only generate SELECT statements — never INSERT, UPDATE, DELETE, DROP, or ALTER.
Always reference the table name exactly as "sales".
"""


def _get_gemini_client():
    """Lazily configure and return the Gemini model, or raise a clear error."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) to use the AI Strategy Consultant."
        )
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    # "gemini-flash-latest" is Google's stable alias that always points to
    # their current-generation Flash model, so this doesn't need updating
    # every time a specific dated model version is deprecated.
    return genai.GenerativeModel("gemini-flash-latest")


def _extract_sql(raw_text: str) -> str:
    """Strip markdown code fences defensively, even though the prompt asks Gemini not to add them."""
    text = raw_text.strip()
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def _is_safe_select(sql: str) -> bool:
    """Only allow a single, read-only SELECT statement.

    Two checks:
    1. Must start with SELECT and must not contain any DDL/DML keywords.
    2. Must not contain a second statement stacked behind a semicolon
       (e.g. "SELECT 1; DROP TABLE sales;"). We strip exactly one
       trailing semicolon (the one the prompt asks the model to end
       with) and then check no semicolon remains — any remaining ";"
       means there's more than one statement present.
    """
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]

    if ";" in stripped:
        return False  # a second statement is stacked behind this one

    normalized = stripped.strip().upper()
    if not normalized.startswith("SELECT"):
        return False

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "PRAGMA"]
    return not any(f" {kw} " in f" {normalized} " for kw in forbidden)


def natural_language_to_sql(question: str) -> str:
    """Convert an English question into a SQLite SELECT statement via Gemini."""
    model = _get_gemini_client()
    prompt = f"{SQL_SYSTEM_PROMPT}\n\nSchema:\n{SCHEMA_DESCRIPTION}\n\nQuestion: {question}\n\nSQL:"
    response = model.generate_content(prompt)
    return _extract_sql(response.text)


def execute_sql(sql: str) -> dict:
    """
    Execute a generated SQL statement against the SQLite warehouse.
    Never raises on bad SQL — returns a structured error instead, since
    LLM-generated SQL can fail in ways the UI needs to display gracefully.
    """
    if not _is_safe_select(sql):
        return {
            "success": False,
            "error": "Only a single, read-only SELECT statement is permitted.",
            "sql": sql,
        }

    try:
        conn = sqlite3.connect(DB_PATH)
        result_df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"success": True, "data": result_df, "sql": sql}
    except Exception as e:
        return {"success": False, "error": str(e), "sql": sql}


def generate_executive_insights(question: str, result_df) -> str:
    """
    AI Business Consultant step of the AI Strategy Consultant workflow.
    Takes the question + its SQL result and asks Gemini to write a short,
    executive-style narrative: what the numbers mean and what to do about it.

    Design decision: we pass the ACTUAL result data (as a compact table),
    not just the question, so Gemini's narrative is grounded in the real
    numbers rather than inventing plausible-sounding figures.
    """
    model = _get_gemini_client()

    # Cap rows sent to the model — keeps the prompt small and prevents
    # dumping the entire warehouse into a single API call.
    preview = result_df.head(20).to_string(index=False)

    prompt = f"""You are a senior pharmaceutical commercial analytics consultant
presenting findings to a VP of Sales. Be concise and business-focused.

The business question asked was: "{question}"

The query returned this data:
{preview}

Write a short executive summary (2-4 sentences) interpreting what this data
shows, followed by 2-3 bullet-point strategic recommendations grounded in
these specific numbers. Do not restate the raw numbers verbatim in a table —
write in prose, the way a consultant would summarize a finding in a
client meeting."""

    response = model.generate_content(prompt)

    text = response.text.strip()

    # Remove markdown formatting
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", "")

    return text


def ask_question(question: str) -> dict:
    """
    Full text-to-SQL pipeline used by the Streamlit 'AI Strategy Consultant'
    page: English question -> SQL -> executed result.
    """
    try:
        sql = natural_language_to_sql(question)
    except Exception as e:
        return {"success": False, "error": f"SQL generation failed: {e}", "sql": None}

    return execute_sql(sql)
