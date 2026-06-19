#!/usr/bin/env python3
"""Script to call the hosted model or agent server manually.

Supports choosing a question from evals/eval_set.jsonl or passing custom input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Ensure we can import from the root workspace directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

load_dotenv()

from agent.prompts import GENERATE_SQL_SYSTEM, GENERATE_SQL_USER
from agent.schema import render_schema
from agent.execution import execute_sql

DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8001/answer")


def load_question_from_eval_set(eval_file: Path, index: int) -> tuple[str, str, str | None]:
    """Loads a specific question from eval_set.jsonl.
    Returns (question, db_id, gold_sql).
    """
    if not eval_file.exists():
        print(f"Error: Eval file not found at {eval_file}", file=sys.stderr)
        sys.exit(1)
        
    lines = eval_file.read_text().splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    
    if index < 0 or index >= len(questions):
        print(f"Error: Index {index} is out of bounds. File has {len(questions)} items.", file=sys.stderr)
        sys.exit(1)
        
    q = questions[index]
    return q["question"], q["db_id"], q.get("gold_sql")


def call_hosted_model(question: str, db_id: str) -> None:
    """Queries the hosted vLLM model directly at VLLM_BASE_URL."""
    print("=" * 80)
    print(f"Target: Hosted Model (vLLM) at {VLLM_BASE_URL}")
    print(f"Model:  {VLLM_MODEL}")
    print(f"DB ID:  {db_id}")
    print(f"Q:      {question}")
    print("=" * 80)
    
    # 1. Render schema
    try:
        schema = render_schema(db_id)
    except Exception as e:
        print(f"Error rendering schema for DB {db_id}: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Construct messages
    system_prompt = GENERATE_SQL_SYSTEM
    user_prompt = GENERATE_SQL_USER.format(schema=schema, question=question)
    
    # 3. Call vLLM API
    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    print("Sending request to model...")
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"\nAPI Call Failed: {e}", file=sys.stderr)
        if isinstance(e, httpx.HTTPStatusError):
            print(f"Response Body: {e.response.text}", file=sys.stderr)
        sys.exit(1)
        
    content = data["choices"][0]["message"]["content"]
    print("\n--- Raw Model Output ---")
    print(content)
    print("-" * 40)
    
    # 4. Try parsing and executing SQL
    import re
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else content).strip()
    
    print("\n--- Extracted SQL ---")
    print(sql)
    print("-" * 40)
    
    print("\n--- Executing SQL ---")
    result = execute_sql(db_id, sql)
    print(result.render())
    print("-" * 40)


def call_agent_server(question: str, db_id: str) -> None:
    """Queries the agent FastAPI server at AGENT_URL."""
    print("=" * 80)
    print(f"Target: Agent Server at {AGENT_URL}")
    print(f"DB ID:  {db_id}")
    print(f"Q:      {question}")
    print("=" * 80)
    
    payload = {
        "question": question,
        "db": db_id,
        "tags": {"run_type": "manual_query"}
    }
    
    print("Sending request to agent...")
    try:
        response = httpx.post(AGENT_URL, json=payload, timeout=90.0)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"\nAgent Call Failed: {e}", file=sys.stderr)
        if isinstance(e, httpx.HTTPStatusError):
            print(f"Response Body: {e.response.text}", file=sys.stderr)
        sys.exit(1)
        
    print("\n--- Agent Response ---")
    print(f"Success (ok): {data.get('ok')}")
    print(f"Iterations:   {data.get('iterations')}")
    if data.get("error"):
        print(f"Error:        {data.get('error')}")
    print(f"Final SQL:    {data.get('sql')}")
    
    print("\n--- Iteration History ---")
    history = data.get("history", [])
    for idx, step in enumerate(history):
        print(f"Step {idx + 1} ({step.get('node')}):")
        print(f"  SQL: {step.get('sql')}")
        
    print("\n--- Execution Preview (Rows) ---")
    rows = data.get("rows")
    if rows is not None:
        print(f"Returned {len(rows)} rows.")
        for r in rows[:10]:
            print(" | ".join(str(c) for c in r))
        if len(rows) > 10:
            print(f"... ({len(rows) - 10} more rows)")
    else:
        print("No rows returned.")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call hosted model or agent server with a query.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE, help="Path to eval_set.jsonl")
    
    # Query source selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, help="Index of the question in eval_set.jsonl (0-indexed)")
    group.add_argument("--question", type=str, help="Custom question to ask")
    
    parser.add_argument("--db", type=str, help="Database ID (required if using --question)")
    
    # Target selection
    parser.add_argument(
        "--target",
        choices=["model", "agent"],
        default="model",
        help="Whether to query the hosted model directly or the agent API server"
    )
    
    args = parser.parse_args()
    
    # Determine the query and db
    gold_sql = None
    if args.question:
        if not args.db:
            parser.error("--db is required when passing a custom --question.")
        question = args.question
        db_id = args.db
    else:
        # Default to first question in eval set if no selection is made
        idx = args.index if args.index is not None else 0
        question, db_id, gold_sql = load_question_from_eval_set(args.eval_set, idx)
        
    if gold_sql:
        print(f"Loaded question index from eval set.")
        print(f"Gold SQL: {gold_sql}\n")
        
    # Execute the target call
    if args.target == "model":
        call_hosted_model(question, db_id)
    else:
        call_agent_server(question, db_id)


if __name__ == "__main__":
    main()
