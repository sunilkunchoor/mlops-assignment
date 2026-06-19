"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    payload = {
        "question": question["question"],
        "db": question["db_id"],
        "tags": {"run_type": "eval_baseline"}
    }
    
    # 1. Query the agent FastAPI endpoint
    try:
        resp = httpx.post(agent_url, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "question": question["question"],
            "db_id": question["db_id"],
            "gold_sql": question["gold_sql"],
            "correct_by_iteration": {"0": False, "1": False, "2": False},
            "final_sql": "",
            "iterations_taken": 0,
            "ok": False,
            "error": f"Agent request failed: {e}",
            "history": []
        }

    # 2. Get gold execution result
    ok_gold, gold_rows, error_gold = run_sql(question["db_id"], question["gold_sql"])
    
    # 3. Evaluate each query in history with carry-forward
    history = data.get("history", [])
    correct_by_iteration = {}
    last_correct = False
    
    # Support at least 3 iterations, or more if history is longer
    num_iters = max(3, len(history))
    
    for k in range(num_iters):
        if k < len(history):
            pred_sql = history[k].get("sql", "")
            if pred_sql:
                ok_pred, pred_rows, error_pred = run_sql(question["db_id"], pred_sql)
                current_correct = matches(gold_rows, pred_rows) if ok_gold and ok_pred else False
            else:
                current_correct = False
            last_correct = current_correct
        else:
            current_correct = last_correct
            
        correct_by_iteration[str(k)] = current_correct

    return {
        "question": question["question"],
        "db_id": question["db_id"],
        "gold_sql": question["gold_sql"],
        "correct_by_iteration": correct_by_iteration,
        "final_sql": data.get("sql", ""),
        "iterations_taken": data.get("iterations", 0),
        "ok": data.get("ok", False),
        "error": data.get("error"),
        "history": history
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    if total == 0:
        return {
            "total_questions": 0,
            "overall_accuracy": 0.0,
            "accuracy_by_iteration": {}
        }
        
    all_iters = set()
    for r in results:
        all_iters.update(r["correct_by_iteration"].keys())
    sorted_iters = sorted(list(all_iters), key=int)
    
    correct_counts = {k: 0 for k in sorted_iters}
    for r in results:
        for k in sorted_iters:
            if r["correct_by_iteration"].get(k, False):
                correct_counts[k] += 1
                
    accuracy_by_iteration = {
        k: count / total for k, count in correct_counts.items()
    }
    
    # The overall accuracy is the accuracy of the final output (max iteration results)
    max_iter_str = sorted_iters[-1] if sorted_iters else "0"
    overall_correct = sum(1 for r in results if r["correct_by_iteration"].get(max_iter_str, False))
    overall_accuracy = overall_correct / total
    
    return {
        "total_questions": total,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_iteration": accuracy_by_iteration
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
