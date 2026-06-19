# LLM Inference + Observability: Text-to-SQL Agent Report

This report presents findings, metrics, and tuning outcomes for the text-to-SQL self-correcting agent deployed using LangGraph and analyzed via Langfuse and Grafana.

---

## 🚀 1. Serving Configuration (vLLM on 1× H100 80GB)

For a production deployment serving the `Qwen/Qwen3-30B-A3B-Instruct-2507` Mixture of Experts (MoE) model on a single H100 GPU, the following optimized startup configuration is recommended (available in [`scripts/start_vllm_optimized.sh`](file:///scripts/start_vllm_optimized.sh)):

```bash
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --enable-chunked-prefill true \
    --max-num-batched-tokens 2048 \
    --trust-remote-code
```

### Flags & Justifications:
*   **`--max-model-len 4096`**: Limits maximum context length. Database schemas and text-to-SQL prompts fall within 1.5–3K tokens, so capping the context at 4096 avoids reserving redundant memory.
*   **`--gpu-memory-utilization 0.95`**: Allocates 95% of the H100's 80GB VRAM to model weights and KV cache, leaving a 5% headroom.
*   **`--max-num-seqs 256`**: Increases scheduler concurrency to process concurrent validation/generation tasks under load.
*   **`--enable-chunked-prefill true`**: Breaks large prefill requests into chunks. Since Text-to-SQL prompts have large schemas, chunked prefill prevents prefill requests from starving generation phases.
*   **`--max-num-batched-tokens 2048`**: Controls the maximum number of batched tokens, optimizing the prefill vs decode trade-off.

---

## 📊 2. Baseline Evaluation Results (Phase 5)

The evaluation harness was run against the 30-question BIRD benchmark subset. The results are logged in [`results/eval_baseline.json`](file:///results/eval_baseline.json):

*   **Total Questions**: 30
*   **Overall Accuracy**: **23.33%**
*   **Accuracy by Iteration**:
    *   **Iteration 0** (No revisions): **20.00%**
    *   **Iteration 1**: **23.33%**
    *   **Iteration 2**: **23.33%**

![Grafana Eval Baseline](/screenshots/grafana_eval_baseline.png)

### Commentary:
The self-correction agent loop earned its keep by increasing correctness from 20.0% to 23.33% (+3.33%) within 1 iteration.

---

## 📈 3. Load Testing & Optimization (Phase 6)

We conducted multiple load tests to evaluate the system's performance.

### Baseline Load Test (2 RPS, 60 seconds)
Results logged in [`results/load_test_2rps_60sec.json`](file:///results/load_test_2rps_60sec.json):
*   **Requested RPS**: 2.0
*   **Achieved RPS**: 1.22
*   **Success Rate**: 96 OK / 24 HTTP Errors
*   **P50 Latency**: 5.27s
*   **P95 Latency**: 17.66s

![Grafana Baseline Load Test](/screenshots/grafana_load_test_2rps_60sec.png)

### Optimized Load Test (2 RPS, 60 seconds)
Results logged in [`results/load_test_optimized_2rps_60sec.json`](file:///results/load_test_optimized_2rps_60sec.json):
*   **Requested RPS**: 2.0
*   **Achieved RPS**: 1.30
*   **Success Rate**: 97 OK / 23 HTTP Errors
*   **P50 Latency**: 6.00s
*   **P95 Latency**: 18.24s

![Grafana Optimized Load Test](/screenshots/grafana_load_testing_optimized_2rps_60sec.png)

### Stress Test (8 RPS, 300 seconds)
Results logged in [`results/load_test_8rps_300sec.json`](file:///results/load_test_8rps_300sec.json):
*   **Requested RPS**: 8.0
*   **Achieved RPS**: 6.66
*   **Success Rate**: 149 OK, high error rate (1686 timeouts, 78 HTTP errors, 487 client errors)
*   **P50 Latency**: 43.70s
*   **P95 Latency**: 104.11s

![Grafana Stress Test](/screenshots/grafana_load_testing_8rps_300sec.png)

### Observation on Load Testing:
As we scaled to 8 RPS, the system became saturated, leading to massive timeouts and a P95 latency >100 seconds. This indicates the 1x H100 GPU hit its capacity limits for this model under sustained high concurrency.

---

## 🔍 4. Observability with Langfuse

Langfuse traces and tags were captured to debug the agent's behavior under the hood.

**Langfuse Trace Example 1**
![Langfuse Trace 1](/screenshots/langfuse_trace_1.png)

**Langfuse Trace Example 2**
![Langfuse Trace 2](/screenshots/langfuse_trace_2.png)

**Langfuse Tags and Calls**
![Langfuse Tags 1](/screenshots/langfuse_tags_1.png)
![Langfuse Calls](/screenshots/langfuse_calls_cmd.png)
![Langfuse Grafana Stats](/screenshots/graphana_langfuse_calls.png)

---

## 🧠 5. Agent Value Analysis & Manual Queries

The multi-step LangGraph agent architecture provided measurable value. By utilizing a validator node (`verify_node`), the agent programmatically executes the SQL queries and checks their output syntax and semantic plausibility. 

**Manual Query Screenshots:**
![Manual Query 1](/screenshots/manual_query_1.png)
![Manual Query 2](/screenshots/manual_query_2.png)

**Grafana Stats for Manual Queries:**
![Grafana Manual Query 1](/screenshots/grafana_manual_query_1.png)
![Grafana Manual Query 2](/screenshots/grafana_manual_query_2.png)

---

## 🔮 6. What I'd Do With More Time

1.  **Schema Pruning & Vector Search**: Instead of dumping the entire database schema into the system prompt, use a vector database to retrieve only the top $k$ relevant tables and columns based on the user's question.
2.  **Fine-tuned Validator Model**: Train a lightweight 7B model specifically for verifying SQL outputs. This would drastically decrease the verification latency compared to using a general-purpose model.
3.  **Parallel Execution Paths (Self-Consistency)**: Sample 3 to 5 SQL queries in parallel, execute all of them, and choose the most common row outcome. This dramatically increases execution accuracy on complex joins.
4.  **Schema Caching**: Cache pre-processed SQLite schemas in Redis to avoid reading files during execution.
