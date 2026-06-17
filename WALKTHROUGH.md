# Project Walkthrough & Repo Findings

This document summarizes the current repository layout, architectural findings, dependencies, the details of the vLLM startup failure, and a structured path forward for the LLM Inference and Observability assignment.

---

## 🔍 1. Repository Structure & Component Audit

The repository contains all the necessary modules to build a self-correcting text-to-SQL agent, evaluate it, and observe its behavior:

| Directory/File | Description & Purpose | Status / Findings |
| :--- | :--- | :--- |
| **[`data/bird/`](file:///home/sunilkunchoor/mlops-assignment/data/bird/)** | Folder containing the sqlite databases from BIRD benchmark. | **Verified:** 12 `.sqlite` databases exist (e.g., `california_schools.sqlite`, `codebase_community.sqlite` etc.). The data population phase is complete. |
| **[`docker-compose.yml`](file:///home/sunilkunchoor/mlops-assignment/docker-compose.yml)** | Defines the observability stack: Prometheus (9090), Grafana (3000), Langfuse Web (3001) + DB backend services. | **Ready:** Configurations scrape port `8000` on the host for vLLM metrics. |
| **[`agent/`](file:///home/sunilkunchoor/mlops-assignment/agent/)** | Python package implementing the LangGraph text-to-SQL logic. | **Incomplete (Phase 3):** `generate_sql_node` is a worked example, but `verify_node`, `revise_node`, and `route_after_verify` router are pending implementation. Prompts are empty. |
| **[`evals/`](file:///home/sunilkunchoor/mlops-assignment/evals/)** | Offline evaluation script using BIRD benchmark questions. | **Incomplete (Phase 5):** `run_eval.py` has helper functions, but `eval_one` and `summarize` are pending. |
| **[`load_test/`](file:///home/sunilkunchoor/mlops-assignment/load_test/)** | Performance driver script to run load tests against the agent. | **Ready:** Accepts `--rps` and `--duration` flags to measure request latencies and response statuses. |
| **[`scripts/`](file:///home/sunilkunchoor/mlops-assignment/scripts/)** | Helper shell and Python scripts. | **Needs Tuning (Phase 1):** `start_vllm.sh` currently executes standard host-based run. |

---

## ⚡ 2. vLLM Host Startup Issue Analysis

When launching `vLLM` directly on the host using `uv run python -m vllm.entrypoints.openai.api_server`, it fails with:
`ImportError: libcuda.so.1: cannot open shared object file: No such file or directory` and `RuntimeError: Failed to infer device type`.

### Technical Rationale
1. **Dynamic Linking Defect:** The vLLM installation on the host is attempting to bind to the GPU platform using PyTorch and CUDA. However, the system's dynamic linker cannot find the NVIDIA CUDA driver shared library (`libcuda.so.1`).
2. **Missing Driver Bindings:** This occurs if NVIDIA drivers are not installed on the host, the driver libraries are not in the dynamic library search path (e.g. `/usr/local/cuda` is missing or not in `LD_LIBRARY_PATH`), or the current user shell cannot access the GPU nodes.

---

## 🛠️ 3. Recommended Development Strategy

To work efficiently, you do not need to debug the H100 CUDA driver immediately. You can split your tasks into **Off-GPU Prototyping** and **GPU-Based Tuning**:

### Option A: Off-GPU Prototyping (Phases 2-5)
You can mock or substitute the heavy 30B model with CPU-based vLLM or a hosted API:
1. **Hosted API (OpenAI/Anthropic):** Add your keys and endpoint variables to `.env`:
   ```bash
   VLLM_BASE_URL=https://api.openai.com/v1
   VLLM_MODEL=gpt-4o-mini
   OPENAI_API_KEY=sk-...
   ```
2. **CPU-only local vLLM:** Run vLLM with CPU routing:
   ```bash
   export VLLM_TARGET_DEVICE=cpu
   uv run python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen3-0.6B-Instruct \
     --device cpu \
     --host 0.0.0.0 \
     --port 8000
   ```
This allows you to implement:
- **Phase 2:** The Grafana dashboard metrics (`vllm:num_requests_running`, `vllm:time_to_first_token_seconds`, etc.).
- **Phase 3:** The LangGraph logic and prompt engineering in `agent/graph.py` and `agent/prompts.py`.
- **Phase 4:** Tracing via Langfuse callback.
- **Phase 5:** Executing evaluation runs via `run_eval.py`.

### Option B: GPU-based Deployment (Phases 1 & 6)
Once the logic is validated:
1. Verify if GPU is visible via `nvidia-smi`.
2. Add CUDA paths to environment:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. Boot up the large model `Qwen/Qwen3-30B-A3B-Instruct-2507`.
4. Perform load testing (`uv run python load_test/driver.py --rps 10 --duration 300`) and optimize parameters (like KV cache allocations, max model length, etc.) to meet the SLO (P95 latency < 5 seconds, 10+ RPS).

---

## 📈 4. Observability Metrics Checklist

To extend [`infra/grafana/provisioning/dashboards/serving.json`](file:///home/sunilkunchoor/mlops-assignment/infra/grafana/provisioning/dashboards/serving.json), build panels for:

1. **Latency:**
   - **TTFT (Time To First Token) P50/P90/P95:** `histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))`
   - **TPOT (Time Per Output Token) P50/P90/P95:** `histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[1m])) by (le))`
   - **E2E Request Latency:** `histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[1m])) by (le))`
2. **Throughput:**
   - **Token Generation Rate:** `rate(vllm:generation_tokens_total[1m])`
   - **Token Prompt Processing Rate:** `rate(vllm:prompt_tokens_total[1m])`
   - **Requests Completed (Success vs. Failures):** `rate(vllm:request_success_total[1m])` and `rate(vllm:request_failure_total[1m])`
3. **KV Cache & Concurrency:**
   - **GPU KV Cache Usage:** `vllm:gpu_cache_usage_perc` (0 to 1 scale)
   - **Active Request Count (Running vs. Waiting):** `vllm:num_requests_running` and `vllm:num_requests_waiting`
   - **Preemptions (Indicators of cache saturation):** `rate(vllm:num_preemptions_total[1m])`

---

## 📋 5. Step-by-Step Actionable Checklist (What to Do)

Follow this checklist sequentially to implement and complete the assignment:

### 🟩 Phase 0: Setup & Infrastructure
* [ ] **Start UIs:** Run `docker compose up -d` and ensure port forwards are active:
  * Prometheus: `http://localhost:9090`
  * Grafana: `http://localhost:3000` (admin/admin)
  * Langfuse: `http://localhost:3001` (sign up locally)
* [ ] **Environment Setup:** Copy `.env.example` to `.env` and fill in necessary token/API values.

### 🟩 Phase 1 & 2: Inference & Serving Monitoring
* [ ] **Boot serving layer:** Decide on local vLLM (CPU / GPU) or hosted fallback. Check if `nvidia-smi` works for GPU.
* [ ] **Extend Grafana:** Build dashboards in Grafana UI covering:
  * **Latency**: TTFT, TPOT, and E2E request latency histograms.
  * **Throughput**: Input/output tokens, success/failure rate.
  * **KV cache**: Cache utilization gauge, queue size, and preemption rate.
* [ ] **Persist Dashboard:** Save Grafana dashboard JSON back to [`infra/grafana/provisioning/dashboards/serving.json`](file:///home/sunilkunchoor/mlops-assignment/infra/grafana/provisioning/dashboards/serving.json) so it's checked into git.

### 🟩 Phase 3 & 4: Agent Coding & Tracing
* [ ] **Code Agent Nodes:** Open [`agent/graph.py`](file:///home/sunilkunchoor/mlops-assignment/agent/graph.py) and implement:
  * `verify_node`: checks if query execution succeeded and returned non-empty, reasonable rows.
  * `revise_node`: revises SQL by taking prior SQL, query result, and verifier's feedback.
  * `route_after_verify`: routes to `revise` if verify failed, or `END` if it passed or iteration limit is reached.
* [ ] **Draft Prompts:** Fill in prompt templates in [`agent/prompts.py`](file:///home/sunilkunchoor/mlops-assignment/agent/prompts.py) for generation, verification, and revision.
* [ ] **Test & Trace:** Run the agent FastAPI app and trigger questions via `curl`.
* [ ] **Verify Langfuse Waterfall:** Open Langfuse UI (`http://localhost:3001`), inspect traces, and ensure spans have tags.

### 🟩 Phase 5: Offline Evaluations
* [ ] **Implement Evals:** In [`evals/run_eval.py`](file:///home/sunilkunchoor/mlops-assignment/evals/run_eval.py), fill in:
  * `eval_one`: queries agent API, executes gold SQL vs predicted SQL against SQLite DB, compares canonicalized row sets, and returns correct-status mapping.
  * `summarize`: aggregates correctness at each step iteration (0, 1, 2, 3+).
* [ ] **Baseline Run:** Execute `uv run python evals/run_eval.py --out results/eval_baseline.json`. Verify the agent loop improves accuracy.

### 🟩 Phase 6: Load Testing & Tuning
* [ ] **Run Load Test:** Run driver via `uv run python load_test/driver.py --rps 8 --duration 300` and watch Grafana.
* [ ] **Tune vLLM:** Modify `scripts/start_vllm.sh` flags (e.g. `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`) to fix bottlenecks.
* [ ] **Final Eval:** Run post-tuning evals: `uv run python evals/run_eval.py --out results/eval_after_tuning.json`.

### 🟩 Phase 7: Reporting
* [ ] **Write Report:** Document findings in `REPORT.md` following guidelines in `README.md` (config rationale, baseline vs post-tuning performance, iteration log, agent value analysis, next steps).
