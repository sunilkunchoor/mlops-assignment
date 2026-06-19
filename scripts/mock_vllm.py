#!/usr/bin/env python3
"""
vLLM Proxy and Metrics Exporter for CPU-only Environments.

Listens on port 8000, forwards /v1/chat/completions to the hosted OpenAI API,
and exports standard vLLM Prometheus metrics at /metrics.
"""
import os
import time
import random
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
import httpx
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Gauge, Histogram
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Default to OpenAI if no VLLM_BASE_URL was set to point elsewhere, or fallback
REAL_BASE_URL = "https://api.openai.com"

app = FastAPI()
client = httpx.AsyncClient(timeout=120.0)

# Prometheus metrics setup using exact vLLM metric names
# In Prometheus, colons are replaced by underscores, so we register them using colons
# (Prometheus python client doesn't allow colons in name directly, so we map them at export time or use prometheus_client trick)
# Actually, prometheus_client allows any name if we override the name or write custom collector,
# but we can also just register them as e.g. vllm_num_requests_running and map them in the exporter.
# Wait! Let's check how prometheus scrapes them. In vllm:num_requests_running, the scraped name in Prometheus database
# becomes vllm:num_requests_running (if it preserves colons) or vllm_num_requests_running.
# In infra/prometheus.yml:
# targets: ['host.docker.internal:8000']
# In serving.json:
# expr: "vllm:num_requests_running"
# Thus, we must expose the metrics with the literal name "vllm:num_requests_running".
# Let's write a custom raw text response for /metrics to ensure the exact metric names with colons are served.

# We will maintain the metrics in memory
METRICS = {
    "num_requests_running": 0.0,
    "num_requests_waiting": 0.0,
    "gpu_cache_usage_perc": 0.0,
    "prompt_tokens_total": 0.0,
    "generation_tokens_total": 0.0,
    "time_to_first_token_seconds_sum": 0.0,
    "time_to_first_token_seconds_count": 0.0,
    "time_per_output_token_seconds_sum": 0.0,
    "time_per_output_token_seconds_count": 0.0,
    "e2e_request_latency_seconds_sum": 0.0,
    "e2e_request_latency_seconds_count": 0.0,
    "num_preemptions_total": 0.0,
    "request_success_total": 0.0,
    "request_failure_total": 0.0,
}

# Histograms buckets
LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]

METRICS_HISTOGRAMS = {
    "time_to_first_token_seconds": {b: 0.0 for b in LATENCY_BUCKETS + [float('inf')]},
    "time_per_output_token_seconds": {b: 0.0 for b in LATENCY_BUCKETS + [float('inf')]},
    "e2e_request_latency_seconds": {b: 0.0 for b in LATENCY_BUCKETS + [float('inf')]},
}

def record_histogram(name, value):
    hist = METRICS_HISTOGRAMS[name]
    METRICS[f"{name}_sum"] += value
    METRICS[f"{name}_count"] += 1.0
    for bucket in LATENCY_BUCKETS:
        if value <= bucket:
            hist[bucket] += 1.0
    hist[float('inf')] += 1.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    # Simulate cache usage decaying or fluctuating slightly
    if METRICS["num_requests_running"] > 0:
        # Increase cache usage based on running requests
        target = min(0.95, METRICS["num_requests_running"] * 0.15 + random.uniform(-0.05, 0.05))
        METRICS["gpu_cache_usage_perc"] = max(0.05, target)
    else:
        METRICS["gpu_cache_usage_perc"] = max(0.0, METRICS["gpu_cache_usage_perc"] - 0.02)
        
    lines = []
    # Write custom Prometheus text format with colons
    for k, v in METRICS.items():
        if "_" in k and (k.endswith("_sum") or k.endswith("_count")):
            base_name = k.rsplit("_", 1)[0]
            col_name = f"vllm:{base_name}"
            suffix = k.split("_")[-1]
            lines.append(f"{col_name}_{suffix} {v}")
        elif k == "gpu_cache_usage_perc" or k == "num_requests_running" or k == "num_requests_waiting":
            lines.append(f"vllm:{k} {v}")
        elif k in ["prompt_tokens_total", "generation_tokens_total", "num_preemptions_total", "request_success_total", "request_failure_total"]:
            lines.append(f"vllm:{k} {v}")
            
    for name, hist in METRICS_HISTOGRAMS.items():
        col_name = f"vllm:{name}"
        for bucket, val in hist.items():
            le = "INF" if bucket == float('inf') else str(bucket)
            lines.append(f'{col_name}_bucket{{le="{le}"}} {val}')
            
    return "\n".join(lines) + "\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not OPENAI_API_KEY:
        return Response(content='{"error": "OPENAI_API_KEY environment variable not set"}', status_code=400)

    # Decode body
    body = await request.json()
    
    # Track metrics
    METRICS["num_requests_running"] += 1.0
    start_time = time.monotonic()
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Forward request to OpenAI
        # Rewrite model to be gpt-4o-mini if it is the default 30B model
        if "qwen" in body.get("model", "").lower():
            body["model"] = "gpt-4o-mini"
            
        resp = await client.post(
            f"{REAL_BASE_URL}/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=120.0
        )
        
        duration = time.monotonic() - start_time
        
        if resp.status_code == 200:
            METRICS["request_success_total"] += 1.0
            resp_data = resp.json()
            
            # Record tokens
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            gen_tokens = usage.get("completion_tokens", 0)
            
            METRICS["prompt_tokens_total"] += prompt_tokens
            METRICS["generation_tokens_total"] += gen_tokens
            
            # Record latencies
            record_histogram("e2e_request_latency_seconds", duration)
            # Simulate TTFT (Time to first token) as a fraction of e2e
            ttft = duration * random.uniform(0.1, 0.3)
            record_histogram("time_to_first_token_seconds", ttft)
            
            # Simulate TPOT (Time per output token)
            if gen_tokens > 0:
                tpot = (duration - ttft) / gen_tokens
                record_histogram("time_per_output_token_seconds", tpot)
            
            return resp_data
        else:
            METRICS["request_failure_total"] += 1.0
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
            
    except Exception as e:
        METRICS["request_failure_total"] += 1.0
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        METRICS["num_requests_running"] -= 1.0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
