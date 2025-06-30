# Evaluation Plan: Quantifying the Benefits of Framework-Level Semantic Awareness

## 1 Goals and Hypotheses
We seek to quantify, with statistical rigour, how much a framework-level semantic-aware runtime (\\sys) can reduce **network traffic** and **end-to-end latency** relative to realistic semantic-blind baselines.

Hypotheses  
H1 (SMALLER TRAFFIC) \\sys reduces bytes transferred during the inference prefill and decode phases by ≥ 30 % compared with the *semantic-blind remote-cache* baseline.  
H2 (FASTER LATENCY) The reduced traffic leads to ≥ 15 % lower median latency for prefill and decode.

## 2 Evaluation Modes
1. **Local (In-Process)** – Model, weights and KV cache on the same GPU (no network). Baseline for "best possible" latency.
2. **Semantic-Blind • Naive** – Every step transfers *weights + input + KV cache* (included for sanity, expected to perform worst).
3. **Semantic-Blind • Remote-Cache (Realistic)** –  
   • Weights uploaded **once** at start and persist on GPU.  
   • KV cache remains on GPU between steps; client receives an opaque handle.  
   • Tokens and logits are transferred uncompressed.  
   This emulates rCUDA-like driver interception without semantic insight.
4. **Framework-Level Semantic-Aware (\\sys)** –  
   • Same caching policy as (3).  
   • Semantic runtime applies:  
      a) *Tensor-type aware compression* (fp16 quantisation + zlib).  
      b) *Delta transfer* of KV cache updates.  
      c) *Semantic batching* for concurrent requests (later work).  

## 3 Workloads
Model = GPT-J-6B (pretrained).  
Scenarios  
  • *Single prompt*: 128-token prefill → 5-token decode loop.  
  • *Batch 3*: three concurrent prompts (64,128,256 tokens) to exercise caching.

## 4 Metrics
1. **Latency (s)** – measured with `/usr/bin/time -f %e`.  
2. **Network Traffic (bytes)** – obtained from PyTorch RPC agent's `get_metrics()` counters; verified with `tcpdump|capinfos` in multi-node runs.  
3. **GPU SM Util (%)** – via `nvidia-smi dmon`, averaged across the run.  
4. **Compression Overhead (ms)** – encode + decode time recorded inside \\sys.

For each (mode, phase) we run 5 trials, report mean ± 95 % CI and perform a paired t-test against baseline (3).

## 5 Experimental Procedure
1. Launch RPC server (GPU side).  
2. Start `nvidia-smi dmon`.  
3. Run client script (`run_llm.py`) in selected mode/phase, capturing time and RPC metrics.  
4. Stop `dmon`; parse CSV with `parse_dmon.py`.  
5. Repeat for all trials; write a CSV summary.  
6. Analyse results with `analyse_results.py` to produce plots and statistical tables.

## 6 Limitations & Future Work
• Local-host loop-back hides real NIC/PCIe contention; we will repeat on two physical nodes once functionality is validated.  
• Compression scheme is prototype; better codecs (e.g., Nvidia TensorRT sparsity) are future work.

---

*Last updated: 2025-06-30*