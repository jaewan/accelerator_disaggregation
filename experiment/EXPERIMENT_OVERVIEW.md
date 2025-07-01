# Framework-Level Semantic Awareness for LLM Inference – Experiment Overview

## Motivation
When the GPU that holds the weights lives on another host, inference requires **shipping tensors across the network**.  By teaching the runtime about *what* is being transferred (its **semantics**) we can avoid redundant or low-value traffic and achieve lower latency.

This repository contains an **end-to-end experiment** that quantifies those benefits.  The codebase implements four execution *modes* and a driver that measures latency, network bytes, and GPU utilisation.

## Execution Modes
| Label (CLI `--mode`) | Description | Traffic Characteristics |
|----------------------|-------------|-------------------------|
| **Local** (`local`) | Model & cache on the *same* GPU.  Upper-bound, no network. | 0 bytes |
| **Semantic-Blind · Naïve** (`naive`) | Every step transfers: **weights + inputs + KV cache + logits**. Slow but simple sanity baseline. | `O(weights + tokens + KV)` per step |
| **Semantic-Blind · Remote-Cache** (`remote_cache`) | 1) Weights copied *once*.<br>2) KV cache **stays on the GPU** and the client only receives an **opaque handle**.<br>3) Tokens & logits sent uncompressed. | `O(tokens + logits)` per step |
| **Framework-Level Semantic-Aware (\sys)** (`sys_simulated`) | Same caching policy as *Remote-Cache* **plus**:<br>a) *Tensor-type aware compression* (fp16-quant + zlib).<br>b) *Delta transfers* of KV cache (future work). | ≪ `O(tokens + logits)` per step |

### Visual Summary
```mermaid
flowchart LR
    subgraph Client
        C1[run_llm.py]
    end
    subgraph Server
        S1[rpc_server.py\nRemoteWorker]
    end
    C1 -- Local --> S1
    C1 -- Naïve --> S1
    C1 -- Remote-Cache --> S1
    C1 -- \sys --> S1
```

## How Caching Is Designed
### 1. Naïve (Stateless) Path
```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    loop Each step
        C->>S: State dict (≈ 6 GB) + input tokens
        S-->>C: Logits + KV cache (~MB)
    end
```
*Implementation*: `run_stateless_forward()` in `rpc_server.RemoteWorker` **loads weights on every call**.

### 2. Remote-Cache (Semantic-Blind) Path
```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: State dict (once)
    C->>S: input tokens
    S-->>C: Logits + **KV handle (UUID)**
    loop Decode steps
        C->>S: Next token + KV handle
        S-->>C: Logits
    end
```
*Implementation*: `run_prefill_with_handle()` and `run_decode_with_handle()`.  KV cache lives on the GPU; the handle is a tiny string.

### 3. \sys Semantic-Aware Path
```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Compressed(tokens)
    S-->>C: Compressed(logits) + kv_id
    loop Decode steps
        C->>S: Compressed(next token) + kv_id
        S-->>C: Compressed(logits)
    end
```
*Key techniques*
1. **Type-aware compression** – fp32→fp16, zlib (`_compress_tensor`).
2. **Symmetric codec** on both ends (`_decompress_tensor`).
3. **KV cache stays remote** but stored **CPU-side** to minimise GPU RAM.

## Component Walk-through
1. **`experiment_driver.py`** – orchestrates multi-trial runs, spawns the RPC server, collects `nvidia-smi dmon`, and writes the final CSV results.
2. **`rpc_server.py` / `RemoteWorker`** – executes model forwards on GPU host.  Provides three families of RPC methods:
   • *Stateless* (`run_stateless_forward_remote`)
   • *Semantic-Blind, cached* (`load_weights_remote`, `run_prefill_with_handle`, …)
   • *Semantic-Aware* (`run_prefill_semantic`, `run_decode_step_semantic`)
3. **`run_llm.py`** – client-side driver.  Selects a mode, prepares inputs, measures and prints `NETWORK_BYTES` so the experiment driver can parse them.

## Measurement Pipeline
```mermaid
graph TD
    subgraph Trial "for each (mode, phase, trial)"
        A[experiment_driver.py] -->|spawn| B[rpc_server.py]
        A -->|/usr/bin/time| C[run_llm.py]
        A -->|dmon CSV| D[nvidia-smi dmon]
        C -->|prints metrics| A
    end
    A --> E[results.csv]
```
1. **Latency**: wall-clock via `/usr/bin/time -f %e`.
2. **Network bytes**: client prints `NETWORK_BYTES` measured in-process.
3. **GPU SM util**: parsed from `dmon` CSV (`parse_dmon.py`).

## Reproducing the Experiment
```bash
python experiment_driver.py \
    --trials 3 \
    --gpu_host 127.0.0.1 \
    --master_port 29501 \
    --model EleutherAI/gpt-j-6B \
    --output results.csv
```
Add `--modes local,remote_cache,sys_simulated` to run a subset, or `--external_server` if you started `rpc_server.py` manually.

## Future Extensions
* **Better codecs** – e.g., sparsity-aware or Nvidia NVCOMP.
* **Cross-request semantic batching**.
* **Real 2-node deployment** to capture NIC/PCIe characteristics.

## Possible Next Action Item
Implement **KV-cache miss handling & delta-transfer** in \sys to expose a clearer gap versus the semantic-blind *Remote-Cache* baseline.

1. **Cache-Miss Path**
   • When a decode step arrives with `kv_id` that is not present on the current GPU (evicted or migrated), \sys should **reconstruct the cache on-demand**.
   • Instead of shipping the *entire* cache (hundreds of MB) back to the GPU, transmit only what is missing:
     – **Layer mask** identifying absent blocks.
     – **Compressed deltas** for those blocks.

2. **Delta-Transfer Protocol**
   | Stage | Semantic-Blind Remote-Cache | \sys w/ Delta Transfer |
   |-------|----------------------------|------------------------|
   | ① Prefill | Upload full KV cache once and keep handle | Same (handle + CPU copy for diffing) |
   | ② GPU Evicts Layer *L* | — (baseline would thrash or OOM) | Move evicted tensors to CPU, mark dirty bits |
   | ③ Next Decode Step | Client still sends token + handle | GPU detects miss → requests only *L* |
   | ④ Network Payload | — (none, cache still resident) | **Δ(L) ≪ full cache** |

3. **Design Sketch**
```mermaid
sequenceDiagram
    participant C as Client
    participant G as GPU_Worker
    participant H as Host_CPU
    C->>G: token + kv_id
    alt cache hit
        G-->>C: logits (compressed)
    else cache miss
        G->>H: request Δ( kv_id )
        H-->>G: compressed tensors for missing layers
        G-->>C: logits (compressed)
    end
```

4. **Why It Beats Semantic-Blind Remote-Cache**
   • Remote-Cache assumes *all* KV tensors are GPU-resident; on an eviction it must reload **everything**, incurring the full transfer cost.
   • \sys tracks per-layer residency + uses semantic compression, so the recovery traffic is *proportional to the miss*, not to the total cache.

5. **Implementation Steps**
   1. Extend `RemoteWorker` to maintain a **CPU-side shadow store** with dirty-bit tracking.
   2. Add RPC methods `fetch_missing_kv_layers(kv_id, mask)` returning compressed deltas.
   3. Update decode path to detect missing layers and issue fetch requests transparently.
   4. Instrument `get_network_counters()` to separate *delta* vs *regular* traffic for analysis.

This feature will enable experiments under **memory pressure scenarios** and should widen the measured gap between \sys and the semantic-blind baseline.

---