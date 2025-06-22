# Experiment Plan: Quantifying the Semantic Translation Gap

**Objective:** To quantitatively demonstrate that semantically-unaware GPU disaggregation systems transfer significantly more data across the network compared to a semantically-aware approach, leading to higher latency and lower accelerator utilization.

**Methodology:** We will implement and test three execution modes for two phases of LLM inference (Prefill and Decode) using a unified RPC framework to ensure a controlled comparison. We will measure wall-clock latency, total network bytes transferred, and GPU utilization for each scenario.

---

## **Phase 1: Environment Setup**

**Goal:** Prepare two physical servers and install all required software.

**Server Roles:**
* `CLIENT_HOST`: An x86-64 machine without a GPU. Runs application logic.
* `GPU_HOST`: An x86-64 machine with an NVIDIA A100 GPU. Acts as the remote resource.

### **Step 1.1: System Provisioning (Both Hosts)**

**Action:** On `CLIENT_HOST` and `GPU_HOST`, ensure the following are installed and configured:
1.  **OS:** Ubuntu 22.04 LTS.
2.  **Networking:** A high-speed Ethernet NIC (at least 25 GbE, e.g., Mellanox ConnectX-5/6 or Intel E810) is installed and configured with static IP addresses. Let `CLIENT_IP` and `GPU_IP` be the respective IP addresses.
3.  **Basic Tools:** Install essential build and monitoring tools.
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential python3.10 python3.10-venv git iperf3 tcpdump nload
    ```

### **Step 1.2: GPU Driver Setup (`GPU_HOST` only)**

**Action:** On `GPU_HOST`, install the NVIDIA CUDA Toolkit 12.2 and verify with `nvidia-smi`.

### **Step 1.3: Python Environment Setup (Both Hosts)**

**Action:** On `CLIENT_HOST` and `GPU_HOST`, create a consistent Python virtual environment and install the required packages from a `requirements.txt` file.
1.  Create and activate the environment:
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```
2.  `requirements.txt` content:
    ```
    torch==2.1.0
    transformers==4.30.0
    accelerate==0.21.0
    ```
3.  Install packages:
    ```bash
    pip install -r requirements.txt
    ```

### **Step 1.4: Network Verification (From `CLIENT_HOST`)**

**Action:** Verify the network link performance.
1.  On `GPU_HOST`, start `iperf3 -s`.
2.  On `CLIENT_HOST`, run `iperf3 -c <GPU_IP> -P 4 -t 30`.
    *(**Expected Outcome:** Throughput should be >= 80% of the NIC's nominal link speed.)*

---

## **Phase 2: Project Architecture and Implementation**

**Goal:** Define the structure and logic for a unified client/server application that can simulate all three experimental modes.

### **2.1: System Architecture Overview**

The experiment uses a single RPC server (`rpc_server.py`) on the `GPU_HOST` and a single client (`run_llm.py`) on the `CLIENT_HOST`. The client's `--mode` flag dictates the communication logic.

* **`LOCAL` Mode:**
    * `run_llm.py` runs on `GPU_HOST`. No network activity. Serves as the baseline.

* **`NAÏVE-REMOTE` Mode (RPC-based Simulation):**
    * The client connects to the RPC server.
    * To simulate statelessness, for **every single inference step**, the client sends the model's weights, input data, and the full KV cache state. The server performs the computation and returns the results, caching nothing. This mimics a system with no semantic awareness of data lifetime or reuse.

* **`\sys (Simulated)` Mode (RPC-based Simulation):**
    * The client connects to the same RPC server.
    * The client first makes a one-time call to load the weights on the server.
    * For subsequent steps (e.g., decode), it only sends the new token and a reference (ID) to the KV cache, which the server manages statefully. This mimics a system with semantic awareness.

### **2.2: File Specification: `rpc_server.py` (GPU Host Logic)**

**Action:** Create the `rpc_server.py` script. It will host a single `RemoteWorker` class with methods for both naive and semantic-aware simulations.

1.  **Imports:** `torch`, `transformers`, `torch.distributed.rpc`, `uuid`.
2.  **`RemoteWorker` Class:**
    * **`__init__(self, model_name)`:**
        * Initializes the model structure without weights: `self.model = AutoModelForCausalLM.from_config(config)`.
        * Moves the model skeleton to GPU: `self.model.to('cuda')`.
        * Initializes state for the semantic mode: `self.weights_loaded = False` and `self.kv_cache_store = {}`.
    * **`run_stateless_forward_remote(self, state_dict, input_ids, kv_cache=None)` Method (for `NAÏVE-REMOTE`):**
        * This method is entirely stateless.
        * Loads the received `state_dict` into `self.model` on every call.
        * Moves `input_ids` to the GPU.
        * If `kv_cache` is provided, moves it to the GPU.
        * Runs one forward pass: `outputs = self.model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True)`.
        * Returns the `outputs.logits` and the full `outputs.past_key_values`, both moved back to the CPU.
    * **`load_weights_remote(self, state_dict)` Method (for `\sys (Simulated)`):**
        * If weights are not yet loaded, loads the received `state_dict` and sets `self.weights_loaded = True`.
    * **`run_prefill_remote(self, token_ids)` Method (for `\sys (Simulated)`):**
        * Generates a unique `kv_cache_id`.
        * Runs a forward pass and stores the resulting `past_key_values` in `self.kv_cache_store` using the ID.
        * Returns `logits` (on CPU) and the `kv_cache_id`.
    * **`run_decode_step_remote(self, token_id, kv_cache_id)` Method (for `\sys (Simulated)`):**
        * Retrieves the KV cache using the `kv_cache_id`.
        * Runs one forward pass with the new token.
        * Updates the KV cache in the store.
        * Returns `logits` (on CPU) and the same `kv_cache_id`.
3.  **Main Execution Block:**
    * Initializes the RPC framework as a worker named `GPU_WORKER`.
    * Instantiates the `RemoteWorker` class.
    * Enters a loop to wait for RPC calls.

### **2.3: File Specification: `run_llm.py` (Client Logic)**

**Action:** Create the `run_llm.py` script.

1.  **Argument Parsing:** As defined previously (`--mode`, `--phase`, etc.).
2.  **`run_naive_mode(args)` Function (Client RPC Logic):**
    * Initializes RPC and gets a reference to the `RemoteWorker`.
    * Loads model `state_dict` into local CPU memory.
    * **Prefill Logic:** Calls `worker_rref.rpc_sync().run_stateless_forward_remote(state_dict, token_ids)`.
    * **Decode Logic:**
        1.  First, run the prefill step by calling `run_stateless_forward_remote`, getting back the initial logits and KV cache.
        2.  Loop 50 times. In each iteration:
            a. Get the next token locally from the logits.
            b. Call `run_stateless_forward_remote` again, this time passing the **full `state_dict`**, the new single token, and the **entire KV cache object** received from the previous step.
            c. Receive the new logits and the new, full KV cache for the next iteration.
3.  **`run_sys_simulated_mode(args)` Function (Client RPC Logic):**
    * Initializes RPC and gets a reference to the `RemoteWorker`.
    * **Prefill Logic:**
        1.  Makes a one-time call to `worker_rref.rpc_sync().load_weights_remote(state_dict)`.
        2.  Calls `worker_rref.rpc_sync().run_prefill_remote(token_ids)` to get logits and the `kv_cache_id`.
    * **Decode Logic:**
        1.  First, run the prefill remotely as above to get the `kv_cache_id`.
        2.  Loop 50 times. In each iteration:
            a. Call `worker_rref.rpc_sync().run_decode_step_remote(new_token_id, kv_cache_id)`, sending only the new token and the cache ID.
            b. Receive new logits and the same `kv_cache_id`.

---

## **Phase 3: Experiment Execution and Data Collection**

**Goal:** Run each mode for both phases and collect all metrics.

### **Step 3.1: Run `LOCAL` Mode (Baseline)**
**Action:** On `GPU_HOST`, run `run_llm.py --mode local` for both phases, collecting latency with `/usr/bin/time`.

### **Step 3.2: Run Remote Modes (`NAÏVE-REMOTE` and `\sys (Simulated)`)**
**Action:** Execute the following sequence.

1.  **On `GPU_HOST`**, start the unified RPC server in the background:
    ```bash
    python rpc_server.py &
    RPC_PID=$!
    ```
2.  **On `GPU_HOST`**, start the GPU utilization monitor:
    ```bash
    nvidia-smi dmon -s u -i 0 -f remote_gpu_log.csv &
    NVIDIA_SMI_PID=$!
    ```
3.  **On `CLIENT_HOST`**, run the `NAÏVE-REMOTE` tests:
    ```bash
    # Prefill
    sudo tcpdump -i <NIC> -w naive_prefill.pcap &
    /usr/bin/time -f "NAIVE_PREFILL_LATENCY:%e" python run_llm.py --mode naive --phase prefill --gpu_host_ip <GPU_IP>
    sudo pkill tcpdump
    # Decode
    sudo tcpdump -i <NIC> -w naive_decode.pcap &
    /usr/bin/time -f "NAIVE_DECODE_LATENCY:%e" python run_llm.py --mode naive --phase decode --gpu_host_ip <GPU_IP>
    sudo pkill tcpdump
    ```
4.  **On `CLIENT_HOST`**, run the `\sys (Simulated)` tests:
    ```bash
    # Prefill
    sudo tcpdump -i <NIC> -w sys_sim_prefill.pcap &
    /usr/bin/time -f "SYS_SIM_PREFILL_LATENCY:%e" python run_llm.py --mode sys_simulated --phase prefill --gpu_host_ip <GPU_IP>
    sudo pkill tcpdump
    # Decode
    sudo tcpdump -i <NIC> -w sys_sim_decode.pcap &
    /usr/bin/time -f "SYS_SIM_DECODE_LATENCY:%e" python run_llm.py --mode sys_simulated --phase decode --gpu_host_ip <GPU_IP>
    sudo pkill tcpdump
    ```
5.  **On `GPU_HOST`**, stop the background processes:
    ```bash
    kill $NVIDIA_SMI_PID
    kill $RPC_PID
    ```

---

## **Phase 4: Data Analysis and Reporting**

**Goal:** Process the collected raw data into a final results table.

### **Step 4.1: Process Network Data**

**Action:** On `CLIENT_HOST`, use `capinfos` to extract the total bytes transferred from each `.pcap` file.
```bash
echo "NAIVE_PREFILL_BYTES: $(capinfos -c naive_prefill.pcap)"
echo "NAIVE_DECODE_BYTES: $(capinfos -c naive_decode.pcap)"
echo "SYS_SIM_PREFILL_BYTES: $(capinfos -c sys_simulated_prefill.pcap)"
echo "SYS_SIM_DECODE_BYTES: $(capinfos -c sys_simulated_decode.pcap)"
```

### **Step 4.2: Process GPU Utilization Data**

**Action:** On `GPU_HOST`, create a Python script `parse_dmon.py` to calculate the average GPU utilization from the CSV logs.
* *(This script should read a `dmon` CSV file, find the 'sm' column (SM utilization), and calculate the average value for the duration of the run.)*
1.  Execute the parsing script:
    ```bash
    python parse_dmon.py naive_gpu_log.csv
    python parse_dmon.py sys_simulated_gpu_log.csv
    ```

### **Step 4.3: Compile Final Results Table**

**Action:** Consolidate all collected and processed data points into a final table.

| Phase   | Mode              | Latency (s) | Network Bytes Transferred | Avg. GPU Utilization (%) |
|---------|-------------------|-------------|---------------------------|--------------------------|
| Prefill | LOCAL             | [Result]    | 0                         | [Result]                 |
| Prefill | NAÏVE-REMOTE      | [Result]    | [Result]                  | [Result]                 |
| Prefill | `\sys (Simulated)`  | [Result]    | [Result]                  | [Result]                 |
| Decode  | LOCAL             | [Result]    | 0                         | [Result]                 |
| Decode  | NAÏVE-REMOTE      | [Result]    | [Result]                  | [Result]                 |
| Decode  | `\sys (Simulated)`  | [Result]    | [Result]                  | [Result]                 |

*(**Note:** Repeat all runs in Phase 3 three times and report the mean and standard deviation for each metric to ensure statistical significance.)*