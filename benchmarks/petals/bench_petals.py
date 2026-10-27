"""Petals on a private swarm: speed, same tokens as HF, and what happens when a server dies.

Run by run.sh, which starts the servers (blocks 0:16 and two replicas of 16:32).
    python bench_petals.py speed            # 3 runs, 48 tokens
    python bench_petals.py kill PIDFILE     # SIGKILL that server 3 s into a 96-token run
"""
import os
import signal
import sys
import threading
import time

import torch
from petals import AutoDistributedModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-360M"
PROMPT = (  # the prompt of benchmarks/bench_pipeline.py
    "Distributed inference splits a large language model across several machines. "
    "Each machine keeps a slice of the transformer layers and passes hidden states "
    "to the next one. The main advantages of this approach are"
)
PEER = open(os.environ.get("PEER_FILE", "peer.txt")).read().strip()

mode = sys.argv[1]
n = 48 if mode == "speed" else 96
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(PROMPT, return_tensors="pt").input_ids
ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).generate(
    ids, max_new_tokens=n, do_sample=False
)[0, ids.shape[1]:].tolist()
model = AutoDistributedModelForCausalLM.from_pretrained(MODEL, initial_peers=[PEER], torch_dtype=torch.float32)
print(f"prompt {ids.shape[1]} tokens, generating {n}", flush=True)

print("warm-up...", flush=True)
model.generate(ids, max_new_tokens=2, do_sample=False)  # warm-up: routing, connections
print("warm-up done", flush=True)
for run in range(3 if mode == "speed" else 1):
    if mode == "kill":
        pid = int(open(sys.argv[2]).read())

        def kill():
            os.kill(pid, signal.SIGKILL)
            print(f"KILLED server {sys.argv[2]} (pid {pid}) at +3 s", flush=True)

        threading.Timer(3.0, kill).start()
    started = time.perf_counter()
    try:
        out = model.generate(ids, max_new_tokens=n, do_sample=False)[0, ids.shape[1]:].tolist()
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {mode}: FAILED after {time.perf_counter() - started:.1f} s: {type(e).__name__}: {e}"[:400])
        break
    seconds = time.perf_counter() - started
    print(f"RESULT {mode} run {run}: {seconds:.2f} s, {n / seconds:.2f} tok/s, same tokens as HF: {out == ref}", flush=True)
