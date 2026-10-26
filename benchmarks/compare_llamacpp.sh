#!/usr/bin/env bash
# distllm vs llama.cpp RPC on one machine: speed (loopback and 60 ms RTT) and node loss.
#
#   LLAMA_BIN=path/to/llama.cpp/build/bin MODEL_GGUF=smollm2-360m-f32.gguf \
#     bash benchmarks/compare_llamacpp.sh
#
# Build llama.cpp with -DGGML_RPC=ON (targets ggml-rpc-server llama-bench llama-cli) and
# convert HuggingFaceTB/SmolLM2-360M with convert_hf_to_gguf.py --outtype f32. The distllm
# side of the comparison is bench_pipeline.py with the same model, prompt and RTT.
set -u
B=${LLAMA_BIN:?set LLAMA_BIN}
M=${MODEL_GGUF:?set MODEL_GGUF}
HERE=$(cd "$(dirname "$0")" && pwd)
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

rpc_server() { "$B/ggml-rpc-server" -H 127.0.0.1 -p "$1" >/dev/null 2>&1 & LAST=$!; PIDS+=($LAST); }

echo "## single process"
"$B/llama-bench" -m "$M" -p 37 -n 48 -r 3 2>/dev/null | grep '^|'

rpc_server 50052; P1=$LAST
rpc_server 50053
sleep 2
echo "## 2 RPC servers, loopback"
"$B/llama-bench" -m "$M" -p 37 -n 48 -r 3 -ngl 99 --rpc 127.0.0.1:50052,127.0.0.1:50053 2>/dev/null | grep '^|'

python3 "$HERE/delay_proxy.py" 51052 50052 60 & PIDS+=($!)
python3 "$HERE/delay_proxy.py" 51053 50053 60 & PIDS+=($!)
sleep 1
echo "## 2 RPC servers, 60 ms RTT to each"
"$B/llama-bench" -m "$M" -p 37 -n 48 -r 2 -ngl 99 --rpc 127.0.0.1:51052,127.0.0.1:51053 2>/dev/null | grep '^|'

echo "## one RPC server killed (SIGKILL) 6 s into a long generation"
(sleep 6; kill -9 "$P1") &
"$B/llama-cli" -m "$M" -n 3000 --ignore-eos -ngl 99 --rpc 127.0.0.1:50052,127.0.0.1:50053 \
  -p "The history of the Roman Empire" -st </dev/null >/dev/null 2>"${TMPDIR:-/tmp}/llamacpp-kill.err"
echo "llama-cli exit code: $?"
grep -m1 -E "RPC|abort" "${TMPDIR:-/tmp}/llamacpp-kill.err"
