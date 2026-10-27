#!/usr/bin/env bash
# Petals 2.2 private swarm on one Linux machine (CPU): blocks 0:16 + two replicas of 16:32.
# Needs Python 3.11 with petals==2.2.0.post1 and torch==2.1.2, dependencies resolved as of
# early 2024 (uv pip install --exclude-newer 2024-02-01, see .github/workflows/bench-petals.yml):
# with today's protobuf the servers start but every inference session is reset.
set -u
PY=${PY:-python}
M=HuggingFaceTB/SmolLM2-360M
cd "$(dirname "$0")"
COMMON="--device cpu --torch_dtype float32 --throughput 1 --skip_reachability_check --quant_type none"

server() {  # name, blocks, port, extra args
  $PY -m petals.cli.run_server $M --block_indices "$2" --host_maddrs "/ip4/127.0.0.1/tcp/$3" \
    --identity_path "$1.key" $COMMON "${@:4}" > "$1.log" 2>&1 &
  echo $! > "$1.pid"
}
wait_started() { for _ in $(seq 120); do grep -q "Started" "$1.log" && return; sleep 2; done; tail -20 "$1.log"; exit 1; }

server s1 0:16 31330 --new_swarm
wait_started s1
grep -oE "/ip4/127.0.0.1/tcp/31330/p2p/[A-Za-z0-9]+" s1.log | head -1 > peer.txt
server s2 16:32 31331 --initial_peers "$(cat peer.txt)"
server s3 16:32 31332 --initial_peers "$(cat peer.txt)"
wait_started s2; wait_started s3
sleep 15  # let the servers announce themselves in the DHT

timeout 600 $PY bench_petals.py speed || echo "RESULT speed: no answer within 600 s"
# We do not know which replica serves the session: kill each in turn (one run hits it).
timeout 300 $PY bench_petals.py kill s2.pid || echo "RESULT kill s2: no answer within 300 s"
# A fresh identity: the killed peer stays in the DHT until its record expires.
server s2b 16:32 31333 --initial_peers "$(cat peer.txt)"; wait_started s2b; sleep 15
timeout 300 $PY bench_petals.py kill s3.pid || echo "RESULT kill s3: no answer within 300 s"
kill $(cat s1.pid s2b.pid) 2>/dev/null
