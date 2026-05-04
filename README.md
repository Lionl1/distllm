# distllm 🚀

A fault-tolerant distributed LLM inference framework designed for unstable compute nodes (like free Google Colab instances).

## Features
- **Magic API**: Simple decorator-based worker definition.
- **Relay Architecture**: Works behind NAT/Firewalls without public IPs.
- **Binary Transport**: High-speed tensor serialization via msgpack.
- **Self-Healing**: Automatic task recovery if workers crash mid-inference.

## Installation
Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/user/distllm.git
cd distllm

# Sync the environment and install dependencies
uv sync
```

## Quick Start

### 1. Start Relay (Central Server)
```bash
uv run distllm relay --port 8000
```

### 2. Define a Worker Node
Create a file named `worker.py`:

```python
from distllm import worker_node
import torch

@worker_node(role="gpu-worker", relay_url="http://your-relay-ip:8000")
def process_layers(payload):
    x = payload["hidden_states"]
    # ... your model logic ...
    return {"hidden_states": x * 2}

if __name__ == "__main__":
    process_layers.start()
```

Run the worker:
```bash
uv run python worker.py
```

### 3. Submit Tasks (Entry Node)
```python
import asyncio
import torch
from distllm import Cluster

async def main():
    cluster = Cluster("http://your-relay-ip:8000")
    task_id = await cluster.submit("gpu-worker", {"hidden_states": torch.randn(1, 10)})
    result = await cluster.wait_for(task_id)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Google Colab Workflow
To quickly set up `distllm` in a free Google Colab notebook:

```python
# 1. Install uv
!pip install uv

# 2. Install distllm and its dependencies using uv
!uv pip install distllm@git+https://github.com/user/distllm.git

# 3. Alternatively, if working on the source:
# !git clone https://github.com/user/distllm.git
# %cd distllm
# !uv pip install -e .
```

## Development
This project uses `uv` for dependency management.
- To add a new dependency: `uv add <package>`
- To run tests: `uv run pytest`
