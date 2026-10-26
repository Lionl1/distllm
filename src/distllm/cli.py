import typer
from .relay import start_relay
from .worker import Worker
import asyncio

app = typer.Typer(help="distllm: Distributed LLM Inference Framework")

@app.command()
def relay(
    port: int = 8000,
    simulate_worker_rtt_ms: float = typer.Option(0.0, help="Add artificial round-trip latency to worker requests (experiments only)."),
    simulate_client_rtt_ms: float = typer.Option(0.0, help="Add artificial round-trip latency to client requests (experiments only)."),
):
    """Starts the distllm Relay server."""
    typer.echo(f"Starting Relay on port {port}...")
    if simulate_worker_rtt_ms or simulate_client_rtt_ms:
        typer.echo(f"Simulating RTT: workers {simulate_worker_rtt_ms} ms, clients {simulate_client_rtt_ms} ms")
    start_relay(port=port, worker_rtt_ms=simulate_worker_rtt_ms, client_rtt_ms=simulate_client_rtt_ms)

@app.command()
def worker(
    role: str, 
    relay_url: str = "http://localhost:8000", 
    node_id: str = None
):
    """Starts a distllm worker node (via CLI)."""
    typer.echo(f"Starting worker for role: {role}...")
    # This is a generic worker that just prints what it receives
    # In practice, users use the decorator, but this is a useful healthcheck.
    async def echo_worker(payload):
        print(f"Received payload: {payload}")
        return {"status": "echo", "received": payload}
    
    worker_inst = Worker(role=role, relay_url=relay_url, node_id=node_id)
    asyncio.run(worker_inst.run(echo_worker))

if __name__ == "__main__":
    app()
