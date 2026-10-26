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
    token: str = typer.Option(None, envvar="DISTLLM_TOKEN", help="Bearer token required from clients (and workers, unless --worker-token)."),
    worker_token: str = typer.Option(None, envvar="DISTLLM_WORKER_TOKEN", help="Separate bearer token for workers."),
    host: str = typer.Option("0.0.0.0", help="Interface to bind."),
):
    """Starts the distllm Relay server."""
    typer.echo(f"Starting Relay on {host}:{port}...")
    if token or worker_token:
        typer.echo("Token authentication enabled.")
    if simulate_worker_rtt_ms or simulate_client_rtt_ms:
        typer.echo(f"Simulating RTT: workers {simulate_worker_rtt_ms} ms, clients {simulate_client_rtt_ms} ms")
    start_relay(
        port=port, worker_rtt_ms=simulate_worker_rtt_ms, client_rtt_ms=simulate_client_rtt_ms,
        host=host, token=token, worker_token=worker_token,
    )

@app.command()
def worker(
    role: str, 
    relay_url: str = "http://localhost:8000", 
    node_id: str = None,
    token: str = typer.Option(None, envvar="DISTLLM_WORKER_TOKEN", help="Worker token (falls back to $DISTLLM_TOKEN)."),
):
    """Starts a distllm worker node (via CLI)."""
    typer.echo(f"Starting worker for role: {role}...")
    # This is a generic worker that just prints what it receives
    # In practice, users use the decorator, but this is a useful healthcheck.
    async def echo_worker(payload):
        print(f"Received payload: {payload}")
        return {"status": "echo", "received": payload}
    
    worker_inst = Worker(role=role, relay_url=relay_url, node_id=node_id, token=token)
    asyncio.run(worker_inst.run(echo_worker))

if __name__ == "__main__":
    app()
