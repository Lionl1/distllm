import httpx
import pytest

from distllm import Cluster, RelayAuthError, Worker
from distllm.relay import app, set_auth_tokens
from distllm.transport import pack_payload


@pytest.fixture()
def relay_client():
    transport = httpx.ASGITransport(app=app)
    yield transport
    set_auth_tokens(None, None)


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_open_relay_when_no_token(relay_client):
    set_auth_tokens(None, None)
    async with httpx.AsyncClient(transport=relay_client, base_url="http://relay") as ac:
        resp = await ac.post("/register", params={"node_id": "n", "role": "r"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_single_token_protects_everything(relay_client):
    set_auth_tokens("s3cret")
    async with httpx.AsyncClient(transport=relay_client, base_url="http://relay") as ac:
        for headers in ({}, _bearer("wrong"), {"Authorization": "s3cret"}):
            assert (await ac.post("/register", params={"node_id": "n", "role": "r"}, headers=headers)).status_code == 401
            assert (await ac.post("/submit", content=pack_payload({}),
                                  params={"target_role": "r", "sender_id": "c"}, headers=headers)).status_code == 401
        ok = await ac.post("/register", params={"node_id": "n", "role": "r"}, headers=_bearer("s3cret"))
        submitted = await ac.post("/submit", content=pack_payload({}),
                                  params={"target_role": "r", "sender_id": "c"}, headers=_bearer("s3cret"))
    assert ok.status_code == 200
    assert submitted.status_code == 200


@pytest.mark.asyncio
async def test_worker_and_client_tokens_are_separate(relay_client):
    set_auth_tokens("client-token", "worker-token")
    async with httpx.AsyncClient(transport=relay_client, base_url="http://relay") as ac:
        # Токен воркера не даёт отправлять задачи и читать результаты...
        assert (await ac.post("/submit", content=pack_payload({}), params={"target_role": "r", "sender_id": "c"},
                              headers=_bearer("worker-token"))).status_code == 401
        assert (await ac.get("/get_result/x", headers=_bearer("worker-token"))).status_code == 401
        # ...а клиентский — регистрироваться воркером и забирать задачи.
        assert (await ac.post("/register", params={"node_id": "n", "role": "r"},
                              headers=_bearer("client-token"))).status_code == 401
        assert (await ac.get("/poll/r", params={"worker_id": "n"},
                             headers=_bearer("client-token"))).status_code == 401

        assert (await ac.post("/register", params={"node_id": "n", "role": "r"},
                              headers=_bearer("worker-token"))).status_code == 200
        assert (await ac.post("/submit", content=pack_payload({}), params={"target_role": "r", "sender_id": "c"},
                              headers=_bearer("client-token"))).status_code == 200


def test_clients_send_tokens_from_env(monkeypatch):
    monkeypatch.setenv("DISTLLM_TOKEN", "from-env")
    monkeypatch.delenv("DISTLLM_WORKER_TOKEN", raising=False)
    assert Cluster("http://relay").client.headers["authorization"] == "Bearer from-env"
    assert Worker("r", "http://relay").client.headers["authorization"] == "Bearer from-env"

    monkeypatch.setenv("DISTLLM_WORKER_TOKEN", "worker-env")
    assert Worker("r", "http://relay").client.headers["authorization"] == "Bearer worker-env"
    assert Worker("r", "http://relay", token="explicit").client.headers["authorization"] == "Bearer explicit"


def test_clients_send_the_first_token_of_a_spec(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTLLM_WORKER_TOKEN", raising=False)
    assert Cluster("http://relay", token="new, old").client.headers["authorization"] == "Bearer new"
    path = tmp_path / "tokens"
    path.write_text("# rotate here\nfrom-file\nolder\n")
    assert Worker("r", "http://relay", token=f"@{path}").client.headers["authorization"] == "Bearer from-file"


def test_no_token_no_header(monkeypatch):
    monkeypatch.delenv("DISTLLM_TOKEN", raising=False)
    monkeypatch.delenv("DISTLLM_WORKER_TOKEN", raising=False)
    assert "authorization" not in Cluster("http://relay").client.headers


@pytest.mark.asyncio
async def test_rejected_token_fails_fast(relay_client, monkeypatch):
    monkeypatch.delenv("DISTLLM_TOKEN", raising=False)
    monkeypatch.delenv("DISTLLM_WORKER_TOKEN", raising=False)
    set_auth_tokens("right")

    cluster = Cluster("http://relay", token="wrong")
    cluster.client = httpx.AsyncClient(transport=relay_client, headers=cluster.client.headers)
    with pytest.raises(RelayAuthError):
        await cluster.submit("r", {"x": 1})
    await cluster.close()

    worker = Worker("r", "http://relay", token="wrong")
    worker.client = httpx.AsyncClient(transport=relay_client, headers=worker.client.headers)
    with pytest.raises(RelayAuthError):
        # Без проверки register бесконечно повторял бы попытку раз в 5 секунд.
        await worker.register()
    await worker.client.aclose()
