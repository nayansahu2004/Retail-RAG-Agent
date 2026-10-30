# File: retail-rag-agent/tests/test_api.py
def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["agent_mode"] is False


def test_chat_rag_returns_answer_sources_and_request_id(client):
    response = client.post("/api/v1/chat", json={"question": "What is the return window?"})
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["mode"] == "rag" and body["session_id"]
    assert body["sources"] and body["sources"][0]["cited"] is True
    assert "total_ms" in body["timings_ms"]
    assert body["search_queries"] == ["What is the return window?"] and body["coverage"] == "full"


def test_agent_mode_disabled_without_key(client):
    response = client.post("/api/v1/chat", json={"question": "Suggest a laptop", "mode": "agent"})
    assert response.status_code == 503


def test_validation_error_on_empty_question(client):
    assert client.post("/api/v1/chat", json={"question": ""}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/api/v1/chat", json={"question": "free delivery?"})
    assert "rag_stage_latency_seconds" in client.get("/metrics").text