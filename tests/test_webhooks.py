"""Tests for the FastAPI webhook server."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

from mira.config import FilterConfig, MiraConfig, ReviewConfig
from mira.platforms.github.auth import GitHubAppAuth
from mira.platforms.server import create_app

WEBHOOK_SECRET = "test-secret-123"
BOT_NAME = "mira-bot"


@pytest.fixture
def app_auth() -> GitHubAppAuth:
    return GitHubAppAuth(app_id="12345", private_key="fake-key")


@pytest.fixture
def app(app_auth: GitHubAppAuth):  # noqa: ANN201
    return create_app(app_auth=app_auth, webhook_secret=WEBHOOK_SECRET, bot_name=BOT_NAME)


@pytest.fixture
async def client(app) -> AsyncClient:  # noqa: ANN001
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _sign(payload_bytes: bytes) -> str:
    """Compute the X-Hub-Signature-256 for a payload."""
    sig = hmac.new(WEBHOOK_SECRET.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _make_pr_payload(
    *,
    action: str = "opened",
    number: int = 42,
    body: str = "",
    labels: list[dict] | None = None,
) -> dict:
    """Build a pull_request webhook payload with all required fields."""
    return {
        "action": action,
        "installation": {"id": 1},
        "pull_request": {
            "number": number,
            "body": body,
            "labels": labels if labels is not None else [],
        },
        "repository": {
            "owner": {"login": "testowner"},
            "name": "testrepo",
        },
    }


def _pr_opened_payload() -> dict:
    return _make_pr_payload()


def _comment_payload(body: str, is_pr: bool = True) -> dict:
    issue: dict = {"number": 7}
    if is_pr:
        issue["pull_request"] = {"url": "https://api.github.com/repos/o/r/pulls/7"}
    return {
        "action": "created",
        "installation": {"id": 1},
        "comment": {"body": body, "user": {"login": "alice"}},
        "issue": issue,
        "repository": {
            "owner": {"login": "testowner"},
            "name": "testrepo",
        },
    }


def test_invalid_bot_name_rejected(app_auth: GitHubAppAuth) -> None:
    with pytest.raises(ValueError, match="Invalid bot_name"):
        create_app(app_auth=app_auth, webhook_secret=WEBHOOK_SECRET, bot_name="bad name!")


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_returns_503_when_postgres_unreachable(
    app_auth: GitHubAppAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")

    def failing_connect(_url: str) -> None:
        raise psycopg.OperationalError("connection refused")

    with patch("mira.db.postgres.connect", failing_connect):
        app = create_app(app_auth=app_auth, webhook_secret=WEBHOOK_SECRET, bot_name=BOT_NAME)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "database unavailable"


async def test_health_closes_postgres_probe_connection(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    closed: list[int] = []

    class _ProbeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: tuple = ()) -> None:
            return None

    class _ProbeConnection:
        def cursor(self) -> _ProbeCursor:
            return _ProbeCursor()

        def close(self) -> None:
            closed.append(1)

    with patch("mira.db.postgres.connect", side_effect=lambda _url: _ProbeConnection()):
        await client.get("/health")
        await client.get("/health")

    assert len(closed) == 2


async def test_webhook_acks_before_dispatch(app) -> None:  # noqa: ANN001
    """A verified event is acknowledged before dispatching starts.

    GitHub aborts deliveries that take too long to respond, so the route must
    write the response while the dispatcher is still pending — otherwise any
    latency in the dispatch path (GitHub API calls, event-loop contention)
    turns into a silently lost webhook.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_dispatch(event, payload, auth, bot_name, background_tasks):  # noqa: ANN001
        started.set()
        await release.wait()
        return "processing"

    payload_bytes = json.dumps(_comment_payload(f"@{BOT_NAME} review")).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/github/webhook",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"x-github-event", b"issue_comment"),
            (b"x-hub-signature-256", _sign(payload_bytes).encode()),
        ],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
    }
    body_sent = False
    messages: list[dict] = []

    async def receive():  # noqa: ANN202
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": payload_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message) -> None:  # noqa: ANN001
        messages.append(message)

    with patch("mira.platforms.server.dispatch_github_event", blocked_dispatch):
        run = asyncio.ensure_future(app(scope, receive, send))
        await started.wait()
        # The dispatcher is still blocked, yet the response must already be
        # fully written to the wire.
        body_messages = [m for m in messages if m["type"] == "http.response.body"]
        assert b'"accepted"' in b"".join(m.get("body", b"") for m in body_messages)
        release.set()
        await run


async def test_invalid_signature(client: AsyncClient) -> None:
    payload = json.dumps({"action": "opened"}).encode()
    resp = await client.post(
        "/webhook",
        content=payload,
        headers={
            "X-Hub-Signature-256": "sha256=invalid",
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 401


@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_opened_triggers_handler(mock_handler: AsyncMock, client: AsyncClient) -> None:
    payload_bytes = json.dumps(_pr_opened_payload()).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    # BackgroundTasks runs synchronously in test, so handler should have been called
    mock_handler.assert_awaited_once()


async def test_pr_closed_ignored(client: AsyncClient) -> None:
    payload = {"action": "closed", "installation": {"id": 1}}
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_comment_with_mention_triggers_handler(
    mock_handler: AsyncMock, client: AsyncClient
) -> None:
    payload = _comment_payload(f"@{BOT_NAME} why is this slow?")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_handler.assert_awaited_once()


async def test_comment_without_mention_ignored(client: AsyncClient) -> None:
    payload = _comment_payload("Just a regular comment")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


async def test_comment_on_issue_not_pr_ignored(client: AsyncClient) -> None:
    payload = _comment_payload(f"@{BOT_NAME} help", is_pr=False)
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


# ── pull_request_review_comment tests ────────────────────────────────────────


def _review_comment_payload(body: str, user: str = "alice") -> dict:
    return {
        "action": "created",
        "installation": {"id": 1},
        "comment": {
            "body": body,
            "node_id": "MDI0Ol_abc",
            "user": {"login": user},
        },
        "pull_request": {"number": 42},
        "repository": {
            "owner": {"login": "testowner"},
            "name": "testrepo",
        },
    }


@patch("mira.platforms.github.webhook.handle_thread_reject", new_callable=AsyncMock)
async def test_review_comment_reject_triggers_handler(
    mock_handler: AsyncMock, client: AsyncClient
) -> None:
    payload = _review_comment_payload(f"@{BOT_NAME} reject")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request_review_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_handler.assert_awaited_once()


async def test_review_comment_without_mention_ignored(client: AsyncClient) -> None:
    payload = _review_comment_payload("Just a regular reply")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request_review_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


async def test_review_comment_from_bot_self_ignored(client: AsyncClient) -> None:
    payload = _review_comment_payload(f"@{BOT_NAME} reject", user=f"{BOT_NAME}[bot]")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request_review_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


# ── pause / resume / ignore tests ────────────────────────────────────────────


@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_with_paused_label_returns_paused(
    mock_handler: AsyncMock, client: AsyncClient
) -> None:
    payload = _make_pr_payload(labels=[{"name": "mira-paused"}])
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_with_ignore_in_description(mock_handler: AsyncMock, client: AsyncClient) -> None:
    payload = _make_pr_payload(body="Some text\n@mira-bot ignore\nMore text")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.handle_pause_resume", new_callable=AsyncMock)
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_pause_comment_dispatches_pause_handler(
    mock_comment: AsyncMock, mock_pause: AsyncMock, client: AsyncClient
) -> None:
    payload = _comment_payload(f"@{BOT_NAME} pause")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_pause.assert_awaited_once()
    mock_comment.assert_not_awaited()


@patch("mira.platforms.github.webhook.handle_pause_resume", new_callable=AsyncMock)
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_resume_comment_dispatches_pause_handler(
    mock_comment: AsyncMock, mock_pause: AsyncMock, client: AsyncClient
) -> None:
    payload = _comment_payload(f"@{BOT_NAME} resume")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_pause.assert_awaited_once()
    mock_comment.assert_not_awaited()


@patch("mira.platforms.github.webhook.handle_pause_resume", new_callable=AsyncMock)
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_review_comment_still_dispatches_handle_comment(
    mock_comment: AsyncMock, mock_pause: AsyncMock, client: AsyncClient
) -> None:
    payload = _comment_payload(f"@{BOT_NAME} review")
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": "issue_comment",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    mock_comment.assert_awaited_once()


async def _post(client: AsyncClient, event: str, payload: dict) -> dict:
    """Helper to POST a webhook payload and return the JSON body."""
    payload_bytes = json.dumps(payload).encode()
    resp = await client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-Hub-Signature-256": _sign(payload_bytes),
            "X-GitHub-Event": event,
            "Content-Type": "application/json",
        },
    )
    return resp.json()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_opened_blocked_author_filtered(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    payload = _make_pr_payload()
    payload["sender"] = {"login": "dependabot[bot]"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_synchronize_skipped_when_review_on_synchronize_off(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(review=ReviewConfig(review_on_synchronize=False))
    payload = _make_pr_payload(action="synchronize")
    payload["sender"] = {"login": "alice"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_opened_still_reviewed_when_review_on_synchronize_off(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(review=ReviewConfig(review_on_synchronize=False))
    payload = _make_pr_payload(action="opened")
    payload["sender"] = {"login": "alice"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_awaited_once()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_synchronize_reviewed_by_default(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig()
    payload = _make_pr_payload(action="synchronize")
    payload["sender"] = {"login": "alice"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_awaited_once()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_opened_allowed_author_not_filtered(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    payload = _make_pr_payload()
    payload["sender"] = {"login": "alice"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_awaited_once()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_pull_request", new_callable=AsyncMock)
async def test_pr_opened_allowlist_filters_off_list(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(allowed_authors=["alice"]))
    payload = _make_pr_payload()
    payload["sender"] = {"login": "bob"}
    await _post(client, "pull_request", payload)
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_push_index", new_callable=AsyncMock)
async def test_push_blocked_author_filtered(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    payload = {
        "ref": "refs/heads/main",
        "sender": {"login": "dependabot[bot]"},
        "repository": {"default_branch": "main"},
        "installation": {"id": 1},
    }
    await _post(client, "push", payload)
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_comment_review_bypass_for_blocked_author(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    """Manual @mira-bot review bypasses the author filter."""
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    payload = _comment_payload(f"@{BOT_NAME} review")
    payload["comment"]["user"]["login"] = "dependabot[bot]"
    await _post(client, "issue_comment", payload)
    mock_handler.assert_awaited_once()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_comment_non_review_no_bypass(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    """Non-review commands do NOT bypass the author filter."""
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["alice"]))
    payload = _comment_payload(f"@{BOT_NAME} pause")
    await _post(client, "issue_comment", payload)
    mock_handler.assert_not_awaited()


@patch("mira.platforms.github.webhook.load_config")
@patch("mira.platforms.github.webhook.handle_comment", new_callable=AsyncMock)
async def test_comment_case_insensitive_review_bypass(
    mock_handler: AsyncMock, mock_load_config, client: AsyncClient
) -> None:
    """Case-insensitive @mira-bot Review bypasses the author filter."""
    mock_load_config.return_value = MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"]))
    payload = _comment_payload(f"@{BOT_NAME} Review")
    payload["comment"]["user"]["login"] = "dependabot[bot]"
    await _post(client, "issue_comment", payload)
    mock_handler.assert_awaited_once()
