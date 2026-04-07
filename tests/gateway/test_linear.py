"""Tests for Linear issue tracking platform adapter.

Covers:
- check_linear_requirements() - aiohttp availability, env vars
- OAuth token storage (load/save)
- GraphQL request helper
- Webhook signature verification
- Duplicate event dedup
- IssueCommentCreate / IssueCreate webhook handling via HTTP
- send() - issueCommentCreate mutation
- get_chat_info() - issue query
- Platform config in gateway/config.py
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**extra):
    """Build a PlatformConfig suitable for LinearAdapter."""
    return PlatformConfig(
        enabled=True,
        extra={
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "webhook_secret": "test-webhook-secret",
            "webhook_port": 0,  # OS will pick free port
            "team_id": "team-123",
            "require_mention": True,
            "bot_user_id": "bot-user-456",
            **extra,
        },
    )


def _make_adapter(**kwargs):
    """Create a LinearAdapter with sensible defaults for testing."""
    config = _make_config(**kwargs)
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(config)
    return adapter


def _linear_signature(body: bytes, secret: str) -> str:
    """Compute Linear webhook HMAC-SHA256 signature (plain hex digest)."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _make_comment_payload(
    comment_id="comment-abc",
    issue_id="issue-xyz",
    body="Hello from user",
    actor_id="user-123",
    actor_name="Test User",
):
    """Build a minimal IssueCommentCreate webhook payload."""
    return {
        "type": "IssueCommentCreate",
        "data": {
            "id": comment_id,
            "issueId": issue_id,
            "body": body,
        },
        "actor": {
            "id": actor_id,
            "name": actor_name,
        },
        "webhookTimestamp": int(time.time() * 1000),
    }


def _make_issue_payload(
    issue_id="issue-new",
    title="New Issue",
    description="Issue description",
    team_id="team-123",
    actor_id="user-123",
    actor_name="Test User",
):
    """Build a minimal IssueCreate webhook payload."""
    return {
        "type": "IssueCreate",
        "data": {
            "id": issue_id,
            "title": title,
            "description": description,
            "teamId": team_id,
        },
        "actor": {
            "id": actor_id,
            "name": actor_name,
        },
        "webhookTimestamp": int(time.time() * 1000),
    }


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestLinearRequirements:
    def test_returns_false_when_aiohttp_missing(self, monkeypatch):
        monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
        monkeypatch.delenv("LINEAR_CLIENT_SECRET", raising=False)
        with patch("gateway.platforms.linear.AIOHTTP_AVAILABLE", False):
            from gateway.platforms.linear import check_linear_requirements
            assert check_linear_requirements() is False

    def test_returns_false_when_client_id_missing(self, monkeypatch):
        monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "secret")
        from gateway.platforms.linear import check_linear_requirements

        assert check_linear_requirements() is False

    def test_returns_false_when_client_secret_missing(self, monkeypatch):
        monkeypatch.setenv("LINEAR_CLIENT_ID", "client-id")
        monkeypatch.delenv("LINEAR_CLIENT_SECRET", raising=False)
        from gateway.platforms.linear import check_linear_requirements

        assert check_linear_requirements() is False

    def test_returns_true_when_all_available(self, monkeypatch):
        monkeypatch.setenv("LINEAR_CLIENT_ID", "client-id")
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "client-secret")
        from gateway.platforms.linear import check_linear_requirements

        assert check_linear_requirements() is True


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------


class TestLinearPlatformEnum:
    def test_linear_enum_exists(self):
        assert Platform.LINEAR.value == "linear"

    def test_linear_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "linear" in platforms


class TestLinearConfigLoading:
    def test_apply_env_overrides_linear(self, monkeypatch):
        monkeypatch.setenv("LINEAR_CLIENT_ID", "env-client-id")
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "env-client-secret")
        monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "env-webhook-secret")
        monkeypatch.setenv("LINEAR_WEBHOOK_PORT", "8645")
        monkeypatch.setenv("LINEAR_TEAM_ID", "env-team-id")
        monkeypatch.setenv("LINEAR_BOT_USER_ID", "env-bot-user-id")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.LINEAR in config.platforms
        lc = config.platforms[Platform.LINEAR]
        assert lc.enabled is True
        assert lc.extra["client_id"] == "env-client-id"
        assert lc.extra["client_secret"] == "env-client-secret"
        assert lc.extra["webhook_secret"] == "env-webhook-secret"
        assert lc.extra["webhook_port"] == 8645
        assert lc.extra["team_id"] == "env-team-id"
        assert lc.extra["bot_user_id"] == "env-bot-user-id"

    def test_linear_not_loaded_without_client_id(self, monkeypatch):
        monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "secret")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.LINEAR not in config.platforms

    def test_connected_platforms_includes_linear(self, monkeypatch):
        monkeypatch.setenv("LINEAR_CLIENT_ID", "client-id")
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "client-secret")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert Platform.LINEAR in connected


# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------


class TestLinearAdapterInit:
    def test_init_parses_config(self):
        adapter = _make_adapter()
        assert adapter._client_id == "test-client-id"
        assert adapter._client_secret == "test-client-secret"
        assert adapter._webhook_secret == "test-webhook-secret"
        assert adapter._team_id == "team-123"
        assert adapter._require_mention is True
        assert adapter._bot_user_id == "bot-user-456"

    def test_init_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("LINEAR_CLIENT_ID", "env-client-id")
        monkeypatch.setenv("LINEAR_CLIENT_SECRET", "env-client-secret")
        monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "env-webhook-secret")
        monkeypatch.setenv("LINEAR_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("LINEAR_TEAM_ID", "env-team")
        monkeypatch.setenv("LINEAR_BOT_USER_ID", "env-bot")

        from gateway.platforms.linear import LinearAdapter

        config = PlatformConfig(enabled=True)
        adapter = LinearAdapter(config)

        assert adapter._client_id == "env-client-id"
        assert adapter._client_secret == "env-client-secret"
        assert adapter._webhook_secret == "env-webhook-secret"
        assert adapter._webhook_port == 9999
        assert adapter._team_id == "env-team"
        assert adapter._bot_user_id == "env-bot"


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------


class TestLinearTokenStorage:
    def test_load_tokens_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        tokens_file = tmp_path / "linear_tokens.json"
        tokens_file.write_text(
            json.dumps(
                {
                    "access_token": "stored-access-token",
                    "refresh_token": "stored-refresh-token",
                    "expires_at": 9999999999.0,
                    "client_id": "test-client-id",
                }
            ),
            encoding="utf-8",
        )

        from gateway.platforms.linear import _load_tokens

        tokens = _load_tokens()
        assert tokens["access_token"] == "stored-access-token"
        assert tokens["refresh_token"] == "stored-refresh-token"

    def test_save_tokens_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.platforms.linear import _save_tokens

        tokens = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_at": time.time() + 86400,
            "client_id": "test-client-id",
        }
        _save_tokens(tokens)

        tokens_file = tmp_path / "linear_tokens.json"
        loaded = json.loads(tokens_file.read_text(encoding="utf-8"))
        assert loaded["access_token"] == "new-access-token"
        assert loaded["refresh_token"] == "new-refresh-token"


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


class TestLinearSignatureVerification:
    def test_verify_valid_signature(self):
        adapter = _make_adapter()
        body = b'{"type": "IssueCommentCreate", "data": {}}'
        secret = "test-webhook-secret"
        sig = _linear_signature(body, secret)

        assert adapter._verify_signature(body, sig) is True

    def test_verify_invalid_signature(self):
        adapter = _make_adapter()
        body = b'{"type": "IssueCommentCreate", "data": {}}'

        assert adapter._verify_signature(body, "invalid-signature") is False

    def test_verify_tampered_body(self):
        adapter = _make_adapter()
        original_body = b'{"type": "IssueCommentCreate"}'
        tampered_body = b'{"type": "IssueCommentCreate", "data": {"body": "hacked"}}'
        secret = "test-webhook-secret"
        sig = _linear_signature(original_body, secret)

        assert adapter._verify_signature(tampered_body, sig) is False


# ---------------------------------------------------------------------------
# Duplicate event dedup
# ---------------------------------------------------------------------------


class TestLinearDuplicateDetection:
    def test_new_event_not_duplicate(self):
        adapter = _make_adapter()
        assert adapter._is_duplicate_event("event-new") is False

    def test_same_event_within_ttl_is_duplicate(self):
        adapter = _make_adapter()
        adapter._processed_event_ids["event-dup"] = time.time()
        assert adapter._is_duplicate_event("event-dup") is True

    def test_old_event_beyond_ttl_not_duplicate(self):
        adapter = _make_adapter()
        adapter._processed_event_ids["event-old"] = time.time() - 400
        assert adapter._is_duplicate_event("event-old") is False


# ---------------------------------------------------------------------------
# Webhook HTTP handler (via TestClient - same pattern as webhook adapter tests)
# ---------------------------------------------------------------------------


class TestLinearWebhookHandler:
    @pytest.mark.asyncio
    async def test_webhook_health_endpoint(self):
        adapter = _make_adapter()
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_get("/health", adapter._handle_health)
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "linear"

    @pytest.mark.asyncio
    async def test_webhook_issue_comment_routes_to_handler(self):
        adapter = _make_adapter()
        captured_event: list = []

        async def capture_message(event):
            captured_event.append(event)

        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._get_issue = AsyncMock(
            return_value={"title": "Test Issue", "team": {"id": "team-123"}}
        )
        adapter.handle_message = MagicMock(side_effect=capture_message)

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        payload = _make_comment_payload(
            comment_id="comment-1",
            issue_id="issue-1",
            body="What is the status?",
            actor_id="user-1",
            actor_name="Alice",
        )
        body = json.dumps(payload).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/linear",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": _linear_signature(body, "test-webhook-secret"),
                },
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"

        assert len(captured_event) == 1
        event = captured_event[0]
        assert event.text == "What is the status?"
        assert event.source.chat_id == "issue-1"
        assert event.source.user_id == "user-1"
        assert event.message_id == "comment-1"

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature_rejected(self):
        adapter = _make_adapter()
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        payload = _make_comment_payload()
        body = json.dumps(payload).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/linear",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": "invalid-signature",
                },
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_webhook_duplicate_event_returns_ignored(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._get_issue = AsyncMock(return_value={"title": "Issue", "team": {"id": "team-123"}})
        adapter.handle_message = AsyncMock()

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        async with TestClient(TestServer(app)) as cli:
            payload1 = _make_comment_payload(comment_id="dup-comment", issue_id="issue-1")
            body1 = json.dumps(payload1).encode()
            sig1 = _linear_signature(body1, "test-webhook-secret")

            resp1 = await cli.post(
                "/webhook/linear",
                data=body1,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": sig1,
                },
            )
            assert resp1.status == 200

            payload2 = _make_comment_payload(comment_id="dup-comment", issue_id="issue-1")
            body2 = json.dumps(payload2).encode()
            sig2 = _linear_signature(body2, "test-webhook-secret")

            resp2 = await cli.post(
                "/webhook/linear",
                data=body2,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": sig2,
                },
            )
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["status"] == "ignored"
            assert data2["reason"] == "duplicate"

    @pytest.mark.asyncio
    async def test_webhook_bot_own_comment_not_routed(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._get_issue = AsyncMock(return_value={"title": "Issue", "team": {"id": "team-123"}})
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = MagicMock(side_effect=capture)

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        payload = _make_comment_payload(
            comment_id="comment-bot",
            issue_id="issue-1",
            body="Reply from bot",
            actor_id="bot-user-456",
            actor_name="Hermes Bot",
        )
        body = json.dumps(payload).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/linear",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": _linear_signature(body, "test-webhook-secret"),
                },
            )
            assert resp.status == 200

        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_webhook_comment_on_wrong_team_not_routed(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._get_issue = AsyncMock(
            return_value={"title": "Other Team Issue", "team": {"id": "other-team"}}
        )
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = MagicMock(side_effect=capture)

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_post("/webhook/linear", adapter._handle_webhook)

        payload = _make_comment_payload(comment_id="comment-2", issue_id="issue-other")
        body = json.dumps(payload).encode()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/linear",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Linear-Signature": _linear_signature(body, "test-webhook-secret"),
                },
            )
            assert resp.status == 200

        assert len(captured) == 0


# ---------------------------------------------------------------------------
# send() method
# ---------------------------------------------------------------------------


class TestLinearSend:
    @pytest.mark.asyncio
    async def test_send_creates_comment_successfully(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._graphql_request = AsyncMock(
            return_value={
                "commentCreate": {
                    "success": True,
                    "comment": {"id": "new-comment-789", "body": "Here is the answer"},
                }
            }
        )

        result = await adapter.send("issue-xyz", "Here is the answer")

        assert result.success is True
        assert result.message_id == "new-comment-789"
        adapter._graphql_request.assert_called_once()
        call_args = adapter._graphql_request.call_args
        assert "commentCreate" in call_args[0][0]
        assert call_args[0][1]["issueId"] == "issue-xyz"
        assert call_args[0][1]["body"] == "Here is the answer"

    @pytest.mark.asyncio
    async def test_send_graphql_error_returns_failure(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._graphql_request = AsyncMock(
            return_value={"commentCreate": {"success": False}}
        )

        result = await adapter.send("issue-xyz", "Some response")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_send_without_valid_token_returns_failure(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=False)

        result = await adapter.send("issue-xyz", "Response")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_image_posts_markdown_link(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._graphql_request = AsyncMock(
            return_value={"commentCreate": {"success": True, "comment": {"id": "img-comment"}}}
        )

        result = await adapter.send_image("issue-1", "https://example.com/img.png", "Screenshot")

        assert result.success is True
        call_args = adapter._graphql_request.call_args
        body = call_args[0][1]["body"]
        assert "![image](https://example.com/img.png)" in body
        assert "Screenshot" in body


# ---------------------------------------------------------------------------
# get_chat_info() method
# ---------------------------------------------------------------------------


class TestLinearGetChatInfo:
    @pytest.mark.asyncio
    async def test_get_chat_info_returns_issue_details(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._graphql_request = AsyncMock(
            return_value={
                "issue": {
                    "id": "issue-xyz",
                    "title": "Critical Bug",
                    "identifier": "PROJ-42",
                    "state": {"name": "In Progress"},
                    "team": {"id": "team-123", "name": "Backend"},
                    "assignee": {"id": "user-1", "name": "Alice"},
                }
            }
        )

        info = await adapter.get_chat_info("issue-xyz")

        assert info["name"] == "Critical Bug"
        assert info["identifier"] == "PROJ-42"
        assert info["state"] == "In Progress"
        assert info["team"] == "Backend"

    @pytest.mark.asyncio
    async def test_get_chat_info_handles_missing_issue(self):
        adapter = _make_adapter()
        adapter._ensure_valid_token = AsyncMock(return_value=True)
        adapter._graphql_request = AsyncMock(return_value={"issue": None})

        info = await adapter.get_chat_info("nonexistent-issue")

        assert info["name"] == "nonexistent-issue"
        assert info["type"] == "thread"


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


class TestLinearTokenRefresh:
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.linear import LinearAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "refresh_token": "old-refresh-token",
            },
        )
        adapter = LinearAdapter(config)
        adapter._access_token = "old-token"
        adapter._refresh_token = "old-refresh-token"
        adapter._token_expires_at = time.time() - 100

        mock_token_response = json.dumps({
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 86400,
        }).encode()

        called_with = {}

        class FakeUrlopenResult:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return mock_token_response

        def fake_urlopen(req, timeout=None):
            called_with["req"] = req
            return FakeUrlopenResult()

        with patch("urllib.request.urlopen", fake_urlopen):
            result = await adapter._refresh_access_token()

        assert result is True
        assert adapter._access_token == "new-access-token"
        assert adapter._refresh_token == "new-refresh-token"

    @pytest.mark.asyncio
    async def test_refresh_token_failure_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.linear import LinearAdapter

        config = PlatformConfig(
            enabled=True,
            extra={
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "refresh_token": "bad-refresh-token",
            },
        )
        adapter = LinearAdapter(config)
        adapter._refresh_token = "bad-refresh-token"

        def fake_urlopen(req, timeout=None):
            raise Exception("Invalid refresh token")

        with patch("urllib.request.urlopen", fake_urlopen):
            result = await adapter._refresh_access_token()

        assert result is False


# ---------------------------------------------------------------------------
# format_message
# ---------------------------------------------------------------------------


class TestLinearFormatMessage:
    def test_format_message_passes_through(self):
        adapter = _make_adapter()
        result = adapter.format_message("Hello **world**")
        assert result == "Hello **world**"
