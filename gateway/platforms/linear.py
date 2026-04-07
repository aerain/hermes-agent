"""Linear issue tracking platform adapter.

Receives webhook events from Linear (new comments on issues) and sends
agent responses as issue comments. Each Linear issue acts as a conversation
thread.

Requires:
  - LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET (OAuth app credentials)
  - LINEAR_ACCESS_TOKEN (obtained via OAuth authorization_code flow)
  - LINEAR_WEBHOOK_SECRET (for webhook signature validation)
  - LINEAR_WEBHOOK_PORT (default: 8645)

Configuration in config.yaml:
  platforms:
    linear:
      enabled: true
      extra:
        webhook_port: 8645
        team_id: "..."
        require_mention: true
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_WEBHOOK_PORT = 8645
GRAPHQL_URL = "https://api.linear.app/graphql"
TOKEN_FILE_NAME = "linear_tokens.json"
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0


def check_linear_requirements() -> bool:
    """Check if Linear adapter dependencies are available."""
    if not AIOHTTP_AVAILABLE:
        return False
    if not os.getenv("LINEAR_CLIENT_ID"):
        return False
    if not os.getenv("LINEAR_CLIENT_SECRET"):
        return False
    return True


def _token_file_path() -> Path:
    """Path to the Linear token storage file."""
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / TOKEN_FILE_NAME


def _load_tokens() -> Dict[str, Any]:
    """Load stored OAuth tokens from disk."""
    path = _token_file_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not load Linear tokens: %s", e)
    return {}


def _save_tokens(tokens: Dict[str, Any]) -> None:
    """Persist OAuth tokens to disk."""
    path = _token_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tokens), encoding="utf-8")
    except Exception as e:
        logger.error("Could not save Linear tokens: %s", e)


class LinearAdapter(BasePlatformAdapter):
    """Linear issue tracking adapter using OAuth + GraphQL + webhooks."""

    platform = Platform.LINEAR

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINEAR)
        self._client_id = config.extra.get("client_id") or os.getenv("LINEAR_CLIENT_ID", "")
        self._client_secret = config.extra.get("client_secret") or os.getenv("LINEAR_CLIENT_SECRET", "")
        self._webhook_secret = config.extra.get("webhook_secret") or os.getenv("LINEAR_WEBHOOK_SECRET", "")
        self._webhook_port = int(
            config.extra.get("webhook_port")
            or os.getenv("LINEAR_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        self._webhook_base_url = config.extra.get("webhook_base_url") or os.getenv("LINEAR_WEBHOOK_BASE_URL", "")
        self._team_id = config.extra.get("team_id") or os.getenv("LINEAR_TEAM_ID", "")
        self._require_mention = config.extra.get("require_mention", True)
        self._bot_user_id = config.extra.get("bot_user_id") or os.getenv("LINEAR_BOT_USER_ID", "")

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._load_oauth_tokens()

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        self._processed_event_ids: Dict[str, float] = {}
        self._dedup_ttl: float = 300.0

    def _load_oauth_tokens(self) -> None:
        """Load OAuth tokens from disk storage."""
        stored = _load_tokens()
        if stored:
            self._access_token = stored.get("access_token")
            self._refresh_token = stored.get("refresh_token")
            self._token_expires_at = stored.get("expires_at", 0)

    def _store_tokens(self) -> None:
        """Persist OAuth tokens to disk."""
        tokens = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expires_at": self._token_expires_at,
            "client_id": self._client_id,
        }
        _save_tokens(tokens)

    async def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refreshing if needed."""
        if not self._access_token:
            return False

        if self._token_expires_at > 0 and time.time() >= self._token_expires_at - 300:
            return await self._refresh_access_token()

        return True

    async def _refresh_access_token(self) -> bool:
        """Refresh the OAuth access token using the refresh token."""
        if not self._refresh_token:
            logger.error("No refresh token available for Linear OAuth")
            return False

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                import urllib.request

                data = urllib.parse.urlencode({
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                }).encode()

                req = urllib.request.Request(
                    "https://api.linear.app/oauth/token",
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    token_data = json.loads(resp.read())

                self._access_token = token_data.get("access_token")
                self._refresh_token = token_data.get("refresh_token", self._refresh_token)
                self._token_expires_at = time.time() + token_data.get("expires_in", 86400)

                self._store_tokens()
                logger.info("Linear access token refreshed successfully")
                return True

            except Exception as e:
                logger.warning("Token refresh attempt %d failed: %s", attempt + 1, e)
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)

        logger.error("Failed to refresh Linear access token after %d attempts", MAX_RETRY_ATTEMPTS)
        return False

    async def _graphql_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GraphQL request to Linear API."""
        import urllib.request

        for attempt in range(MAX_RETRY_ATTEMPTS):
            if not await self._ensure_valid_token():
                raise Exception("No valid Linear access token")

            try:
                body = json.dumps({"query": query, "variables": variables or {}}).encode()
                req = urllib.request.Request(
                    GRAPHQL_URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._access_token}",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())

                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown GraphQL error")
                    logger.warning("[Linear] GraphQL error response: %s", result["errors"])
                    if "401" in str(result) or "Unauthorized" in error_msg:
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                    raise Exception(f"GraphQL error: {error_msg}")

                return result.get("data", {})

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""

                if e.code == 401:
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        if not await self._refresh_access_token():
                            raise Exception("Token refresh failed after 401")
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    raise Exception(f"HTTP Error {e.code}: {error_body}")

                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"HTTP Error {e.code}: {error_body}")
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    raise

    async def connect(self) -> bool:
        """Start the webhook server and connect to Linear."""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False

        if not self._client_id or not self._client_secret:
            logger.error("LINEAR_CLIENT_ID or LINEAR_CLIENT_SECRET not configured")
            return False

        try:
            self._app = web.Application()
            self._app.router.add_post("/webhook/linear", self._handle_webhook)
            self._app.router.add_get("/health", self._handle_health)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            self._site = web.TCPSite(self._runner, "0.0.0.0", self._webhook_port)
            await self._site.start()

            logger.info("[Linear] Webhook server started on port %s", self._webhook_port)
            return True

        except Exception as e:
            logger.error("[Linear] Failed to start webhook server: %s", e)
            return False

    async def disconnect(self) -> None:
        """Stop the webhook server."""
        try:
            if self._site:
                await self._site.stop()
            if self._runner:
                await self._runner.cleanup()
            logger.info("[Linear] Webhook server stopped")
        except Exception as e:
            logger.error("[Linear] Error during disconnect: %s", e)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "platform": "linear"})

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming Linear webhook."""
        try:
            body = await request.read()
            signature = request.headers.get("Linear-Signature", "")

            if self._webhook_secret and signature:
                if not self._verify_signature(body, signature):
                    logger.warning("[Linear] Invalid webhook signature")
                    return web.Response(status=401, text="Invalid signature")

            payload = json.loads(body)
            event_type = payload.get("type") or payload.get("event", "")

            event_id = payload.get("data", {}).get("id", "") or payload.get("webhookTimestamp", "")

            if event_id and self._is_duplicate_event(event_id):
                logger.debug("[Linear] Duplicate event %s, skipping", event_id)
                return web.json_response({"status": "ignored", "reason": "duplicate"})

            logger.info("[Linear] Received event: %s", event_type)

            if event_type == "IssueCommentCreate":
                await self._handle_comment_created(payload)
            elif event_type == "IssueCreate":
                await self._handle_issue_created(payload)
            elif event_type == "AppUserNotification":
                await self._handle_app_user_notification(payload)
            else:
                logger.debug("[Linear] Unhandled event type: %s", event_type)

            return web.json_response({"status": "ok"})

        except Exception as e:
            logger.error("[Linear] Webhook handling error: %s", e, exc_info=True)
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    def _verify_signature(self, body: bytes, signature_header: str) -> bool:
        """Verify Linear webhook HMAC signature.

        Linear sends a plain HMAC-SHA256 hex digest of the raw body.
        """
        if not signature_header:
            return False

        expected = hmac.new(
            self._webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        return secrets.compare_digest(expected, signature_header)

    def _is_duplicate_event(self, event_id: str) -> bool:
        """Check if an event has already been processed (dedup)."""
        now = time.time()
        cutoff = now - self._dedup_ttl
        self._processed_event_ids = {k: v for k, v in self._processed_event_ids.items() if v > cutoff}
        return event_id in self._processed_event_ids

    async def _handle_comment_created(self, payload: Dict[str, Any]) -> None:
        """Handle IssueCommentCreate webhook event."""
        data = payload.get("data", {})
        comment_body = data.get("body", "")
        comment_id = data.get("id", "")
        issue_id = data.get("issueId", "")

        if not comment_body or not issue_id:
            return

        actor = payload.get("actor", {})
        actor_id = actor.get("id", "") if isinstance(actor, dict) else ""

        if self._require_mention and self._bot_user_id:
            if actor_id == self._bot_user_id:
                logger.debug("[Linear] Skipping bot's own comment")
                return

        if self._team_id:
            issue = await self._get_issue(issue_id)
            if issue and issue.get("team", {}).get("id") != self._team_id:
                logger.debug("[Linear] Comment on issue not in monitored team, skipping")
                return

        issue_info = await self._get_issue(issue_id)

        source = self.build_source(
            chat_id=issue_id,
            chat_name=issue_info.get("title", issue_id) if issue_info else issue_id,
            chat_type="thread",
            user_id=actor_id,
            user_name=actor.get("name", "Unknown") if isinstance(actor, dict) else "Unknown",
            thread_id=issue_id,
        )

        event = MessageEvent(
            text=comment_body,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=comment_id,
            timestamp=datetime.now(timezone.utc),
        )

        self._processed_event_ids[comment_id] = time.time()
        await self.handle_message(event)

    async def _handle_issue_created(self, payload: Dict[str, Any]) -> None:
        """Handle IssueCreate webhook event."""
        data = payload.get("data", {})
        issue_id = data.get("id", "")
        title = data.get("title", "")
        description = data.get("description", "")

        if not issue_id:
            return

        if self._team_id:
            issue_team_id = data.get("teamId", "")
            if issue_team_id and issue_team_id != self._team_id:
                logger.debug("[Linear] New issue not in monitored team, skipping")
                return

        actor = payload.get("actor", {})
        actor_id = actor.get("id", "") if isinstance(actor, dict) else ""

        source = self.build_source(
            chat_id=issue_id,
            chat_name=title or issue_id,
            chat_type="thread",
            user_id=actor_id,
            user_name=actor.get("name", "Unknown") if isinstance(actor, dict) else "Unknown",
            thread_id=issue_id,
        )

        body = title
        if description:
            body = f"{title}\n\n{description}"

        event = MessageEvent(
            text=body,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=issue_id,
            timestamp=datetime.now(timezone.utc),
        )

        self._processed_event_ids[issue_id] = time.time()
        await self.handle_message(event)

    async def _handle_app_user_notification(self, payload: Dict[str, Any]) -> None:
        """Handle AppUserNotification webhook event (e.g. issueCommentMention)."""
        notification = payload.get("notification", {})
        action = payload.get("action", "")

        if action == "issueCommentMention":
            comment_data = notification.get("comment", {})
            comment_body = comment_data.get("body", "")
            comment_id = comment_data.get("id", "")
            issue_id = notification.get("issueId", "")

            if not comment_body or not issue_id:
                logger.debug("[Linear] AppUserNotification missing comment body or issueId")
                return

            actor = notification.get("actor", {})
            actor_id = actor.get("id", "") if isinstance(actor, dict) else ""

            if self._require_mention and self._bot_user_id:
                if actor_id == self._bot_user_id:
                    logger.debug("[Linear] Skipping bot's own mention")
                    return

            if self._team_id:
                issue_team_id = notification.get("issue", {}).get("teamId", "")
                if issue_team_id and issue_team_id != self._team_id:
                    logger.debug("[Linear] Comment on issue not in monitored team, skipping")
                    return

            issue_info = notification.get("issue", {})
            issue_title = issue_info.get("title", issue_id)

            source = self.build_source(
                chat_id=issue_id,
                chat_name=issue_title,
                chat_type="thread",
                user_id=actor_id,
                user_name=actor.get("name", "Unknown") if isinstance(actor, dict) else "Unknown",
                thread_id=issue_id,
            )

            event = MessageEvent(
                text=comment_body,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=comment_id,
                timestamp=datetime.now(timezone.utc),
            )

            self._processed_event_ids[comment_id] = time.time()
            await self.handle_message(event)
        else:
            logger.debug("[Linear] Unhandled AppUserNotification action: %s", action)

    async def _get_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Fetch issue details from Linear GraphQL API."""
        try:
            query = """
            query GetIssue($id: String!) {
                issue(id: $id) {
                    id
                    title
                    description
                    identifier
                    state { name }
                    team { id name }
                    assignee { id name displayName }
                }
            }
            """
            data = await self._graphql_request(query, {"id": issue_id})
            return data.get("issue")
        except Exception as e:
            logger.error("[Linear] Failed to fetch issue %s: %s", issue_id, e)
            return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message as a comment on a Linear issue."""
        try:
            mutation = """
            mutation CreateIssueComment($issueId: String!, $body: String!) {
                commentCreate(input: {issueId: $issueId, body: $body}) {
                    success
                    comment {
                        id
                        body
                    }
                }
            }
            """
            variables = {"issueId": chat_id, "body": content}
            result = await self._graphql_request(mutation, variables)

            comment_data = result.get("commentCreate", {})
            if not comment_data.get("success"):
                return SendResult(success=False, error="Failed to create comment")

            comment_id = comment_data.get("comment", {}).get("id", "")
            return SendResult(success=True, message_id=comment_id)

        except Exception as e:
            logger.error("[Linear] Failed to send comment on issue %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image as a comment on a Linear issue (as a URL attachment)."""
        content = f"![image]({image_url})"
        if caption:
            content = f"{caption}\n\n{content}"
        return await self.send(chat_id, content, reply_to, metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a video as a comment on a Linear issue (as a URL attachment)."""
        content = f"[video]({video_path})"
        if caption:
            content = f"{caption}\n\n{content}"
        return await self.send(chat_id, content, reply_to, metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a document as a comment on a Linear issue (as a URL attachment)."""
        content = f"[document]({file_path})"
        if file_name:
            content = f"**{file_name}**: {content}"
        if caption:
            content = f"{caption}\n\n{content}"
        return await self.send(chat_id, content, reply_to, metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Linear issue."""
        try:
            issue = await self._get_issue(chat_id)
            if issue:
                return {
                    "name": issue.get("title", chat_id),
                    "type": "thread",
                    "identifier": issue.get("identifier"),
                    "state": issue.get("state", {}).get("name") if isinstance(issue.get("state"), dict) else issue.get("state"),
                    "team": issue.get("team", {}).get("name") if isinstance(issue.get("team"), dict) else issue.get("team"),
                }
            return {"name": chat_id, "type": "thread"}
        except Exception as e:
            logger.error("[Linear] Failed to get chat info for %s: %s", chat_id, e)
            return {"name": chat_id, "type": "thread", "error": str(e)}

    def format_message(self, content: str) -> str:
        """Format message content for Linear (handle Markdown)."""
        return content
