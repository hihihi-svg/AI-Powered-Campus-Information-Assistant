"""Operational Model Context Protocol (MCP) Server & Client for RVCE Campus Gmail & RAG.

Implements the official MCP JSON-RPC 2.0 protocol (`initialize`, `notifications/initialized`,
`ping`, `tools/list`, `tools/call`) over both in-process and stdio transports.

Tools exposed:
  - ``gmail_auth_status``: Verify OAuth credentials, token validity, and Gmail API connectivity.
  - ``gmail_list_messages``: List Gmail message IDs matching an RVCE query.
  - ``gmail_get_message``: Retrieve and parse a Gmail message (headers, body, ISO dates, attachments).
  - ``gmail_get_attachment``: Retrieve and extract PDF attachment content from a Gmail message.
  - ``gmail_sync_notices``: Run incremental Gmail notice ingestion and atomic index update.
  - ``campus_retrieve``: Perform date-aware hybrid/TF-IDF retrieval over the indexed campus corpus.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("campus_pipeline.mcp")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    candidate = REPO_ROOT / p
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate if not p.is_absolute() else p)

MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_SERVER_NAME = "rvce-campus-gmail-mcp"
MCP_SERVER_VERSION = "1.0.0"


def _scrub_sensitive(text: Any) -> str:
    """Redact any credentials, tokens, or secret strings before logging or returning."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***REDACTED***", s)
    s = re.sub(r"hf_[A-Za-z0-9_\-]{6,}", "hf_***REDACTED***", s)
    s = re.sub(r"AIza[A-Za-z0-9_\-]{6,}", "AIza***REDACTED***", s)
    s = re.sub(r"ya29\.[A-Za-z0-9_\-\.]+", "ya29.***REDACTED***", s)
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***REDACTED***", s, flags=re.IGNORECASE)
    return s


MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "gmail_auth_status",
        "description": "Verify Gmail OAuth credentials, cached token status, and live API connectivity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "credentials_path": {"type": "string", "default": "secrets/credentials.json"},
                "token_path": {"type": "string", "default": "secrets/token.json"},
                "verify_live": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "gmail_list_messages",
        "description": "List Gmail message IDs matching an RVCE search query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "default": "from:(*@rvce.edu.in OR *@rvei.edu.in OR *@vtu.ac.in) -label:SPAM -label:TRASH",
                },
                "max_results": {"type": "integer", "default": 25},
                "page_token": {"type": "string"},
            },
        },
    },
    {
        "name": "gmail_get_message",
        "description": "Fetch and parse a full Gmail message into structured campus notice metadata and body.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},
                "include_attachments": {"type": "boolean", "default": True},
            },
            "required": ["message_id"],
        },
    },
    {
        "name": "gmail_get_attachment",
        "description": "Download and extract text from a PDF attachment on a Gmail message.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},
                "attachment_id": {"type": "string"},
                "filename": {"type": "string", "default": "attachment.pdf"},
            },
            "required": ["message_id", "attachment_id"],
        },
    },
    {
        "name": "gmail_sync_notices",
        "description": "Execute incremental Gmail synchronization and atomically update the retrieval index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 7},
                "max_emails": {"type": "integer", "default": 50},
                "reindex": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "campus_retrieve",
        "description": "Search the indexed RVCE campus knowledge base with date-aware filtering and ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "strategy": {"type": "string", "default": "tfidf"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "gmail_search_messages",
        "description": "Query-time Gmail search: find messages relevant to a user question and return parsed email and attachment chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
                "include_attachments": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
]


class CampusGmailMCPServer:
    """Standards-compliant MCP JSON-RPC 2.0 Server for Gmail ingestion and campus retrieval."""

    def __init__(
        self,
        service: Optional[Any] = None,
        credentials_path: str = "secrets/credentials.json",
        token_path: str = "secrets/token.json",
        processed_path: str = "data/processed/campus_documents.jsonl",
        raw_path: str = "data/raw/campus_documents.jsonl",
        state_path: str = "data/raw/gmail_ingestion_state.json",
        manifest_path: str = "data/raw/collection_manifest.json",
        report_path: str = "reports/gmail_ingestion_report.json",
    ) -> None:
        self.service = service
        self.credentials_path = _resolve_repo_path(credentials_path)
        self.token_path = _resolve_repo_path(token_path)
        self.processed_path = _resolve_repo_path(processed_path)
        self.raw_path = _resolve_repo_path(raw_path)
        self.state_path = _resolve_repo_path(state_path)
        self.manifest_path = _resolve_repo_path(manifest_path)
        self.report_path = _resolve_repo_path(report_path)
        self.initialized: bool = False

    def _get_service(self) -> Any:
        if self.service is not None:
            return self.service
        from src.gmail_client import get_gmail_service

        self.service = get_gmail_service(
            credentials_path=self.credentials_path,
            token_path=self.token_path,
            allow_interactive_oauth=False,
        )
        return self.service

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 request or notification."""
        if not isinstance(request, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        # Notifications have no id and expect no response
        if method == "notifications/initialized":
            self.initialized = True
            return None

        if method == "initialize":
            self.initialized = True
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": MCP_SERVER_NAME,
                        "version": MCP_SERVER_VERSION,
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "pong"}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_SCHEMA},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            return self._dispatch_tool(req_id, tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _dispatch_tool(
        self,
        req_id: Any,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if tool_name == "gmail_auth_status":
                payload = self._tool_gmail_auth_status(arguments)
            elif tool_name == "gmail_list_messages":
                payload = self._tool_gmail_list_messages(arguments)
            elif tool_name == "gmail_get_message":
                payload = self._tool_gmail_get_message(arguments)
            elif tool_name == "gmail_get_attachment":
                payload = self._tool_gmail_get_attachment(arguments)
            elif tool_name == "gmail_sync_notices":
                payload = self._tool_gmail_sync_notices(arguments)
            elif tool_name == "campus_retrieve":
                payload = self._tool_campus_retrieve(arguments)
            elif tool_name == "gmail_search_messages":
                payload = self._tool_gmail_search_messages(arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }
        except Exception as exc:
            safe_err = _scrub_sensitive(f"{type(exc).__name__}: {exc}")
            logger.error("MCP tool '%s' failed: %s", tool_name, safe_err)
            err_payload = {
                "status": "error",
                "tool": tool_name,
                "error_type": type(exc).__name__,
                "message": "MCP tool execution failed.",
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err_payload, ensure_ascii=False)}],
                    "structuredContent": err_payload,
                    "isError": True,
                },
            }

    def _tool_gmail_auth_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        cred_path = args.get("credentials_path") or self.credentials_path
        tok_path = args.get("token_path") or self.token_path
        verify_live = bool(args.get("verify_live", False))

        has_credentials = os.path.exists(cred_path)
        has_token = os.path.exists(tok_path)
        token_valid = False

        if self.service is not None:
            token_valid = True
        elif has_token:
            try:
                from google.oauth2.credentials import Credentials
                from src.gmail_client import SCOPES

                creds = Credentials.from_authorized_user_file(tok_path, SCOPES)
                token_valid = bool(creds and (creds.valid or (creds.expired and creds.refresh_token)))
            except Exception as exc:
                logger.warning("Token check failed: %s", type(exc).__name__)

        live_connected = False
        if verify_live and (token_valid or self.service is not None):
            svc = self._get_service()
            profile = svc.users().getProfile(userId="me").execute()
            live_connected = bool(profile)

        return {
            "status": "ok",
            "credentials_present": has_credentials,
            "token_present": has_token,
            "authenticated": token_valid,
            "live_connected": live_connected,
        }

    def _tool_gmail_list_messages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.gmail_client import fetch_message_ids

        svc = self._get_service()
        query = args.get(
            "query",
            "from:(*@rvce.edu.in OR *@rvei.edu.in OR *@vtu.ac.in) -label:SPAM -label:TRASH",
        )
        max_results = int(args.get("max_results", 25))
        page_token = args.get("page_token")

        msg_ids, next_page = fetch_message_ids(
            service=svc,
            query=query,
            max_results=max_results,
            page_token=page_token,
            raise_on_error=True,
        )
        return {
            "status": "ok",
            "message_ids": msg_ids,
            "next_page_token": next_page,
            "count": len(msg_ids),
        }

    def _tool_gmail_get_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.gmail_client import fetch_attachment_bytes, fetch_message_full
        from src.gmail_parser import _find_attachments, parse_email_message

        message_id = str(args.get("message_id") or "").strip()
        if not message_id:
            raise ValueError("message_id is required")

        include_attachments = bool(args.get("include_attachments", True))
        svc = self._get_service()
        msg = fetch_message_full(svc, message_id)
        if not msg:
            raise RuntimeError(f"Failed to fetch Gmail message {message_id}")

        att_bytes_map: Dict[str, bytes] = {}
        payload = msg.get("payload", {})
        attachments_meta = _find_attachments(payload)
        if include_attachments:
            for att in attachments_meta:
                if att.get("filename", "").lower().endswith(".pdf"):
                    raw_b = fetch_attachment_bytes(svc, message_id, att["attachmentId"])
                    if raw_b:
                        att_bytes_map[att["attachmentId"]] = raw_b

        parent_rec, child_recs = parse_email_message(msg, att_bytes_map)
        clean_children = []
        for ch in child_recs:
            ch_copy = dict(ch)
            ch_copy.pop("raw_bytes", None)
            clean_children.append(ch_copy)

        return {
            "status": "ok",
            "message_id": message_id,
            "email_record": parent_rec,
            "attachment_records": clean_children,
            "attachments_found": [a.get("filename") for a in attachments_meta],
        }

    def _tool_gmail_get_attachment(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.extractors import PDFExtractor
        from src.gmail_client import fetch_attachment_bytes

        message_id = str(args.get("message_id") or "").strip()
        attachment_id = str(args.get("attachment_id") or "").strip()
        filename = str(args.get("filename") or "attachment.pdf").strip()
        if not message_id or not attachment_id:
            raise ValueError("message_id and attachment_id are required")

        svc = self._get_service()
        raw_bytes = fetch_attachment_bytes(svc, message_id, attachment_id)
        if not raw_bytes:
            raise RuntimeError(f"Could not download attachment {attachment_id} from message {message_id}")

        extractor = PDFExtractor()
        att_url = f"gmail://messages/{message_id}/attachments/{filename}"
        extracted = extractor.extract(raw_bytes=raw_bytes, url=att_url)
        content = (extracted.get("content") or "").strip()
        title = (extracted.get("title") or filename).strip()

        return {
            "status": "ok",
            "message_id": message_id,
            "attachment_id": attachment_id,
            "filename": filename,
            "title": title,
            "content": content,
            "byte_length": len(raw_bytes),
        }

    def _tool_gmail_sync_notices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.gmail_ingestion import run_gmail_ingestion

        days = int(args.get("days", 7))
        max_emails = int(args.get("max_emails", 50))
        reindex = bool(args.get("reindex", True))
        svc = self._get_service()

        stats = run_gmail_ingestion(
            service=svc,
            credentials_path=self.credentials_path,
            token_path=self.token_path,
            incremental=True,
            days=days,
            max_emails=max_emails,
            dry_run=False,
            reindex=reindex,
            state_path=self.state_path,
            raw_path=self.raw_path,
            processed_path=self.processed_path,
            manifest_path=self.manifest_path,
            report_path=self.report_path,
        )
        if stats.get("status") == "error":
            raise RuntimeError(stats.get("error") or "Gmail sync failed")
        return stats

    def _tool_campus_retrieve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from src.retrieval import enrich_with_companion_chunks, get_default_retriever, query_campus_knowledge
        from src.temporal import filter_and_rank_by_date, parse_temporal_query

        query = str(args.get("query") or "").strip()
        top_k = int(args.get("top_k", 5))
        strategy = str(args.get("strategy") or "tfidf")

        temporal_info = parse_temporal_query(query)
        search_q = temporal_info["subject_query"] if temporal_info["is_temporal"] else query
        candidates = query_campus_knowledge(
            query=search_q,
            top_k=max(top_k * 3, 10),
            score_threshold=0.03,
            dataset_path=self.processed_path,
            strategy=strategy,
        )
        retriever = get_default_retriever(self.processed_path)
        if candidates:
            candidates = enrich_with_companion_chunks(candidates, retriever=retriever, max_companions=2)
            candidates = filter_and_rank_by_date(candidates, temporal_info=temporal_info)

        return {
            "status": "ok",
            "query": query,
            "temporal_intent": temporal_info.get("intent"),
            "results": candidates[:top_k],
            "count": len(candidates[:top_k]),
        }

    def _tool_gmail_search_messages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search Gmail messages relevant to a user question and return parsed email/attachment chunks."""
        from src.gmail_client import fetch_attachment_bytes, fetch_message_full, fetch_message_ids
        from src.gmail_parser import _find_attachments, parse_email_message
        from src.retrieval import _chunk_to_result_dict, chunk_document

        raw_query = str(args.get("query") or "").strip()
        if not raw_query:
            return {"status": "ok", "query": "", "gmail_query": "", "messages_matched": 0, "chunks": [], "count": 0}

        max_results = max(1, min(int(args.get("max_results", 5)), 15))
        include_attachments = bool(args.get("include_attachments", True))

        # Extract meaningful search keywords from the user question
        filler = {
            "what", "is", "the", "are", "was", "were", "show", "tell", "me", "give", "find",
            "latest", "recent", "newest", "last", "most", "upcoming", "next", "when", "where",
            "who", "how", "which", "can", "could", "please", "about", "regarding", "related",
            "for", "in", "on", "at", "to", "from", "of", "and", "or", "with", "by", "rvce",
            "campus", "college", "email", "emails", "notice", "notices", "circular", "circulars",
            "update", "updates", "information", "details", "document", "documents",
        }
        tokens = [
            tok for tok in re.findall(r"[a-zA-Z0-9_\-\.]{2,}", raw_query.lower())
            if tok not in filler
        ]
        domain_filter = "from:(*@rvce.edu.in OR *@rvei.edu.in OR *@vtu.ac.in) -label:SPAM -label:TRASH"
        if tokens:
            keyword_expr = " ".join(tokens[:6])
            gmail_query = f"{domain_filter} ({keyword_expr})"
        else:
            gmail_query = domain_filter

        svc = self._get_service()
        msg_ids, _ = fetch_message_ids(
            service=svc,
            query=gmail_query,
            max_results=max_results,
            raise_on_error=True,
        )

        chunk_dicts: List[Dict[str, Any]] = []
        for msg_id in msg_ids[:max_results]:
            msg = fetch_message_full(svc, msg_id)
            if not msg:
                continue

            att_bytes_map: Dict[str, bytes] = {}
            payload = msg.get("payload", {})
            attachments_meta = _find_attachments(payload)
            if include_attachments and len(att_bytes_map) < 1:
                for att in attachments_meta:
                    att_name = str(att.get("filename") or "").lower()
                    if att_name.endswith(".pdf") and len(att_bytes_map) < 1:
                        raw_b = fetch_attachment_bytes(svc, msg_id, att["attachmentId"])
                        if raw_b:
                            att_bytes_map[att["attachmentId"]] = raw_b
                            break

            parent_rec, child_recs = parse_email_message(msg, att_bytes_map)
            records_to_chunk: List[Dict[str, Any]] = []
            if parent_rec and (parent_rec.get("content") or "").strip():
                records_to_chunk.append(parent_rec)

            parent_meta = {
                "sender": parent_rec.get("sender") if parent_rec else None,
                "sender_name": parent_rec.get("sender_name") if parent_rec else None,
                "received_at": parent_rec.get("received_at") or parent_rec.get("collected_at") if parent_rec else None,
                "document_date": parent_rec.get("document_date") or parent_rec.get("collected_at") if parent_rec else None,
                "email_subject": parent_rec.get("title") if parent_rec else None,
                "parent_email_body": (parent_rec.get("content") or "")[:600] if parent_rec else None,
            }
            for ch in child_recs:
                ch_copy = dict(ch)
                ch_copy.pop("raw_bytes", None)
                for k, v in parent_meta.items():
                    if v and not ch_copy.get(k):
                        ch_copy[k] = v
                if (ch_copy.get("content") or "").strip():
                    records_to_chunk.append(ch_copy)

            for rec in records_to_chunk:
                for c in chunk_document(rec):
                    c_dict = _chunk_to_result_dict(c, relevance_score=0.25)
                    c_dict["retrieved_via"] = "gmail_mcp"
                    chunk_dicts.append(c_dict)

        return {
            "status": "ok",
            "query": raw_query,
            "gmail_query": gmail_query,
            "messages_matched": len(msg_ids),
            "chunks": chunk_dicts,
            "count": len(chunk_dicts),
        }


class CampusGmailMCPClient:
    """Operational MCP Client that connects to CampusGmailMCPServer, discovers tools, and invokes them safely."""

    def __init__(self, server: Optional[CampusGmailMCPServer] = None) -> None:
        self.server = server or CampusGmailMCPServer()
        self._req_counter: int = 0
        self.connected: bool = False
        self.server_info: Dict[str, Any] = {}
        self.discovered_tools: List[str] = []
        self.last_error: Optional[str] = None

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def connect(self) -> Dict[str, Any]:
        """Perform MCP initialize handshake and discover available tools."""
        try:
            init_resp = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "rvce-rag-mcp-client", "version": "1.0.0"},
                    },
                }
            )
            if not init_resp or "error" in init_resp:
                raise RuntimeError("MCP initialize handshake failed")

            self.server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
            res = init_resp.get("result", {})
            self.server_info = res.get("serverInfo", {})
            self.connected = True

            tools_resp = self.list_tools()
            self.discovered_tools = [t["name"] for t in tools_resp]
            self.last_error = None
            return {
                "connected": True,
                "protocolVersion": res.get("protocolVersion"),
                "serverInfo": self.server_info,
                "tools": self.discovered_tools,
            }
        except Exception as exc:
            safe_err = _scrub_sensitive(f"{type(exc).__name__}: {exc}")
            logger.error("MCP client connection failed: %s", safe_err)
            self.connected = False
            self.last_error = safe_err
            return {"connected": False, "error": "MCP connection failed"}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Discover tools registered on the MCP server."""
        if not self.connected:
            self.connect()
        resp = self.server.handle_request(
            {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"}
        )
        if not resp or "error" in resp:
            return []
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke an MCP tool by name and return its structured response safely."""
        if not self.connected:
            conn = self.connect()
            if not conn.get("connected"):
                return {
                    "isError": True,
                    "error": "MCP server is not connected.",
                    "tool": name,
                }

        try:
            resp = self.server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                }
            )
            if not resp:
                self.last_error = f"Empty MCP response for tool {name}"
                return {"isError": True, "error": self.last_error, "tool": name}

            if "error" in resp:
                err_msg = _scrub_sensitive(resp["error"].get("message", "MCP protocol error"))
                self.last_error = err_msg
                logger.error("MCP protocol error calling '%s': %s", name, err_msg)
                return {"isError": True, "error": err_msg, "tool": name}

            result = resp.get("result", {})
            is_err = bool(result.get("isError", False))
            structured = result.get("structuredContent") or {}
            if is_err:
                self.last_error = structured.get("error_type") or "ToolError"
            else:
                self.last_error = None
            return {
                "isError": is_err,
                "data": structured,
                "content": result.get("content", []),
                "tool": name,
            }
        except Exception as exc:
            safe_err = _scrub_sensitive(f"{type(exc).__name__}: {exc}")
            logger.error("Unexpected MCP client error calling '%s': %s", name, safe_err)
            self.last_error = type(exc).__name__
            return {
                "isError": True,
                "error": "MCP tool call failed safely.",
                "tool": name,
            }

    def get_status(self) -> Dict[str, Any]:
        """Return operational status of the MCP integration."""
        if not self.connected:
            self.connect()
        auth_res = self.call_tool("gmail_auth_status", {"verify_live": False})
        auth_data = auth_res.get("data", {}) if not auth_res.get("isError") else {}
        return {
            "mcp_connected": self.connected,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "server_name": self.server_info.get("name", MCP_SERVER_NAME),
            "server_version": self.server_info.get("version", MCP_SERVER_VERSION),
            "discovered_tools": self.discovered_tools,
            "gmail_authenticated": bool(auth_data.get("authenticated")),
            "credentials_present": bool(auth_data.get("credentials_present")),
            "token_present": bool(auth_data.get("token_present")),
            "last_error": self.last_error,
        }

    def search_gmail_for_question(
        self,
        question: str,
        max_results: int = 5,
        include_attachments: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query-time Gmail MCP search: invoke gmail_search_messages and return matching chunks safely."""
        if not question or not question.strip():
            return []
        try:
            # Avoid blocking on live OAuth if neither an injected service nor a valid token is available
            if self.server.service is None:
                auth_res = self.call_tool("gmail_auth_status", {"verify_live": False})
                auth_data = auth_res.get("data", {}) if not auth_res.get("isError") else {}
                if not auth_data.get("authenticated"):
                    return []
            res = self.call_tool(
                "gmail_search_messages",
                {
                    "query": question,
                    "max_results": max_results,
                    "include_attachments": include_attachments,
                },
            )
            if res.get("isError"):
                return []
            data = res.get("data") or {}
            chunks = data.get("chunks")
            return list(chunks) if isinstance(chunks, list) else []
        except Exception as exc:
            logger.debug("Query-time Gmail MCP search skipped: %s", type(exc).__name__)
            return []


def run_stdio_server() -> None:
    """Run CampusGmailMCPServer over standard input/output (MCP stdio transport)."""
    server = CampusGmailMCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = server.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {type(exc).__name__}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()

