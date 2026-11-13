"""Email platform adapter for the Hermes gateway: users talk to Hermes by sending email; IMAP (polled)
receives, SMTP sends. Configured via EMAIL_* env vars or ``platforms.email`` in config.yaml (see website docs).

------------------------------------------------------------------------------
LOCAL PATCH (dgxarley) — synced to upstream tag v0.23.1
(plugins/platforms/email/adapter.py, 61613 bytes, blob 49abaf5f6a). Must move in
lockstep with hermes.image_tag: the baseline calls base-class methods that only
exist from its own tag on. Re-sync procedure: HERMES_EMAIL_UPSTREAM.md.

Adds four behaviours that upstream lacks:

  1.  Two-stage IMAP folder lifecycle:
        INBOX  -- fetch -->  Hermes_Working  -- on_processing_complete() -->  Hermes_Done
      so anything left in Hermes_Working after a crash is visibly "interrupted".
      The move happens when the agent turn for that mail has finished: SUCCESS
      -> Done, FAILURE -> Failed; an interrupted/cancelled turn stays in Working.
      Not processed: rejected sender -> Rejected; error (exception, no handler,
      identity/route drop, attachments gone, retry bound hit) -> Failed; command
      handled inline -> Done. Mail the gateway refused because the sender's
      session was busy or its queue full stays in Working and is re-dispatched
      once the session is idle (bounded; on giving up -> Failed, or stays in
      Working when failed_folder is ""). Empty failed/rejected folder -> Done
      (except the give-up case above).
  2.  Sent-mail archival via IMAP APPEND (upstream only sends via SMTP).
  3.  Opt-in processing of pre-existing INBOX mail on startup (upstream marks
      everything already there as seen).
  4.  Opt-in quoting of the received email below the turn's final reply to it.

Knobs live in config.yaml under an explicit ``extra:`` block (an explicit
``extra:`` value wins over a bare platform key). The two quote knobs also accept
an env override (EMAIL_QUOTE_ORIGINAL / EMAIL_QUOTE_MAX_CHARS win):

  platforms.email.extra.working_folder     default "Hermes_Working"  ("" → INBOX→Done)
  platforms.email.extra.done_folder        default "Hermes_Done"     ("" → no moves at all)
  platforms.email.extra.failed_folder      default "Hermes_Failed"   ("" → done_folder)
  platforms.email.extra.rejected_folder    default "Hermes_Rejected" ("" → done_folder)
  platforms.email.extra.sent_folder        default "Sent"            ("" → no APPEND)
  platforms.email.extra.process_existing   default true              (upstream PR: false)
  platforms.email.extra.quote_original     default false
  platforms.email.extra.quote_max_chars    default 10000             (quote body only)
  platforms.email.extra.quote_header       default "On {date}, {name} <{address}> wrote:"

Patch sections (grep for the markers):

  [PATCH-1] module docstring (this block)
  [PATCH-2] __init__: new self._* attributes (+ lifecycle log line)
  [PATCH-3] helpers: module-level _open_imap_conn (twin of upstream's
            _connect_imap) + _imap_append_to_sent; EmailAdapter._ensure_folder,
            _imap_move, _search_message_id, _finalize_message, _append_to_sent
  [PATCH-4] _probe_imap(): folder ensure + process_existing ``elif`` between
            upstream's reconnect-restore and mark-all-seen branches
  [PATCH-5] _fetch_new_messages(): INBOX→Working MOVE after
            _parse_fetched_message(), only if done_folder AND working_folder AND
            a Message-ID are present; _mark_uid_consumed(): INBOX→Rejected MOVE for
            mail the pre-fetch header gate (upstream, since v0.21.6) rejects
  [PATCH-6] _dispatch_message(): try/finally around _sender_accepted + event
            build + handle_message; mail that no background turn owns
            (event._gateway_accepted is not True) finalizes out of Working at once
  [PATCH-12] completion-driven Done move: _EmailMessageEvent (records Message-IDs
            merged into a queued event via absorb_reply_expected, which both the
            pending merge and the text debounce call before the debounce
            overwrites the head's message_id), self._inflight
            (bounded Message-ID -> source folder), on_processing_start() (only so
            the runner fires hooks for queued follow-ups), on_processing_complete()
            (SUCCESS/FAILURE -> Done, CANCELLED -> stays), _sweep_inflight()
            (fallback for turns whose hook never fires); busy/queue-full refusal
            retry: self._retry, _deferrable(), _retry_note(), _retry_deferred(),
            send() swallows the gateway's QUEUE_FULL_REPLY
  [PATCH-7] _smtp_send(): APPEND to Sent (single sink for all adapter sends)
  [PATCH-8] _standalone_send(): APPEND to Sent via the shared helper (cron /
            `hermes send` path)
  [PATCH-9] retired (charset fix, native upstream since v2026.8.13)
  [PATCH-10] _imap_default_security(): port-derived IMAP security default
            (993 → tls, else starttls); upstream defaults to tls on every port,
            which breaks port-143 mailboxes. An explicit setting still wins.
  [PATCH-11] quoting: _format_quote_date() / _build_quote(); _remember_original()
            inside the [PATCH-6] try, only for msg_data["sender_granted"] (else
            _forget_sender_original()); _claim_quote() quotes only final-reply
            sends (metadata["notify"]), _claim_quote()/_settle_quote() wrap the
            single _smtp_send() call site. _standalone_send never quotes.

dgxarley-only deviations from upstream code paths: our own IMAP teardowns go
through _close_imap() (same fd-leak class as upstream #79889), and
_standalone_send's IMAP reads use the profile-scoped _get_secret().

Upstreaming (all OPEN; drop the matching sections once merged):
  - #28697 Sent-folder APPEND          [PATCH-3/7/8]
  - #28699 process_existing            [PATCH-4]
  - #28702 Working/Done lifecycle      [PATCH-3/4/5/6/12]
  - #113192 quote original             [PATCH-11]
[PATCH-10] stays dgxarley-only (per-user ``imap_security: starttls`` would make
it unnecessary).
------------------------------------------------------------------------------
"""

import asyncio
import email as email_lib
from contextlib import contextmanager, suppress
import imaplib
import logging
import os
import re
import smtplib
import socket
import ssl
import threading  # [PATCH-11] not imported upstream; _quote_lock guards claim/settle across executor threads
import time  # dgxarley: not imported upstream; _imap_append_to_sent needs time.time()
import uuid
from collections import OrderedDict  # [PATCH-11] not imported upstream; bounded per-Message-ID quote lookup
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.parser import BytesHeaderParser
from email.utils import formatdate, parseaddr, parsedate_to_datetime
from email import encoders
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import cancel_task
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.config import Platform, PlatformConfig
from utils import is_truthy_value
from gateway.platforms._shared import (
    get_scoped_secret as _get_secret,
    coerce_port,
    decode_json_list_literal,
    send_error,
)
from gateway.platforms.send_targets import StandaloneSendPolicy, regex_stripped_parser

logger = logging.getLogger(__name__)

_SECURITY_ALIASES = {
    "tls": "tls",
    "ssl": "tls",
    "implicit": "tls",
    "starttls": "starttls",
    "plain": "plain",
    "none": "plain",
}
# Automated senders (address substrings / bulk-mail headers) are silently ignored.
_NOREPLY_PATTERNS = (
    "noreply",
    "no-reply",
    "no_reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
    "bounce",
    "notifications@",
    "automated@",
    "auto-confirm",
    "auto-reply",
    "automailer",
)
_AUTOMATED_HEADERS = {
    "Auto-Submitted": lambda v: v.lower() != "no",
    "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}
MAX_MESSAGE_LENGTH = 50_000  # Gmail-safe max length per email body
SMTP_CONNECT_TIMEOUT = 30
_TRUTHY = {"true", "1", "yes"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# Charset labels seen in the wild that Python's codec registry doesn't know: "unknown-8bit"/"x-unknown" are
# RFC 1428 placeholders (QQ Mail emits them); gb2312/gbk map to the gb18030 superset so GBK extensions decode.
_CHARSET_ALIASES = {
    "unknown-8bit": "utf-8",
    "unknown": "utf-8",
    "x-unknown": "utf-8",
    "default": "utf-8",
    "ansi_x3.110-1983": "latin-1",
    "cp-850": "cp850",
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "ks_c_5601-1987": "cp949",
}
# Ordered (pattern, replacement) substitutions for _strip_html.
_HTML_SUBS = (
    (re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"),
    (re.compile(r"<p[^>]*>", re.IGNORECASE), "\n"),
    (re.compile(r"</p>", re.IGNORECASE), "\n"),
    (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"\n{3,}"), "\n\n"),
)
# ``display <bracketed>`` split for the _extract_email_address fallback (linear: neither part can match the other's delimiters).
_SINGLE_BRACKET_FROM_RE = re.compile(r'([^"<>]*)<([^<>\s]+)>\s*')
_COMMENT_RE = re.compile(r"\([^()]*\)")
# Longest From: value we parse. parseaddr is pure Python and superlinear on hostile input (~1s at 100KB, GIL held);
# a real mailbox plus display name stays far below this (RFC 5322 caps a line at 998 chars).
_MAX_FROM_LEN = 2048
# Only these headers are needed to reject a sender. Request one byte past the
# limit so a truncated preflight fails closed without downloading the MIME body.
_MAX_PREAUTH_HEADER_BYTES = 64 * 1024
_PREAUTH_FETCH = (
    f"(BODY.PEEK[HEADER.FIELDS (FROM AUTHENTICATION-RESULTS {' '.join(h.upper() for h in _AUTOMATED_HEADERS)})]"
    f"<0.{_MAX_PREAUTH_HEADER_BYTES + 1}>)"
)
# Authentication-Results clause head (``dmarc=pass``), matched only at the start of a clause.
_AUTH_METHOD_RE = re.compile(r"\s*(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
_NO_AUTH_RESULTS_REASON = "no Authentication-Results header"
_UNTRUSTED_AUTHSERV_REASON = "no Authentication-Results from trusted authserv-id"
_MISSING_AUTHSERV_REASON = "authserv-id is not configured; refusing to trust Authentication-Results"
_OPT_OUT_HINT = (
    "set platforms.email.require_authenticated_sender: false (or EMAIL_TRUST_FROM_HEADER=true) to accept the risk."
)
# Operator-fixable reasons a granted sender's mail fails authentication, and the fix each log line names.
_DROP_HINTS = {
    _NO_AUTH_RESULTS_REASON: " If your mail server does not stamp Authentication-Results, " + _OPT_OUT_HINT,
    _UNTRUSTED_AUTHSERV_REASON: " Check that platforms.email.authserv_id (EMAIL_AUTHSERV_ID) names your mail server.",
}
# A missing pin is account config, not one sender's mail, so connect() names its fix once per account.
_MISSING_AUTHSERV_HINT = (
    " Set EMAIL_AUTHSERV_ID (or platforms.email.authserv_id) to the receiving MTA's exact authserv-id, "
    "or " + _OPT_OUT_HINT
)
# One token of a clause: a property we read (``header.from=x``; the value may be or contain a quoted-string), or
# any other whitespace-delimited token consumed whole, so text inside quotes or other values is never read as a prop.
_QUOTED = r'"(?:[^"\\]|\\.)*"'
_AUTH_PROP_RE = re.compile(
    rf'(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*((?:{_QUOTED}|[^\s";])+)'
    rf'|(?:{_QUOTED}|[^\s"])+',
    re.IGNORECASE,
)
# [PATCH-11] Quoting the inbound mail under replies (opt-in, platforms.email.extra.quote_original / EMAIL_QUOTE_ORIGINAL).
_DEFAULT_QUOTE_HEADER = "On {date}, {name} <{address}> wrote:"
_DEFAULT_QUOTE_MAX_CHARS = 10_000
_QUOTE_TRUNCATED = "> [...]"
_QUOTE_LOOKUP_MAX = 500  # remembered originals (oldest evicted first)


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read."""
    return coerce_port(str(_get_secret(name, "")).strip() or default, default)


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read."""
    return is_truthy_value(raw, default=default) if (raw := str(_get_secret(name, "")).strip()) else default


def _normalize_security(value: Any, default: str = "tls") -> str:
    """Map to ``tls`` | ``starttls`` | ``plain``; unknown values warn and fall back to *default* (a typo never downgrades to plaintext)."""
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if raw and raw not in _SECURITY_ALIASES:
        logger.warning("Unknown email security mode %r; using %r", value, default)
    return _SECURITY_ALIASES.get(raw, default)


def _tls_context(verify: bool, host: str) -> ssl.SSLContext:
    """Verified context by default; unverified only when explicitly opted out."""
    if verify:
        return ssl.create_default_context()
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("TLS verification disabled for non-loopback host %s", host)
    return ssl._create_unverified_context()


def _close_imap(imap: imaplib.IMAP4) -> None:
    """Teardown that guarantees the socket closes: ``logout()`` only guards ``OSError``, so ``IMAP4.abort`` on a
    broken connection skipped ``shutdown()`` and leaked one fd per failed poll (fatal on macOS's 256 soft limit).

    ``IMAP4.logout()`` only guards against ``OSError`` internally: a broken connection makes
    ``_simple_command('LOGOUT')`` raise ``IMAP4.abort`` (which is *not* an ``OSError``), so ``logout()``
    propagates before its own ``shutdown()`` call and the TCP socket stays open. On macOS, where the default
    soft fd limit is 256 and pollers may run through a local proxy, these abandoned sockets accumulate one
    per failed poll until the gateway hits ``[Errno 24] Too many open files`` (#79889).
    """
    try:
        imap.logout()
    except Exception:
        with suppress(Exception):
            imap.shutdown()


def _create_ipv4_connection(host: str, port: int, timeout: float, source_address: Any = None) -> socket.socket:
    """``socket.create_connection`` constrained to ``AF_INET`` (no process-global socket mutation — sends run in executor threads)."""
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise last_error if last_error is not None else OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(host, port, timeout, source_address=self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return self.context.wrap_socket(
            _create_ipv4_connection(host, port, timeout, source_address=self.source_address),
            server_hostname=getattr(self, "_host", host),
        )


def _open_smtp(
    host: str, port: int, security: str, ctx: ssl.SSLContext, smtp_cls: type, smtp_ssl_cls: type, **kwargs: Any
) -> smtplib.SMTP:
    """Open one SMTP connection with TLS established per *security*; *kwargs* go to the constructor."""
    if security == "tls":
        return smtp_ssl_cls(host, port, context=ctx, **kwargs)
    smtp = smtp_cls(host, port, **kwargs)
    if security == "starttls":
        try:
            smtp.starttls(context=ctx)
        except Exception:
            smtp.close()
            raise
    return smtp


def _send_imap_id(imap: imaplib.IMAP4) -> None:
    """Send RFC 2971 IMAP ID: 163/NetEase require it after LOGIN (else every UID command
    returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed.

    Sent only when the server advertises ``ID`` (RFC 2971 requires advertising it): a server
    without the extension can answer an untagged ``* BYE Unknown command.`` and close the
    connection, which imaplib cannot surface here — the failure appears one command later
    as a misleading SELECT error and the adapter retries forever (Purelymail, #39856).
    ``imap.capabilities`` is populated by imaplib at connect, so the check is free."""
    if "ID" not in imap.capabilities:
        logger.debug("[Email] Server does not advertise IMAP ID capability; skipping ID")
        return
    try:
        try:
            from hermes_cli.version_info import get_version_info

            version = get_version_info().base_version
        except Exception:
            version = "0"
        imap.xatom(
            "ID",
            f'("name" "hermes-agent" "version" "{version}" '
            '"vendor" "NousResearch" "support-email" "noreply@nousresearch.com")',
        )
    except Exception as e:
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _imap_default_security(imap_port: int) -> str:
    """[PATCH-10] Port-derived IMAP security default when none is configured.

    Upstream defaults ``imap_security`` to ``tls`` regardless of the port, which
    breaks every existing 143/STARTTLS setup that never had to set the key (the
    server greets in plaintext, so the implicit-TLS handshake dies with
    ``[SSL: RECORD_LAYER_FAILURE]``). Mirror upstream's own SMTP rule instead
    (465 → ``tls``, else ``starttls``): 993 → ``tls``, any other port →
    ``starttls``. An explicit EMAIL_IMAP_SECURITY / ``imap_security`` still wins.
    This keeps the port-based behaviour our former ``_open_imap_conn`` had.
    """
    return "tls" if imap_port == 993 else "starttls"


def _open_imap_conn(host: str, port: int, security: str, tls_verify: bool) -> imaplib.IMAP4:
    """[PATCH-3] Unauthenticated IMAP connection (tls / starttls / plain).

    Module-level twin of upstream's ``EmailAdapter._connect_imap`` (same
    branches, same timeout) so the out-of-process ``_standalone_send`` Sent
    APPEND opens IMAP over the SAME transport rules as the live adapter.
    Kept as a copy rather than rewiring ``_connect_imap`` so the upstream
    method stays byte-identical for future re-syncs.
    """
    if security == "tls":
        return imaplib.IMAP4_SSL(host, port, timeout=30, ssl_context=_tls_context(tls_verify, host))
    imap = imaplib.IMAP4(host, port, timeout=30)
    if security == "starttls":
        try:
            imap.starttls(ssl_context=_tls_context(tls_verify, host))
        except Exception:
            _close_imap(imap)
            raise
    return imap


def _imap_append_to_sent(
    *,
    imap_host: str,
    imap_port: int,
    imap_security: str,
    imap_tls_verify: bool,
    address: str,
    password: str,
    sent_folder: str,
    raw_bytes: bytes,
) -> None:
    """[PATCH-3] IMAP-APPEND a freshly-sent outbound mail to ``sent_folder``.

    No-op when the folder is unset (empty string) or no IMAP host is
    configured. Best-effort: failures are logged as warnings and never
    re-raised — losing the Sent-folder copy must NOT roll back an SMTP send
    that already succeeded.

    Shared by the live :meth:`EmailAdapter._append_to_sent` and the
    out-of-process ``_standalone_send`` so both SMTP paths archive identically
    (parity with upstream PR #28697's review-driven refactor).
    """
    if not sent_folder or not imap_host:
        return
    try:
        imap = _open_imap_conn(imap_host, imap_port, imap_security, imap_tls_verify)
        try:
            imap.login(address, password)
            _send_imap_id(imap)
            # CREATE is idempotent; most servers return NO on "already exists".
            try:
                imap.create(sent_folder)
            except Exception:  # ignore "already exists" and similar
                pass
            # imaplib returns ("NO"/"BAD", ...) on a rejected APPEND WITHOUT
            # raising — inspect the status tuple explicitly so a silent failure
            # isn't logged as success.
            typ, data = imap.append(
                sent_folder,
                "(\\Seen)",
                imaplib.Time2Internaldate(time.time()),
                raw_bytes,
            )
            if typ != "OK":
                detail = b" ".join(p for p in data if isinstance(p, bytes)).decode("utf-8", "replace")
                logger.warning("[Email] APPEND to %r returned %s: %s", sent_folder, typ, detail)
            else:
                logger.debug("[Email] APPEND to %r ok", sent_folder)
        finally:
            _close_imap(imap)
    except Exception as e:  # Sent-folder mirror is best-effort
        logger.warning("[Email] APPEND to %r failed: %s", sent_folder, e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """True if this email is from an automated/noreply source."""
    addr = address.lower()
    return any(pattern in addr for pattern in _NOREPLY_PATTERNS) or any(
        (value := headers.get(header, "")) and check(value) for header, check in _AUTOMATED_HEADERS.items()
    )


def _imap_payload(data: Any) -> Optional[bytes]:
    """The bytes of a one-message IMAP FETCH response, or ``None`` for an unexpected shape (see #80032)."""
    try:
        payload = data[0][1]
    except (IndexError, TypeError):
        return None
    return bytes(payload) if isinstance(payload, (bytes, bytearray)) else None


def check_email_requirements() -> bool:
    """True when all email settings are present and non-blank (blank keys left by an abandoned setup must not enable the platform).

    Treats blank/whitespace-only values as missing so an abandoned setup that left empty ``EMAIL_*`` keys in
    ``.env`` does not enable the platform (#40715).
    """
    return all(
        _get_secret(name, "").strip()
        for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST")
    )


def _safe_decode(payload: bytes, charset: Optional[str]) -> str:
    """Decode without ever raising: ``errors="replace"`` does not guard a missing codec (``LookupError``), so fall back alias → UTF-8 → latin-1.

    Unknown or malformed charset labels (``unknown-8bit``, misspelled names, attacker-controlled garbage)
    previously raised ``LookupError`` from ``bytes.decode`` — ``errors="replace"`` only guards decode
    errors, not a missing codec — which aborted the whole IMAP fetch and dropped every message in the batch
    (#35901, #55381, #55383). Fall back through a small alias table, then UTF-8, then latin-1 (which never
    fails).
    """
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    for candidate in (_CHARSET_ALIASES.get(label, label), "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 header into a plain string; never raises.

    Never raises: malformed encoded-words or unknown charsets degrade to replacement characters instead of
    crashing the fetch loop (#55381).
    """
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    return " ".join(_safe_decode(part, charset) if isinstance(part, bytes) else part for part, charset in parts)


def _first_body_part(msg: email_lib.message.Message, content_type: str) -> str:
    """Decoded text of the first non-attachment part of *content_type*, or ''."""
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")) or part.get_content_type() != content_type:
            continue
        if payload := part.get_payload(decode=True):
            return _safe_decode(payload, part.get_content_charset())
    return ""


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        html = _first_body_part(msg, "text/html")
        return _first_body_part(msg, "text/plain") or (_strip_html(html) if html else "")
    text = _safe_decode(payload, msg.get_content_charset()) if (payload := msg.get_payload(decode=True)) else ""
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    for pattern, repl in _HTML_SUBS:
        html = pattern.sub(repl, html)
    return html.strip()


def _extract_email_address(raw: str) -> str:
    """Bare lowercased addr-spec from a From: value. Uses parseaddr, not a first-<...> regex (GHSA-rxqh-5572-8m77);
    RFC 5322 folding is unfolded first because parseaddr misreads a folded quoted display name. Unquoted
    ``Name <addr>`` values parseaddr rejects (``Doe, John <j@x>``) fall back to their single bracketed address,
    but only when the display part (``(comments)`` removed) cannot hold another mailbox or group: no quotes, ``;``,
    ``:`` or stray parens, and no ``@`` unless it is exactly the bracketed address (``a@x <a@x>``). Values over
    ``_MAX_FROM_LEN`` (or with more than 64 ``(``) and results without ``@`` return ``""`` so the caller drops the message.
    """
    value = re.sub(r"\r?\n[ \t]+", " ", str(raw or ""))
    if len(value) > _MAX_FROM_LEN or value.count("(") > 64:
        return ""  # hostile size/nesting: take the empty-sender drop (parseaddr recurses per nested comment)
    _, addr = parseaddr(value)
    if not addr and (m := _SINGLE_BRACKET_FROM_RE.fullmatch(value)):
        display, bracketed = _strip_comments(m.group(1)).strip(), m.group(2)
        if (
            "@" in bracketed[1:-1]
            and not any(c in display for c in ";:()")
            and ("@" not in display or display.lower() == bracketed.lower())
        ):
            addr = bracketed
    addr = addr.strip().lower()
    return addr if "@" in addr else ""  # a bare word (``John``) is not a sender identity


def _strip_comments(text: str) -> str:
    """Remove (possibly nested) ``(comments)``, innermost first, until nothing changes."""
    while (stripped := _COMMENT_RE.sub(" ", text)) != text:
        text = stripped
    return text


def _ar_clauses(text: str) -> Optional[list[str]]:
    """Split an Authentication-Results value on ``;`` outside quoted-strings and (nested) comments; comments are
    dropped, quoted-strings kept (``header.from="x"`` stays readable). ``None`` when a quote or comment is unbalanced.
    """
    clauses, cur, depth, quoted, i = [], [], 0, False, 0
    while i < len(text):
        c = text[i]
        if c == "\\" and (quoted or depth):
            if quoted:
                cur.append(text[i : i + 2])
            i += 2
            continue
        if quoted:
            quoted = c != '"'
            cur.append(c)
        elif depth:
            depth += {"(": 1, ")": -1}.get(c, 0)
            if not depth:
                cur.append(" ")
        elif c == "(":
            depth = 1
        elif c == '"':
            quoted = True
            cur.append(c)
        elif c == ")":
            return None  # stray close paren: unbalanced
        elif c == ";":
            clauses.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    return None if quoted or depth else clauses + ["".join(cur)]


def _auth_props(text: str) -> list[tuple[str, str]]:
    """``(property, value)`` pairs (``header.from=x``) of one comment-free Authentication-Results clause, property
    lowercased, surrounding quotes stripped. Quoted-string contents are never scanned for properties."""
    return [(p.lower(), v.strip('"')) for p, v in _AUTH_PROP_RE.findall(text) if p]


def _domain_of(address: str) -> str:
    """Lowercased domain part of an email address, or ''."""
    return address.rpartition("@")[2].strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Relaxed DMARC alignment: equal, or one is a dot-suffix of the other."""
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def _verify_sender_authentication(
    msg: email_lib.message.Message, from_addr: str, *, authserv_id: str
) -> tuple[bool, str]:
    """Verify the ``From:`` domain is authenticated; returns ``(authenticated, reason)``.
    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy signal is the
    ``Authentication-Results`` header stamped by the *receiving* server. It prepends, so only the
    FIRST instance is authoritative, and it must match the required, already-normalised *authserv_id* exactly. A matching id is a
    pin, not proof of provenance: the receiving MTA must strip inbound results claiming its id (RFC 8601).
    True on DMARC pass, aligned SPF pass, or aligned DKIM (``header.d``) pass. No header or no pin → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``)."""
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    if not (headers := msg.get_all("Authentication-Results")):
        return False, _NO_AUTH_RESULTS_REASON
    trusted = " ".join(str(headers[0]).split())
    # _ar_clauses removes RFC 8601 CFWS comments and ignores semicolons inside them, so supported
    # receiver variants remain valid without allowing a lower field or a related domain to satisfy the pin.
    if (clauses := _ar_clauses(trusted)) is None:
        return False, "unbalanced quote or comment in Authentication-Results"
    if not authserv_id:  # without a pin the topmost header may be one the sender wrote
        return False, _MISSING_AUTHSERV_REASON
    if clauses[0].strip().lower() != authserv_id:
        return False, _UNTRUSTED_AUTHSERV_REASON
    # Each verdict comes from the head of its own clause (split outside quotes/comments) and its domains only from that
    # clause: a quoted local part or comment can otherwise smuggle ``spf=pass``/``header.d=`` (GHSA-rxqh-5572-8m77).
    results: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {"dmarc": [], "spf": [], "dkim": []}
    for clause in clauses:
        if m := _AUTH_METHOD_RE.match(clause):
            results[m.group(1).lower()].append((m.group(2).lower(), _auth_props(clause)))

    def aligned(props: list[tuple[str, str]], names: tuple[str, ...], *, required: bool = True) -> bool:
        domains = [_domain_of(v) for p, v in props if p in names]
        return (bool(domains) or not required) and all(_domains_aligned(d, from_domain) for d in domains)

    if len(results["dmarc"]) > 1:
        return False, "ambiguous dmarc result"
    # every header.from in the dmarc clause must be the From domain we parsed (absent header.from: trust the verdict)
    if any(r == "pass" and aligned(props, ("header.from",), required=False) for r, props in results["dmarc"]):
        return True, "dmarc=pass"
    # one SMTP transaction has one MAIL FROM verdict: a second spf clause means the SPF signal is not trusted
    if (
        len(results["spf"]) == 1
        and (spf := results["spf"][0])[0] == "pass"
        and aligned(spf[1], ("smtp.mailfrom", "smtp.from", "envelope-from"))
    ):
        return True, "spf=pass aligned"
    # several dkim clauses are normal (one per signature): any single pass whose own header.d aligns is enough
    if any(
        r == "pass" and aligned(props, ("header.d",) if any(p == "header.d" for p, _ in props) else ("header.from",))
        for r, props in results["dkim"]
    ):
        return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(msg: email_lib.message.Message, skip_attachments: bool = False) -> list[dict[str, Any]]:
    """Extract attachment metadata and cache files locally (nothing when *skip_attachments*)."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        disposition, content_type = str(part.get("Content-Disposition", "")), part.get_content_type()
        if skip_attachments or (
            "attachment" not in disposition
            and ("inline" not in disposition or content_type in {"text/plain", "text/html"})
        ):
            continue  # not an attachment, or an inline text/html body part
        filename = (
            _decode_header_value(fn)
            if (fn := part.get_filename())
            else f"attachment.{part.get_content_subtype() or 'bin'}"
        )
        if not (payload := part.get_payload(decode=True)):
            continue
        if (ext := Path(filename).suffix.lower()) in _IMAGE_EXTS:
            try:
                cached_path, kind = cache_image_from_bytes(payload, ext), "image"
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
        else:
            cached_path, kind = cache_document_from_bytes(payload, filename), "document"
        attachments.append({"path": cached_path, "filename": filename, "type": kind, "media_type": content_type})
    return attachments


def _format_quote_date(raw: str) -> str:
    """[PATCH-11] Human-readable ``Date:`` value; the raw header when unparseable, '' when missing."""
    if not (raw := (raw or "").strip()):
        return ""
    try:
        return parsedate_to_datetime(raw).strftime("%a, %d %b %Y %H:%M %z").strip()
    except (TypeError, ValueError, IndexError, OverflowError):
        return raw


def _build_quote(original_body: str, *, date: str, name: str, address: str, header_fmt: str, max_chars: int) -> str:
    """[PATCH-11] Classic ``> `` quote block for appending under a reply: ``"\\n\\n" + header + "\\n" + quoted lines``.

    *max_chars* bounds the quoted lines (prefixes included, header excluded); the cut lands on a line end and is
    marked with ``> [...]``. Already-quoted lines just gain another level (``>> ``). Empty body (or no line fits) → ''.
    """
    lines = [line.rstrip() for line in (original_body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if not lines or max_chars <= 0:
        return ""
    quoted = [f">{line}" if line.startswith(">") else f"> {line}" if line else ">" for line in lines]
    text = "\n".join(quoted)
    if len(text) > max_chars:
        kept: list[str] = []
        used = len(_QUOTE_TRUNCATED)
        for line in quoted:
            if used + len(line) + 1 > max_chars:
                break
            kept.append(line)
            used += len(line) + 1
        if not kept:
            return ""
        text = "\n".join([*kept, _QUOTE_TRUNCATED])
    shown_date, shown_name = _format_quote_date(date), (name or "").strip() or address
    fmt = header_fmt or _DEFAULT_QUOTE_HEADER
    if not shown_date:  # "On {date}, X wrote:" -> "X wrote:"
        fmt = re.sub(r"On\s*\{date\},?\s*", "", fmt).replace("{date}", "").strip()
    try:
        header = fmt.format(date=shown_date, name=shown_name, address=address)
    except (KeyError, IndexError, ValueError):  # operator typo in quote_header must not break sending
        header = (
            _DEFAULT_QUOTE_HEADER.format(date=shown_date, name=shown_name, address=address)
            if shown_date
            else f"{shown_name} <{address}> wrote:"
        )
    return f"\n\n{header}\n{text}"


def _attach_file(msg: MIMEMultipart, path: Path, filename: str) -> None:
    """Attach *path* to *msg* as base64 application/octet-stream."""
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


_INFLIGHT_MAX = 500  # [PATCH-12]
_INFLIGHT_SWEEP_GRACE_S = 300.0  # [PATCH-12]
_RETRY_MAX_ATTEMPTS = 5  # [PATCH-12]
_RETRY_MAX_AGE_S = 7200.0  # [PATCH-12]
try:
    from gateway.session_ingress import QUEUE_FULL_REPLY as _QUEUE_FULL_REPLY  # [PATCH-12]
except ImportError:
    _QUEUE_FULL_REPLY = None


class _EmailMessageEvent(MessageEvent):
    """[PATCH-12] MessageEvent that remembers the Message-IDs merged into it while queued."""

    def absorb_reply_expected(self, other: MessageEvent) -> None:
        super().absorb_reply_expected(other)
        ids: set[str] = self.__dict__.setdefault("_absorbed_ids", set())
        ids.update(i for i in (self.message_id, other.message_id) if i)
        ids.update(getattr(other, "_absorbed_ids", ()))


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    serial_receive_loop = True  # the IMAP poll loop awaits each message before dispatching the next
    # One email carries the whole body, so cron delivery hands over the full payload untruncated.
    splits_long_messages = True

    # Per-account seen-UID snapshot surviving adapter recreation: the reconnect watcher builds a FRESH
    # adapter per retry; without this connect(is_reconnect=True) would re-mark the mailbox seen and skip
    # mail that arrived during the outage. Keyed by address (multiplex runs several accounts); same-process only.
    _seen_uids_snapshot: dict[str, set] = {}
    # Accounts already warned about a missing authserv_id pin. Per address, not per first connect: an account whose
    # first connect fails is brought up by the reconnect watcher (is_reconnect=True) and must still warn once.
    _missing_pin_warned: set = set()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)
        # Env first, then PlatformConfig.extra (config.yaml-only setups). Host/address are stripped: a stray
        # newline made IMAP4_SSL raise ``[Errno 8] nodename nor servname`` instead of "host not set".
        extra = config.extra or {}
        setting = lambda env, key: _get_secret(env, "") or extra.get(key, "")
        tls_verify = lambda env, key: _esecret_bool(env, is_truthy_value(extra.get(key), default=True))
        self._address = setting("EMAIL_ADDRESS", "address").strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = setting("EMAIL_IMAP_HOST", "imap_host").strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        # [PATCH-10] default derived from the port (upstream: always "tls"), see _imap_default_security.
        self._imap_security = _normalize_security(
            setting("EMAIL_IMAP_SECURITY", "imap_security"), default=_imap_default_security(self._imap_port)
        )
        self._imap_tls_verify = tls_verify("EMAIL_IMAP_TLS_VERIFY", "imap_tls_verify")
        self._smtp_host = setting("EMAIL_SMTP_HOST", "smtp_host").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._smtp_security = _normalize_security(
            setting("EMAIL_SMTP_SECURITY", "smtp_security"), default="tls" if self._smtp_port == 465 else "starttls"
        )
        self._smtp_tls_verify = tls_verify("EMAIL_SMTP_TLS_VERIFY", "smtp_tls_verify")
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)
        self._skip_attachments = extra.get("skip_attachments", False)  # platforms.email.skip_attachments
        # [PATCH-2] dgxarley: folder-lifecycle + APPEND-to-Sent + process-existing,
        # all configured via config.yaml ``platforms.email.extra.*``:
        #   platforms:
        #     email:
        #       extra:
        #         working_folder: "Hermes_Working"  # "" to skip the Working stage
        #         done_folder: "Hermes_Done"        # "" to disable all folder moves
        #         sent_folder: "Sent"               # "" to disable IMAP APPEND
        #         process_existing: true            # process pre-existing backlog
        # All four mirror the upstream PRs (NousResearch/hermes-agent #28697,
        # #28699, #28702). Empty string is a deliberate opt-out for the folder
        # vars — do NOT collapse with `or`. Lifecycle semantics:
        #   done_folder="" .............. no moves at all (INBOX, \Seen)
        #   working_folder="", done set . INBOX -> Done directly
        #   both set .................... INBOX -> Working -> Done
        # NOTE: our default for process_existing is True (process the backlog),
        # unlike upstream's False.
        self._working_folder = extra.get("working_folder", "Hermes_Working")
        self._done_folder = extra.get("done_folder", "Hermes_Done")
        self._failed_folder = extra.get("failed_folder", "Hermes_Failed")  # [PATCH-12] "" -> done_folder
        self._rejected_folder = extra.get("rejected_folder", "Hermes_Rejected")  # [PATCH-12] "" -> done_folder
        self._sent_folder = extra.get("sent_folder", "Sent")
        self._process_existing = bool(extra.get("process_existing", True))
        # Require an authenticated From: domain (SPF/DKIM/DMARC) before trusting it for authorization
        # (GHSA-rxqh-5572-8m77). Default ON; opt out via require_authenticated_sender: false / EMAIL_TRUST_FROM_HEADER=true.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        else:
            self._require_authenticated_sender = not _esecret_bool("EMAIL_TRUST_FROM_HEADER", False)
        # Pin Authentication-Results to the operator's receiving MTA. With sender auth on, an absent pin fails
        # closed for EVERY sender rather than trusting a sender-supplied result header; connect() warns once per account.
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()
        # [PATCH-11] Quote the inbound mail under the turn's final reply to it (default off; env wins
        # over config.yaml, same precedence as every other EMAIL_*/extra pair above).
        self._quote_original = _esecret_bool(
            "EMAIL_QUOTE_ORIGINAL", is_truthy_value(extra.get("quote_original"), default=False)
        )
        self._quote_max_chars = _esecret_int(
            "EMAIL_QUOTE_MAX_CHARS", coerce_port(extra.get("quote_max_chars"), _DEFAULT_QUOTE_MAX_CHARS)
        )
        self._quote_header = str(extra.get("quote_header") or _DEFAULT_QUOTE_HEADER)
        # Originals by Message-ID so parallel mails from one sender quote the right one; the per-sender entry
        # covers mails without a Message-ID. Only populated while quoting is on.
        self._original_by_msg_id: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._last_original_by_sender: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._quote_lock = threading.Lock()  # sends run concurrently in executor threads
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000  # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._last_fetch_failed, self._last_fetch_error = (
            False,
            "",
        )  # "checked, nothing new" vs "the check itself failed"
        # chat_id (sender email) -> last subject + message-id for threading
        # Track the last IMAP fetch attempt so the poll loop can distinguish "checked, nothing new" from
        # "the check itself failed" (#80016).
        self._thread_context: dict[str, dict[str, str]] = {}
        # [PATCH-12] Message-ID -> (source folder, event, registered-at monotonic); event loop only.
        self._inflight: "OrderedDict[str, tuple[str, MessageEvent, float]]" = OrderedDict()
        # [PATCH-12] Message-ID -> refused-while-busy mail awaiting re-dispatch: [msg_data, event, first_ts, attempts].
        self._retry: "OrderedDict[str, list[Any]]" = OrderedDict()
        self._queue_full_chats: set[str] = set()  # [PATCH-12]
        logger.info("[Email] Adapter initialized for %s", self._address)
        logger.info(
            "[Email] Folder lifecycle: working=%r done=%r failed=%r rejected=%r sent=%r process_existing=%s imap_security=%s",
            self._working_folder,
            self._done_folder,
            self._failed_folder,
            self._rejected_folder,
            self._sent_folder,
            self._process_existing,
            self._imap_security,
        )

    def _trim_seen_uids(self) -> None:
        """Keep only the highest half of UIDs once over the cap (UIDs are monotonic; UNSEEN prevents re-delivery)."""
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))  # UIDs are bytes like b'1234'
            self._seen_uids = set(sorted_uids[-(self._seen_uids_max // 2) :])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2 :])

    def _connect_imap(self) -> imaplib.IMAP4:
        """Create an IMAP connection using implicit TLS, STARTTLS, or plaintext."""
        if self._imap_security == "tls":
            return imaplib.IMAP4_SSL(
                self._imap_host,
                self._imap_port,
                timeout=30,
                ssl_context=_tls_context(self._imap_tls_verify, self._imap_host),
            )
        imap = imaplib.IMAP4(self._imap_host, self._imap_port, timeout=30)
        if self._imap_security == "starttls":
            try:
                imap.starttls(ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
            except Exception:
                _close_imap(imap)
                raise
        return imap

    @contextmanager
    def _inbox(self):
        """Logged-in IMAP handle on INBOX; always ``_close_imap``-ed on exit (a login/select failure used to leak one fd per reconnect)."""
        # Test IMAP connection. The handle is closed in ``finally`` — before this, a failure in
        # login/select/search left the TCP socket open with no owner, leaking one fd per connect attempt.
        # Under the gateway's reconnect watcher (fresh adapter instance per retry) against an
        # unreachable/proxied host this grew monotonically until fd exhaustion on macOS's 256 soft limit
        # (#79889).
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            _close_imap(imap)

    def _connect_smtp(self) -> smtplib.SMTP:
        """SMTP connection with TLS established (callers go straight to ``login()``). An unreachable IPv6 address can
        hang until the socket timeout, so connection-level failures retry through an IPv4-only socket path (no global
        resolver mutation); TLS verification errors are not retried."""
        host, port, security, ctx = (
            self._smtp_host,
            self._smtp_port,
            self._smtp_security,
            _tls_context(self._smtp_tls_verify, self._smtp_host),
        )
        try:
            return _open_smtp(host, port, security, ctx, smtplib.SMTP, smtplib.SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)
        except (TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            return _open_smtp(host, port, security, ctx, _IPv4SMTP, _IPv4SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)

    # [PATCH-3] IMAP folder-lifecycle helpers (dgxarley patch). All IMAP opens
    # go through upstream's _inbox() (login + ID + SELECT INBOX, _close_imap on
    # exit) or the module-level _open_imap_conn twin, so upstream's
    # imap_security / imap_tls_verify apply everywhere.

    @staticmethod
    def _ensure_folder(imap: imaplib.IMAP4, name: str) -> None:
        """Idempotently CREATE *name* on the IMAP server.

        ``IMAP CREATE`` returns NO if the folder already exists, which we
        accept silently — there is no portable EXISTS check across servers.
        """
        if not name:
            return
        try:
            status, _ = imap.create(name)
            if status == "OK":
                logger.info("[Email] Created IMAP folder %r", name)
        except Exception as e:  # best-effort
            logger.debug("[Email] CREATE %r ignored: %s", name, e)

    @staticmethod
    def _imap_move(imap: imaplib.IMAP4, uid: bytes, dst_folder: str) -> bool:
        """MOVE *uid* (in the currently SELECTed folder) to *dst_folder*.

        Tries the RFC 6851 ``UID MOVE`` first; falls back to
        ``UID COPY`` + ``UID STORE +FLAGS \\Deleted`` + ``EXPUNGE`` on
        servers that don't advertise MOVE. Returns True on apparent success.
        """
        try:
            status, data = imap.uid("MOVE", uid, dst_folder)
            if status == "OK":
                return True
            logger.debug("[Email] UID MOVE %s → %r: %s %s", uid, dst_folder, status, data)
        except Exception as e:  # fall through to COPY+EXPUNGE
            logger.debug("[Email] UID MOVE %s → %r raised: %s", uid, dst_folder, e)

        try:
            status, _ = imap.uid("COPY", uid, dst_folder)
            if status != "OK":
                logger.warning("[Email] UID COPY %s → %r failed", uid, dst_folder)
                return False
            imap.uid("STORE", uid, "+FLAGS", "(\\Deleted)")
            # Prefer UID EXPUNGE (RFC 4315) so we don't expunge collateral
            # deleted-flagged mails the user has in the same folder.
            try:
                imap.uid("EXPUNGE", uid)
            except Exception:  # server lacks UIDPLUS
                imap.expunge()
            return True
        except Exception as e:
            logger.error("[Email] COPY+EXPUNGE move %s → %r failed: %s", uid, dst_folder, e)
            return False

    @staticmethod
    def _search_message_id(imap: imaplib.IMAP4, message_id: str) -> list[bytes]:
        """Return UIDs in the currently SELECTed folder whose Message-ID matches.

        Used to re-locate a mail after we MOVE'd it (UIDs change on MOVE,
        but the RFC 2822 Message-ID header is stable).
        """
        if not message_id:
            return []
        try:
            status, data = imap.uid("SEARCH", None, "HEADER", "Message-ID", message_id)
            if status != "OK" or not data or not data[0]:
                return []
            return data[0].split()
        except Exception as e:
            logger.debug("[Email] SEARCH Message-ID %r failed: %s", message_id, e)
            return []

    def _target_folder(self, kind: str, *, strict: bool = False) -> Optional[str]:
        """[PATCH-12] Destination for ``kind`` (done/failed/rejected); None when no move applies.

        An empty failed/rejected folder falls back to ``done_folder`` unless ``strict`` (then None).
        """
        if not self._done_folder:
            return None
        specific = {"failed": self._failed_folder, "rejected": self._rejected_folder}.get(kind)
        if kind == "done" or (not specific and not strict):
            return self._done_folder
        return specific or None

    def _finalize_message(self, message_id: str, source_folder: str, target_folder: Optional[str]) -> None:
        """Move a mail to ``target_folder`` (best-effort).

        Runs in an executor thread (synchronous IMAP). If ``target_folder`` is
        empty, this is a no-op — the mail stays where it is, with the IMAP
        \\Seen flag set automatically by the earlier RFC822 FETCH.
        """
        if not target_folder or target_folder == source_folder:
            return
        if not message_id:
            logger.debug(
                "[Email] Skipping finalize MOVE — no Message-ID for mail in %r (cannot relocate after handle_message)",
                source_folder,
            )
            return
        try:
            with self._inbox() as imap:
                imap.select(source_folder)
                uids = self._search_message_id(imap, message_id)
                if not uids:
                    logger.warning(
                        "[Email] Cannot find Message-ID %s in %r — skipping MOVE → %r",
                        message_id,
                        source_folder,
                        target_folder,
                    )
                    return
                for uid in uids:
                    self._imap_move(imap, uid, target_folder)
                logger.info("[Email] Finalized %s: %r → %r", message_id, source_folder, target_folder)
        except Exception as e:
            logger.error("[Email] Finalize MOVE failed: %s", e)

    def _append_to_sent(self, raw_bytes: bytes) -> None:
        """IMAP-APPEND an outbound mail to ``self._sent_folder`` via the shared helper (best-effort)."""
        _imap_append_to_sent(
            imap_host=self._imap_host,
            imap_port=self._imap_port,
            imap_security=self._imap_security,
            imap_tls_verify=self._imap_tls_verify,
            address=self._address,
            password=self._password,
            sent_folder=self._sent_folder,
            raw_bytes=raw_bytes,
        )

    # End of [PATCH-3] block.

    def _fail(self, log_fmt: str, err: object, code: str, detail: str, *, retryable: bool) -> bool:
        """Log *err*, record a fatal error for the gateway's reconnect machinery, return False."""
        logger.error(log_fmt, err)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    def _probe_imap(self, is_reconnect: bool) -> bool:
        """Connection test + seen-UID baseline. Sets a fatal error and returns False on failure."""
        try:
            with self._inbox() as imap:
                # [PATCH-4] Ensure our managed folders exist before any MOVE
                # touches them. CREATE is idempotent (NO on "already exists"),
                # so this is safe on every reconnect. Working/Done are only
                # created when the lifecycle is active (done_folder set); the
                # Sent folder is independent (APPEND regardless of lifecycle).
                if self._done_folder:
                    self._ensure_folder(imap, self._working_folder)
                    self._ensure_folder(imap, self._done_folder)
                    for extra_folder in (self._failed_folder, self._rejected_folder):
                        if extra_folder:
                            self._ensure_folder(imap, extra_folder)
                self._ensure_folder(imap, self._sent_folder)
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # Same-process reconnect: restore the previous adapter's baseline so mail that
                    # arrived during the outage stays eligible for the next poll.
                    self._seen_uids = set(snapshot)
                    passed = "[Email] IMAP reconnect test passed. Restored %d seen UIDs; messages received during the outage will be processed."
                elif self._process_existing:
                    # [PATCH-4] First connect with process_existing=true (config.yaml):
                    # leave _seen_uids empty so the next poll picks up the pre-existing
                    # UNSEEN backlog. Orthogonal to the reconnect branch above, which
                    # only fires on a same-process RECONNECT, never on a cold start.
                    passed = "[Email] IMAP connection test passed. process_existing=true — %d seen UIDs, pre-existing UNSEEN mail is processed on first poll."
                else:  # first connect (or no snapshot): mark all existing messages seen
                    status, data = imap.uid("search", None, "ALL")
                    self._seen_uids.update(data[0].split() if status == "OK" and data and data[0] else ())
                    passed = "[Email] IMAP connection test passed. %d existing messages skipped."
                self._trim_seen_uids()
                logger.info(passed, len(self._seen_uids))
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
            return True
        except Exception as e:
            # Always set an explicit fatal code, else the gateway treats every failure as transient with zero
            # owner signal. retryable=True because imaplib raises the same generic IMAP4.error for bad credentials
            # AND transient NOs (Gmail "too many simultaneous connections"); loops surface via NEEDS_ATTENTION.
            return self._fail(
                "[Email] IMAP connection failed: %s",
                e,
                "email_imap_connect_error",
                f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}",
                retryable=True,
            )

    def _probe_smtp(self) -> bool:
        """SMTP connect + login test. Sets a fatal error and returns False on failure."""
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # Typed auth failure (535 & friends) can never self-heal, so drop out of the reconnect queue — unambiguous, unlike IMAP4.error.
            return self._fail(
                "[Email] SMTP authentication failed: %s",
                e,
                "email_auth_error",
                f"SMTP authentication failed for {self._address}: {e}. Check EMAIL_PASSWORD (for Gmail/Outlook "
                "this must be an app password, not the account password).",
                retryable=False,
            )
        except Exception as e:
            return self._fail(
                "[Email] SMTP connection failed: %s",
                e,
                "email_smtp_connect_error",
                f"SMTP connection to {self._smtp_host} failed: {e}",
                retryable=True,
            )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host is an actionable config error, not IMAP4_SSL("") raising ``[Errno 8]``.
        required = (
            ("EMAIL_ADDRESS", self._address),
            ("EMAIL_PASSWORD", self._password),
            ("EMAIL_IMAP_HOST", self._imap_host),
            ("EMAIL_SMTP_HOST", self._smtp_host),
        )
        if missing := [name for name, value in required if not value]:
            message = f"Not configured — missing {', '.join(missing)}. Set it via `hermes gateway setup` (env) or platforms.email in config.yaml."
            # Non-retryable: a blank-but-present env var used to drive an indefinite retry loop that leaked until OOM.
            return self._fail("[Email] %s", message, "email_missing_configuration", message, retryable=False)
        if not self._probe_imap(is_reconnect) or not self._probe_smtp():
            return False
        if (
            self._require_authenticated_sender
            and not self._authserv_id
            and self._address not in EmailAdapter._missing_pin_warned
        ):
            EmailAdapter._missing_pin_warned.add(self._address)
            logger.warning("[Email] %s: %s.%s", self._address, _MISSING_AUTHSERV_REASON, _MISSING_AUTHSERV_HINT)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        await cancel_task(self._poll_task)
        self._poll_task = None
        logger.info("[Email] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages at regular intervals."""
        while self._running:
            try:
                await self._sweep_inflight()
                await self._retry_deferred()
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        loop = asyncio.get_running_loop()

        async def authorize_on_loop(msg_data: dict[str, Any]) -> bool:
            return self._sender_accepted(msg_data["sender_addr"], msg_data)

        def gate(candidate: dict[str, Any]) -> bool:
            # Authorization reads profile-scoped policy and pairing state on the adapter's event loop;
            # the blocking IMAP worker waits for that verdict before asking the server for RFC822.
            # A closed loop (stop race) yields None: fail closed without leaking the coroutine.
            fut = safe_schedule_threadsafe(
                authorize_on_loop(candidate),
                loop,
                logger=logger,
                log_message="[Email] Could not schedule sender authorization",
            )
            return fut is not None and fut.result()

        messages = await asyncio.to_thread(self._fetch_new_messages, gate)
        # Dispatch partial results BEFORE escalating a failure — a mid-batch exception returns what was fetched (already marked seen).
        for msg_data in messages:
            await self._dispatch_message(msg_data)
        if self._last_fetch_failed:
            # The IMAP check itself failed (not an empty inbox): route through the fatal-error hook so the gateway's
            # reconnect/backoff re-establishes the mailbox. The handler runs detached (gateway/run.py), so awaiting it is safe.
            # The handler runs in a detached task (gateway/run.py), so awaiting it from our own poll task is
            # safe even though teardown cancels this task. See #80016.
            self._last_fetch_failed = False
            self._set_fatal_error(
                "email_imap_fetch_failed", self._last_fetch_error or "IMAP fetch failed", retryable=True
            )
            await self._notify_fatal_error()

    def _mark_uid_consumed(self, imap: imaplib.IMAP4, uid: Any) -> None:
        """Remember a rejected UID and mark it seen without fetching its MIME body."""
        self._seen_uids.add(uid)
        self._trim_seen_uids()
        status, _ = imap.uid("store", uid, "+FLAGS.SILENT", r"(\Seen)")
        if status != "OK":
            logger.warning("[Email] Could not mark rejected UID %s seen", uid)
        # [PATCH-5] Rejected before any RFC822 fetch, so it never reaches _dispatch_message/_finalize_message;
        # move it straight INBOX→Rejected (the UID is still valid here) so a drop does not stay in INBOX.
        target = self._target_folder("rejected")
        if target and not self._imap_move(imap, uid, target):
            logger.warning("[Email] Rejected-folder MOVE failed for rejected UID %s — mail stays in INBOX", uid)

    def _fetch_new_messages(self, preauthorize: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        """Fetch unseen messages; bounded headers pass *preauthorize* before RFC822 is requested."""
        results = []
        try:
            with self._inbox() as imap:
                status, data = imap.uid("search", None, "UNSEEN")
                for uid in (data[0].split() if status == "OK" and data and data[0] else []):
                    if uid in self._seen_uids:
                        continue
                    header_status, header_data = imap.uid("fetch", uid, _PREAUTH_FETCH)
                    if header_status != "OK":
                        continue
                    raw_headers = _imap_payload(header_data)
                    if raw_headers is None or len(raw_headers) > _MAX_PREAUTH_HEADER_BYTES:
                        logger.warning("[Email] Unusable pre-authorization headers for UID %s, skipping", uid)
                        self._mark_uid_consumed(imap, uid)
                        continue
                    try:
                        candidate = self._parse_fetched_headers(uid, raw_headers)
                        accepted = candidate is not None and preauthorize(candidate)
                    except Exception as auth_exc:
                        logger.error("[Email] Failed to authorize message UID %s, skipping: %s", uid, auth_exc)
                        accepted = False
                    if not accepted:
                        self._mark_uid_consumed(imap, uid)
                        continue
                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        continue  # transient per-UID refusal: leave unseen so the next poll retries
                    # Mark seen once a response arrived (even malformed) so garbage is skipped once, not retried forever —
                    # but NOT before the fetch: a connection failure must leave the rest of the batch eligible for the next poll.
                    self._seen_uids.add(uid)
                    self._trim_seen_uids()
                    if (raw_email := _imap_payload(msg_data)) is None:
                        logger.warning("[Email] Unexpected IMAP response structure for UID %s, skipping", uid)
                        continue
                    # One poison message (unparseable headers, pathological attachment, DNS hiccup) must not abort the batch or force a reconnect.
                    try:
                        # See #80032.
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error("[Email] Failed to process message UID %s, skipping: %s", uid, parse_exc)
                        continue
                    if parsed is not None:
                        # [PATCH-5] Two-stage move: park the mail in Working while the
                        # agent runs, so a crash mid-processing leaves a visible trail.
                        # Gated on done_folder (no Done → no moves; Working-without-Done
                        # would strand the mail) AND a Message-ID (the final MOVE → Done
                        # re-locates by Message-ID, UIDs do not survive a MOVE). Runs here,
                        # not in _parse_fetched_message, because it needs the open `imap`
                        # and must not run for unparseable or skipped (None) messages.
                        source_folder = "INBOX"
                        if self._done_folder and self._working_folder and parsed["message_id"]:
                            if self._imap_move(imap, uid, self._working_folder):
                                source_folder = self._working_folder
                            else:
                                logger.warning(
                                    "[Email] Working-folder MOVE failed for UID %s — continuing with mail still in INBOX",
                                    uid,
                                )
                        # [PATCH-5] Tells _dispatch_message -> _finalize_message where to look.
                        parsed["source_folder"] = source_folder
                        results.append(parsed)
        except Exception as e:
            # _close_imap guarantees the socket dies even when logout() raises IMAP4.abort on a broken
            # connection (#79889).
            logger.error("[Email] IMAP fetch error: %s", e)
            self._last_fetch_failed, self._last_fetch_error = True, str(e)
        # Keep the reconnect snapshot current so a mid-outage adapter recreation does not re-dispatch messages already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _message_metadata(self, uid: bytes, msg: email_lib.message.Message) -> Optional[dict[str, Any]]:
        """Parse sender-facing headers and authentication without touching the MIME body."""
        if not (sender_addr := _extract_email_address(msg.get("From", ""))):  # never dispatch an empty identity
            logger.debug("[Email] Dropping message with no parseable From address: %r", msg.get("From", ""))
            return None
        sender_name = _decode_header_value(msg.get("From", ""))
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')
        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        if _is_automated_sender(sender_addr, dict(msg.items())):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None
        sender_authenticated, auth_reason = _verify_sender_authentication(
            msg, sender_addr, authserv_id=self._authserv_id
        )
        return {
            "uid": uid,
            "sender_addr": sender_addr,
            "sender_name": sender_name,
            "subject": subject,
            "message_id": msg.get("Message-ID", ""),
            "in_reply_to": msg.get("In-Reply-To", ""),
            "date": msg.get("Date", ""),
            "sender_authenticated": sender_authenticated,
            "auth_reason": auth_reason,
        }

    def _parse_fetched_headers(self, uid: bytes, raw_headers: bytes | bytearray) -> Optional[dict[str, Any]]:
        """Parse the bounded IMAP header preflight without constructing a MIME tree."""
        return self._message_metadata(uid, BytesHeaderParser().parsebytes(bytes(raw_headers)))

    def _parse_fetched_message(self, uid: bytes, raw_email: bytes | bytearray) -> Optional[dict[str, Any]]:
        """Parse an authorized RFC822 payload into a dispatchable dict."""
        msg = email_lib.message_from_bytes(raw_email)
        if (metadata := self._message_metadata(uid, msg)) is None:
            return None
        metadata.update(
            body=_extract_text_body(msg), attachments=_extract_attachments(msg, skip_attachments=self._skip_attachments)
        )
        return metadata

    @staticmethod
    def _allow_all_senders() -> bool:
        """True when the operator opted into any sender (EMAIL_ or GATEWAY_ALLOW_ALL_USERS).

        Both names go through the scoped reader: under multiplex ``os.environ`` is the DEFAULT
        profile's opt-in, and borrowing it opened every secondary mailbox to any sender."""
        return any(
            _get_secret(name, "").strip().lower() in _TRUTHY
            for name in ("EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS")
        )

    def _answers_unknown_senders(self) -> bool:
        """True when ``platforms.email.unauthorized_dm_behavior`` opts into ``pair`` or ``decline``."""
        behavior = (self.config.extra or {}).get("unauthorized_dm_behavior")
        return isinstance(behavior, str) and behavior.strip().lower() in {"pair", "decline"}

    def _sender_accepted(self, sender_addr: str, msg_data: dict[str, Any]) -> bool:
        """Pre-dispatch sender gate: self, automated, authorization, From: authentication.

        Records the authorization verdict as ``msg_data["sender_granted"]`` (False for pair/decline answers)."""
        if sender_addr == self._address.lower():
            return False
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return False
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        # Parsed like the gateway's allowlists (JSON list literals included), or '["alice"]' would dodge the guard below.
        listed = set()
        for raw in (allowed_raw, _get_secret("GATEWAY_ALLOWED_USERS", "")):
            raw = decode_json_list_literal(raw)
            listed.update(
                str(a).strip().lower()
                for a in (raw if isinstance(raw, list) else str(raw).split(","))
                if str(a).strip()
            )
        is_listed = sender_addr.lower() in listed
        if is_listed:
            granted = True
        elif sender_addr.split("@", 1)[0].lower() in listed:
            # The gateway's check also matches an address by its bare local part (#119446), so an entry like "alice"
            # would admit, or pair, alice@<any domain>; the domain is the sender's to choose.
            logger.debug("[Email] Dropping sender whose local part alone matches an allowlist entry: %s", sender_addr)
            return False
        else:
            # Approved pairings grant access too, and only the gateway's own check sees them. Its verdict also decides
            # open access: GATEWAY_ALLOW_ALL_USERS beside a GATEWAY_ALLOWED_USERS list grants a stranger nothing there.
            verdict = self._is_sender_authorized(sender_addr, "dm", sender_addr)
            granted = verdict if verdict is not None else (not allowed_raw and self._allow_all_senders())
        # Drop senders the gateway would neither authorize nor answer (pair/decline) before a MessageEvent (and thread
        # context) exists — otherwise a dispatch/authorization race can send a reply even though the handler returned None.
        msg_data["sender_granted"] = granted  # [PATCH-11]
        if not granted and not self._answers_unknown_senders():
            logger.debug(
                "[Email] Dropping unauthorized sender at dispatch (unknown senders are ignored): %s", sender_addr
            )
            return False
        # Reject spoofed senders (GHSA-rxqh-5572-8m77): every grant keys on the attacker-controlled From:, and a pairing
        # code or decline is mailed back to it; fail-closed. Open access is no exception: the session and every reply
        # key on From:, so a forged one lands in that address's conversation and makes the agent mail it.
        # Warn only where the operator can act: a known misconfiguration, or a listed contact's own mail failing.
        # Forged stranger mail (open access grants everyone) stays at debug.
        if self._require_authenticated_sender and not msg_data.get("sender_authenticated", False):
            auth_reason = msg_data.get("auth_reason", "no verdict")
            hint = _DROP_HINTS.get(auth_reason, "")
            if is_listed or (granted and hint):
                logger.warning(
                    "[Email] Dropping sender with unauthenticated From: %s (%s).%s", sender_addr, auth_reason, hint
                )
            else:
                logger.debug(
                    "[Email] Dropping %s sender with unauthenticated From: %s (%s)",
                    "authorized" if granted else "unknown",
                    sender_addr,
                    auth_reason,
                )
            return False
        return True

    async def _dispatch_message(self, msg_data: dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        # [PATCH-6] Mail no background turn owns finalizes in the finally; accepted mail via [PATCH-12].
        source_folder = msg_data.get("source_folder", "INBOX")
        message_id = msg_data["message_id"]
        event: Optional[MessageEvent] = None
        defer = queued = False
        kind = "failed"
        try:
            if not self._sender_accepted(sender_addr, msg_data):
                kind = "rejected"
                return
            subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]
            text = (
                f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body
            )  # subject unless reply
            # DOCUMENT wins over PHOTO for mixed attachments: run.py keys image handling off the per-path mime type regardless
            # of message_type, but document-context injection gates strictly on MessageType.DOCUMENT — so DOCUMENT surfaces both.
            kinds = {att["type"] for att in attachments}
            self._thread_context[sender_addr] = {"subject": subject, "message_id": msg_data["message_id"]}
            # [PATCH-11] After _sender_accepted (already returned above on a drop) and inside the [PATCH-6]
            # try; pair/decline replies to not-granted senders never quote.
            if self._quote_original:
                if msg_data.get("sender_granted"):
                    self._remember_original(msg_data)
                else:
                    self._forget_sender_original(sender_addr)
            name = msg_data["sender_name"] or sender_addr
            event = _EmailMessageEvent(
                text=text or "(empty email)",
                message_id=msg_data["message_id"],
                message_type=(
                    MessageType.DOCUMENT
                    if "document" in kinds
                    else MessageType.PHOTO if "image" in kinds else MessageType.TEXT
                ),
                source=self.build_source(
                    chat_id=sender_addr,
                    chat_name=name,
                    chat_type="dm",
                    user_id=sender_addr,
                    user_name=name,
                    message_id=msg_data["message_id"],
                ),
                media_urls=[att["path"] for att in attachments],
                media_types=[att["media_type"] for att in attachments],
                reply_to_message_id=msg_data["in_reply_to"] or None,
            )
            logger.info("[Email] New message from %s: %s", sender_addr, subject)
            self._inflight_register(message_id, source_folder, event)
            await self.handle_message(event)
            queued = event._gateway_accepted is True or self._queued_by_base(event)
            defer = not queued and self._deferrable(event, msg_data)
            kind = "done" if event.get_command() else "failed"
        finally:
            if event is not None and queued:
                self._retry.pop(message_id, None)
            elif defer and event is not None:
                self._inflight.pop(message_id, None)
                if self._retry_note(msg_data, event):
                    await self._finalize_many([(message_id, source_folder, "failed", True)])
            else:
                self._retry.pop(message_id, None)
                self._inflight.pop(message_id, None)
                await self._finalize_many([(message_id, source_folder, kind, False)])

    def _queued_by_base(self, event: MessageEvent) -> bool:
        """[PATCH-12] Busy text debounce and the pending slot never set _gateway_accepted; the shared runtime does."""
        if self._shared_runtime_owns_input(event) or event.get_command():
            return False
        key = self._event_session_key(event)
        return key in (self.__dict__.get("_text_debounce") or {}) or key in self._pending_messages

    def _deferrable(self, event: MessageEvent, msg_data: dict[str, Any]) -> bool:
        """[PATCH-12] Refused because the sender's session was busy or its queue full (not a deliberate drop)."""
        if not self._done_folder or not msg_data["message_id"] or event.get_command():
            return False
        if msg_data["sender_addr"] in self._queue_full_chats:
            self._queue_full_chats.discard(msg_data["sender_addr"])
            return True
        try:
            return not self._session_idle(event)
        except Exception as e:
            logger.debug("[Email] Busy check failed for %s: %s", msg_data["message_id"], e)
            return False

    def _retry_note(self, msg_data: dict[str, Any], event: MessageEvent) -> bool:
        """[PATCH-12] Keep a refused mail in its folder for re-dispatch; True when the bound is hit (give up)."""
        mid = msg_data["message_id"]
        entry = self._retry.get(mid)
        now = time.monotonic()
        attempts, first = (entry[3], entry[2]) if entry else (0, now)
        if attempts >= _RETRY_MAX_ATTEMPTS or now - first > _RETRY_MAX_AGE_S:
            self._retry.pop(mid, None)
            logger.warning("[Email] Giving up re-dispatch of %s after %d attempts", mid, attempts)
            return True
        self._retry[mid] = [msg_data, event, first, attempts]
        while len(self._retry) > _INFLIGHT_MAX:
            dropped, _ = self._retry.popitem(last=False)
            logger.warning("[Email] Retry map full; leaving %s in its folder", dropped)
        logger.info("[Email] %s refused while session busy; will re-dispatch (attempt %d)", mid, attempts + 1)
        return False

    async def _retry_deferred(self) -> None:
        """[PATCH-12] Re-dispatch refused mails whose session is idle now."""
        for mid, (msg_data, event, first, attempts) in list(self._retry.items()):
            if time.monotonic() - first > _RETRY_MAX_AGE_S:
                del self._retry[mid]
                logger.warning("[Email] Re-dispatch of %s timed out", mid)
                await self._finalize_many([(mid, msg_data.get("source_folder", "INBOX"), "failed", True)])
                continue
            try:
                if not self._session_idle(event):
                    continue
            except Exception as e:
                logger.debug("[Email] Idle check failed for %s: %s", mid, e)
                continue
            missing = [a["path"] for a in msg_data["attachments"] if not os.path.exists(a["path"])]
            if missing:
                del self._retry[mid]
                logger.warning("[Email] Attachments of %s are gone (%s)", mid, missing)
                await self._finalize_many([(mid, msg_data.get("source_folder", "INBOX"), "failed", True)])
                continue
            self._retry[mid][3] = attempts + 1
            await self._dispatch_message(msg_data)

    def _inflight_register(self, message_id: str, source_folder: str, event: MessageEvent) -> None:
        """[PATCH-12] Track a dispatched mail until its turn completes (bounded; oldest stays in Working)."""
        if not message_id or not self._done_folder:
            return
        self._inflight[message_id] = (source_folder, event, time.monotonic())
        while len(self._inflight) > _INFLIGHT_MAX:
            dropped, _ = self._inflight.popitem(last=False)
            logger.warning("[Email] In-flight map full; leaving %s in its folder", dropped)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """[PATCH-12] Overridden so the runner also fires the hooks for queued follow-up turns."""
        await super().on_processing_start(event)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """[PATCH-12] Move the mail (and mails merged into its turn) to Done/Failed; CANCELLED keeps them."""
        try:
            await super().on_processing_complete(event, outcome)
        except Exception as e:
            logger.warning("[Email] on_processing_complete base hook failed: %s", e)
        try:
            ids = {event.message_id} | set(getattr(event, "_absorbed_ids", ()))
            entries = [(mid, self._inflight.pop(mid)) for mid in sorted(i for i in ids if i) if mid in self._inflight]
            if outcome == ProcessingOutcome.CANCELLED or not entries:
                return
            kind = "done" if outcome == ProcessingOutcome.SUCCESS else "failed"
            await self._finalize_many([(mid, entry[0], kind, False) for mid, entry in entries])
        except Exception as e:
            logger.error("[Email] Finalize on processing complete failed: %s", e)

    async def _finalize_many(self, items: list[tuple[str, str, str, bool]]) -> None:
        """[PATCH-12] Move each (message_id, source_folder, kind, strict) mail off the event loop."""

        def run() -> None:
            for message_id, source_folder, kind, strict in items:
                self._finalize_message(message_id, source_folder, self._target_folder(kind, strict=strict))

        await asyncio.get_running_loop().run_in_executor(None, run)

    def _session_idle(self, event: MessageEvent) -> bool:
        """[PATCH-12] True only when neither the adapter nor the runner holds a turn or queue for the event's session."""
        key = self._event_session_key(event)
        held = (self.__dict__.get("_text_debounce") or {}, self.__dict__.get("_shared_bursts") or {})
        if key in self._active_sessions or key in self._pending_messages or any(key in h for h in held):
            return False
        runner = self.gateway_runner
        if runner is None:
            return True
        if runner._is_session_running(key):
            return False
        state = runner._peek_session_state(key)
        return not (state is not None and state.conversation.queued_events)

    async def _sweep_inflight(self) -> None:
        """[PATCH-12] Finalize mails whose completion hook never fired (merged away or replaced in the pending slot)."""
        now = time.monotonic()
        stale: list[tuple[str, str, str, bool]] = []
        for mid, (folder, event, ts) in list(self._inflight.items()):
            if now - ts < _INFLIGHT_SWEEP_GRACE_S or mid in self._retry:
                continue
            try:
                idle = self._session_idle(event)
            except Exception as e:
                logger.debug("[Email] Idle check failed for %s: %s", mid, e)
                continue
            if idle:
                del self._inflight[mid]
                stale.append((mid, folder, "done", False))
                logger.warning("[Email] No completion hook for %s (outcome unknown); session idle, moving to Done", mid)
        if stale:
            await self._finalize_many(stale)

    # [PATCH-11] Quote-the-original-email helpers (dgxarley patch, opt-in).

    def _remember_original(self, msg_data: dict[str, Any]) -> None:
        """[PATCH-11] Keep what a later reply needs to quote this mail (bounded)."""
        message_id = (msg_data.get("message_id") or "").strip()
        record = {
            "message_id": message_id,
            "body": msg_data.get("body") or "",
            "date": msg_data.get("date") or "",
            "name": msg_data.get("sender_name") or "",
            "address": msg_data["sender_addr"],
            "state": "",
        }
        with self._quote_lock:
            if message_id:
                self._original_by_msg_id[message_id] = record
                self._original_by_msg_id.move_to_end(message_id)
                while len(self._original_by_msg_id) > _QUOTE_LOOKUP_MAX:
                    self._original_by_msg_id.popitem(last=False)
            self._last_original_by_sender[msg_data["sender_addr"]] = record
            self._last_original_by_sender.move_to_end(msg_data["sender_addr"])
            while len(self._last_original_by_sender) > _QUOTE_LOOKUP_MAX:
                self._last_original_by_sender.popitem(last=False)

    def _forget_sender_original(self, sender_addr: str) -> None:
        """[PATCH-11] Drop the per-sender fallback so a reply to a not-granted mail cannot quote an earlier one."""
        with self._quote_lock:
            self._last_original_by_sender.pop(sender_addr, None)

    def _claim_quote(
        self, to_addr: str, body: str, reply_to_msg_id: Optional[str], *, final: bool
    ) -> tuple[str, Optional[dict[str, str]]]:
        """[PATCH-11] ``(body with quote, claimed record)``; the record is ``None`` when nothing was quoted.

        Only a turn's final-reply send (``final``, from ``metadata["notify"]``) quotes: the claim marks the
        record pending so a concurrent send skips it, ``_settle_quote`` finalizes it after SMTP success or
        releases it on failure. The agent's own text is never shortened; the quote shrinks (or is dropped) to
        fit MAX_MESSAGE_LENGTH.
        """
        if not self._quote_original or not final:
            return body, None
        original_msg_id = (reply_to_msg_id or self._thread_context.get(to_addr, {}).get("message_id") or "").strip()
        with self._quote_lock:
            record = self._original_by_msg_id.get(original_msg_id) if original_msg_id else None
            if record is None:
                # Fall back to the sender's last mail only when it is the same mail (or carried no Message-ID),
                # never quote a different mail than the one being replied to.
                last = self._last_original_by_sender.get(to_addr)
                if last is not None and (not original_msg_id or last["message_id"] == original_msg_id):
                    record = last
            if record is None or record["address"] != to_addr or record["state"]:
                return body, None
            fixed = len(
                _build_quote(
                    "x",
                    date=record["date"],
                    name=record["name"],
                    address=record["address"],
                    header_fmt=self._quote_header,
                    max_chars=3,
                )
            ) - len("> x")
            budget = min(self._quote_max_chars, MAX_MESSAGE_LENGTH - len(body) - fixed)
            quote = _build_quote(
                record["body"],
                date=record["date"],
                name=record["name"],
                address=record["address"],
                header_fmt=self._quote_header,
                max_chars=budget,
            )
            if not quote:
                return body, None
            record["state"] = "pending"
        return (body + quote if body else quote.lstrip("\n")), record

    def _settle_quote(self, record: Optional[dict[str, str]], sent: bool) -> None:
        """[PATCH-11] Finalize a claimed quote after SMTP success, or release it so a retry quotes again."""
        if record is not None:
            with self._quote_lock:
                record["state"] = "done" if sent else ""

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking SMTP sender in the executor; wrap its Message-ID in a SendResult."""
        try:
            return SendResult(
                success=True, message_id=await asyncio.get_running_loop().run_in_executor(None, fn, *args)
            )
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None
    ) -> SendResult:
        """Send an email reply to the given address; [PATCH-11] only a turn's final-reply send quotes."""
        if _QUEUE_FULL_REPLY is not None and content == _QUEUE_FULL_REPLY:
            self._queue_full_chats.add(chat_id)  # [PATCH-12] retried by the adapter instead of telling the sender
            return SendResult(success=True)
        final = bool((metadata or {}).get("notify"))  # [PATCH-11]
        return await self._run_send(
            self._send_email, (chat_id, content, reply_to, final), "[Email] Send failed to %s: %s", chat_id
        )

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(
        self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, *, attach_empty_body: bool = False
    ) -> tuple[MIMEMultipart, str, str]:
        """Build a threaded reply skeleton. Returns ``(msg, msg_id, subject)``."""
        msg, ctx = MIMEMultipart(), self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        threading = (("In-Reply-To", original_msg_id), ("References", original_msg_id)) if original_msg_id else ()
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        for key, value in (
            ("From", self._address),
            ("To", to_addr),
            ("Subject", subject),
            *threading,
            ("Date", formatdate(localtime=True)),
            ("Message-ID", msg_id),
        ):
            msg[key] = value
        if body or attach_empty_body:
            msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg, msg_id, subject

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Login, send, and always release the SMTP connection (quit, else close)."""
        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()
        # [PATCH-7] Mirror outbound mail to Sent (no-op when configured empty).
        # _smtp_send is the single SMTP sink for _send_email and _send_with_files
        # (= _send_email_with_attachment{,s}), so one call site covers all three.
        self._append_to_sent(msg.as_bytes())

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, final: bool = False) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        # [PATCH-11] Claim before the MIME skeleton is built (so the quote lands in the same body the
        # Sent-folder APPEND archives); settle after _smtp_send so a retry after a failed send quotes again.
        body, quoted = self._claim_quote(to_addr, body, reply_to_msg_id, final=final)
        try:
            msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, attach_empty_body=True)
            self._smtp_send(msg)
        except BaseException:
            self._settle_quote(quoted, sent=False)
            raise
        self._settle_quote(quoted, sent=True)
        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    def _send_with_files(
        self,
        to_addr: str,
        body: str,
        files: list[tuple[Path, str]],
        *,
        lenient: bool,
        reply_to_msg_id: Optional[str] = None,
        final: bool = False,
    ) -> str:
        """Send a reply with attachments; *lenient* logs-and-skips unattachable files instead of raising.
        An explicit *reply_to_msg_id* threads the mail like ``_send_email`` does (#10131)."""
        # [PATCH-11] Same claim/settle contract as _send_email (see there); wraps the attach loop too so a
        # failed attachment (non-lenient) also releases the claim for the retry.
        body, quoted = self._claim_quote(to_addr, body, reply_to_msg_id, final=final)
        try:
            msg, msg_id, _ = self._new_reply(to_addr, body, reply_to_msg_id)
            for path, name in files:
                try:
                    _attach_file(msg, path, name)
                except Exception as e:
                    if not lenient:
                        raise
                    logger.warning("[Email] Failed to attach %s: %s", path, e)
            self._smtp_send(msg)
        except BaseException:
            self._settle_quote(quoted, sent=False)
            raise
        self._settle_quote(quoted, sent=True)
        return msg_id

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image URL as part of an email body."""
        return await self.send(chat_id, f"{caption or ''}\n\nImage: {image_url}".strip(), reply_to, metadata)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: list[tuple[str, str]],
        metadata: Optional[dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> SendResult:
        """One email per batch: local files attached, URL images linked in the body (no remote download); base-class fallback on failure."""
        if not images:
            return SendResult(success=False, error="no images to send")
        from urllib.parse import unquote as _unquote

        body_parts, local_paths = [], []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if not image_url.startswith("file://"):
                body_parts.append(t("platform.email.image_line", url=image_url))  # parity with send_image
            elif Path(local_path := _unquote(image_url[7:])).exists():
                local_paths.append(local_path)
            else:
                logger.warning("[Email] Skipping missing image: %s", local_path)
        if not local_paths and not body_parts:
            return SendResult(success=False, error="no valid images in batch")
        final = bool((metadata or {}).get("notify"))  # [PATCH-11]
        try:
            message_id = await asyncio.get_running_loop().run_in_executor(
                None, self._send_email_with_attachments, chat_id, "\n\n".join(body_parts), local_paths, final
            )
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        return SendResult(success=True, message_id=message_id)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: list[str], final: bool = False) -> str:
        """Send an email with multiple file attachments via SMTP (unattachable files are skipped)."""
        msg_id = self._send_with_files(
            to_addr, body, [(Path(f), Path(f).name) for f in file_paths], lenient=True, final=final
        )
        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        return msg_id

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as an email attachment; [PATCH-11] only a turn's final-reply send quotes."""
        final = bool((kwargs.get("metadata") or {}).get("notify"))  # [PATCH-11]
        return await self._run_send(
            self._send_email_with_attachment,
            (chat_id, caption or "", file_path, file_name, reply_to, final),
            "[Email] Send document failed: %s",
        )

    def _send_email_with_attachment(
        self,
        to_addr: str,
        body: str,
        file_path: str,
        file_name: Optional[str] = None,
        reply_to_msg_id: Optional[str] = None,
        final: bool = False,
    ) -> str:
        """Send an email with a single file attachment via SMTP (raises if unattachable)."""
        return self._send_with_files(
            to_addr,
            body,
            [(Path(file_path), file_name or Path(file_path).name)],
            lenient=False,
            reply_to_msg_id=reply_to_msg_id,
            final=final,
        )

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        """Return basic info about the email chat."""
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": self._thread_context.get(chat_id, {}).get("subject", ""),
        }


# Plugin glue: register() exposes the platform via the registry; EMAIL_* env → PlatformConfig seeding stays in core.
async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery via SMTP (one-shot); standalone_sender_fn contract."""
    extra = getattr(pconfig, "extra", {}) or {}
    address, password = extra.get("address") or _get_secret("EMAIL_ADDRESS", ""), _get_secret("EMAIL_PASSWORD", "")
    smtp_host, smtp_port = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", ""), _esecret_int(
        "EMAIL_SMTP_PORT", 587
    )
    smtp_security = _normalize_security(
        _get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"),
        default="tls" if smtp_port == 465 else "starttls",
    )
    smtp_tls_verify = _esecret_bool(
        "EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True)
    )
    # [PATCH-8] IMAP Sent-folder archival config (parity with EmailAdapter and
    # upstream PR #28697's standalone-path review fix). The standalone path only
    # requires SMTP, so IMAP may be unconfigured — the shared helper no-ops on a
    # missing host or an empty folder. Security/verify resolved exactly like
    # EmailAdapter.__init__, incl. the [PATCH-10] port-derived default.
    imap_host = (extra.get("imap_host") or _get_secret("EMAIL_IMAP_HOST", "")).strip()
    imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
    imap_security = _normalize_security(
        _get_secret("EMAIL_IMAP_SECURITY", "") or extra.get("imap_security"),
        default=_imap_default_security(imap_port),
    )
    imap_tls_verify = _esecret_bool(
        "EMAIL_IMAP_TLS_VERIFY", is_truthy_value(extra.get("imap_tls_verify"), default=True)
    )
    # Empty string is a deliberate opt-out — do NOT collapse with `or`.
    sent_folder = extra.get("sent_folder", "Sent")
    if not all([address, password, smtp_host]):
        return send_error("Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)")
    try:
        msg = MIMEText(message, "plain", "utf-8")
        for key, value in (
            ("From", address),
            ("To", chat_id),
            ("Subject", t("platform.email.standalone_subject")),
            ("Date", formatdate(localtime=True)),
        ):
            msg[key] = value
        server = _open_smtp(
            smtp_host,
            smtp_port,
            smtp_security,
            _tls_context(smtp_tls_verify, smtp_host),
            smtplib.SMTP,
            smtplib.SMTP_SSL,
        )
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        # [PATCH-8] Best-effort Sent-folder mirror — never fails the (already-completed) SMTP send.
        _imap_append_to_sent(
            imap_host=imap_host,
            imap_port=imap_port,
            imap_security=imap_security,
            imap_tls_verify=imap_tls_verify,
            address=address,
            password=password,
            sent_folder=sent_folder,
            raw_bytes=msg.as_bytes(),
        )
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e

            return _e(f"Email send failed: {e}")
        except Exception:
            return send_error(f"Email send failed: {e}")


def _is_connected(config) -> bool:
    """Connected when an address is configured (PlatformConfig.extra or EMAIL_ADDRESS)."""
    if (getattr(config, "extra", {}) or {}).get("address"):
        return True
    import hermes_cli.gateway as gateway_mod

    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are communicating via email. Write clear, well-structured responses suitable for email. Use "
    "plain text formatting (no markdown). Keep responses concise but complete. You can send file "
    "attachments — include MEDIA:/absolute/path/to/file in your response. The subject line is preserved "
    "for threading. Do not include greetings or sign-offs unless contextually appropriate."
)


_SEND_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)

    ctx.register_platform(
        target_ref_syntax_fn=regex_stripped_parser(_SEND_TARGET_RE),
        standalone_send=StandaloneSendPolicy(text_direct=True),
        name="email",
        label="Email",
        adapter_factory=EmailAdapter,
        check_fn=check_email_requirements,
        is_connected=_is_connected,
        dashboard_description="Talk to Hermes through an IMAP/SMTP mailbox.",
        docs_url="https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        dashboard_env_vars=("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        dashboard_required_env=("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib) — no extra deps",
        cron_deliver_env_var="EMAIL_HOME_ADDRESS",
        standalone_sender_fn=_standalone_send,
        max_message_length=50_000,
        pii_safe=True,
        emoji="📧",
        allow_update_command=True,
        platform_hint=_PLATFORM_HINT,
        display_tier="minimal",
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS,
        **gateway_env.AUTHZ,
    )
