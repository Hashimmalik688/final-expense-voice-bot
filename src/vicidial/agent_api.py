"""
src/vicidial/agent_api.py
══════════════════════════
VICIdial HTTP API client for the Final Expense Voice Bot.

Handles all interactions with the VICIdial backend:
  • Lead data retrieval and field updates
  • Disposition management
  • Call transfer coordination
  • DNC (Do Not Call) registry
  • Callback scheduling
  • Notes / comment synchronisation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VICIDIAL API OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VICIdial exposes two HTTP API scripts, both accepting GET or POST:

  1.  agent_api.php   — requires active agent session (agent_user + agent_pass)
  2.  non_agent_api.php — admin-level API, requires api_user + api_pass

All responses are plain text in one of two formats:
  • "SUCCESS: <value>" / "ERROR: <message>"   (most agent_api.php calls)
  • Pipe-delimited fields with a header record    (non_agent_api.php)

Authentication
──────────────
  agent_api.php requires:
    agent_user, agent_pass, source, campaign_id, and usually call_id

  non_agent_api.php requires:
    source, user (admin user), pass (admin pass), function=<function_name>

Common function values (agent_api.php)
───────────────────────────────────────
  agent_login, agent_logout, agent_pause, agent_resume,
  blind_transfer, warm_transfer, hangup, set_disposition,
  schedule_callback

Common function values (non_agent_api.php)
───────────────────────────────────────────
  get_lead_info       → returns lead fields for a lead_id
  update_lead         → update any field on a lead record
  add_dnc             → add a phone to the system DNC list
  add_list_leads      → insert a new lead
  get_active_calls    → list all active calls on the system

Disposition codes used by this bot
────────────────────────────────────
  QUALIFIED   — lead qualified, transferred to closer
  NOT_INT     — not interested
  NO_ANS      — did not answer (voicemail)
  DNC         — do not call (lead requested removal)
  CALLBACK    — schedule a callback
  DISC        — disconnected / bad number
  WRONG       — wrong number

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED ENVIRONMENT VARIABLES  (see .env.example)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VICIDIAL_API_URL            http://192.168.1.10/vicidial
  VICIDIAL_API_USER           admin
  VICIDIAL_API_PASS           password123
  VICIDIAL_CAMPAIGN_ID        FINALEXP
  VICIDIAL_AGENT_USER         voicebot
  VICIDIAL_AGENT_PASS         botpass
  VICIDIAL_TRANSFER_EXT       8300
  VICIDIAL_CLOSER_INGROUP     CLOSERS
  VICIDIAL_CLOSER_LIST_ID     <list ID for the closer queue>
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

from config.settings import VICIdialConfig, get_vicidial_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeadInfo:
    """Structured representation of a VICIdial lead record."""

    lead_id:     str = ""
    first_name:  str = ""
    last_name:   str = ""
    phone:       str = ""
    alt_phone:   str = ""
    email:       str = ""
    address1:    str = ""
    city:        str = ""
    state:       str = ""
    zip_code:    str = ""
    dob:         str = ""       # date of birth  (YYYY-MM-DD)
    comments:    str = ""
    list_id:     str = ""
    status:      str = ""       # VICIdial lead status code

    # Custom fields (VICIdial supports up to 20 custom columns)
    custom_field1: str = ""     # used for: beneficiary name
    custom_field2: str = ""     # used for: coverage amount requested
    custom_field3: str = ""     # used for: health conditions disclosed
    custom_field4: str = ""     # used for: SSN last 4
    custom_field5: str = ""     # used for: closer notes

    raw: dict = field(default_factory=dict, repr=False)

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p).strip()


@dataclass
class CallDisposition:
    """Result of a VICIdial disposition call."""

    success:      bool
    disposition:  str
    lead_id:      str
    message:      str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Retry / backoff helper
# ─────────────────────────────────────────────────────────────────────────────

RETRY_DELAYS = [1.0, 2.0, 4.0]   # seconds between retries (3 attempts total)
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


# ─────────────────────────────────────────────────────────────────────────────
# Main API client
# ─────────────────────────────────────────────────────────────────────────────

class AgentAPI:
    """Async HTTP client for VICIdial's agent_api.php and non_agent_api.php.

    Usage::

        api = AgentAPI()
        await api.initialize()
        await api.login()
        ...
        lead = await api.get_lead_info(lead_id="12345")
        await api.set_disposition(lead_id="12345", call_id="abc", disposition="QUALIFIED")
        await api.add_call_notes(lead_id="12345", notes="Qualified for Plan B, $50/mo")
        await api.blind_transfer(call_id="abc", extension="8300")
        await api.logout()
        await api.shutdown()
    """

    def __init__(self, config: Optional[VICIdialConfig] = None) -> None:
        self._config  = config or get_vicidial_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._logged_in = False
        self._agent_session_id: str = ""    # returned by agent_login

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the underlying HTTP session.  Call before any API method."""
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": "FinalExpenseVoiceBot/1.0"},
        )
        logger.info("AgentAPI session created  (url=%s)", self._config.api_url)

    async def shutdown(self) -> None:
        """Logout and close the HTTP session."""
        if self._logged_in:
            await self.logout()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("AgentAPI session closed.")

    # ─────────────────────────────────────────────────────────────────────
    # Internal HTTP helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _http_get(self, url: str) -> str:
        """Perform one HTTP GET and return the response body as a string.

        Raises ``aiohttp.ClientResponseError`` for 4xx/5xx responses,
        and ``aiohttp.ClientConnectionError`` for network failures.
        """
        if self._session is None:
            raise RuntimeError("AgentAPI not initialized — call initialize() first.")
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def _retry_request(
        self,
        url: str,
        max_attempts: int = 3,
    ) -> str:
        """Wrap ``_http_get`` with exponential back-off retry.

        Retries on:
          • aiohttp.ClientConnectionError  (network failure)
          • aiohttp.ClientResponseError with a transient HTTP status code

        Non-transient errors (400, 401, 403, 404) are raised immediately.

        Returns the raw response body on success.
        """
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate(RETRY_DELAYS[:max_attempts], start=1):
            try:
                return await self._http_get(url)
            except aiohttp.ClientResponseError as exc:
                if exc.status not in TRANSIENT_HTTP_CODES:
                    raise   # permanent error — don't retry
                last_exc = exc
                logger.warning(
                    "VICIdial API HTTP %d on attempt %d/%d — retry in %.1fs",
                    exc.status, attempt, max_attempts, delay,
                )
            except aiohttp.ClientConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "VICIdial API connection error on attempt %d/%d — retry in %.1fs: %s",
                    attempt, max_attempts, delay, exc,
                )
            if attempt < max_attempts:
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"VICIdial API unreachable after {max_attempts} attempts"
        ) from last_exc

    def _build_agent_url(self, params: dict[str, Any]) -> str:
        """Build a complete agent_api.php URL with auth params."""
        base = {
            "agent_user":  self._config.agent_user,
            "agent_pass":  self._config.agent_pass,
            "source":      "bot",
            "campaign_id": self._config.campaign_id,
        }
        base.update(params)
        return f"{self._config.api_url}/agc/agent_api.php?{urlencode(base)}"

    def _build_admin_url(self, params: dict[str, Any]) -> str:
        """Build a complete non_agent_api.php URL with auth params."""
        base = {
            "user":   self._config.api_user,
            "pass":   self._config.api_pass,
            "source": "bot",
        }
        base.update(params)
        return f"{self._config.api_url}/vicidial/non_agent_api.php?{urlencode(base)}"

    @staticmethod
    def _parse_response(raw: str) -> tuple[bool, str]:
        """Parse a VICIdial 'SUCCESS: value' / 'ERROR: message' response.

        Returns ``(True, value)`` or ``(False, error_message)``.
        """
        raw = raw.strip()
        if raw.upper().startswith("SUCCESS"):
            return True, raw.split(":", 1)[-1].strip() if ":" in raw else ""
        if raw.upper().startswith("ERROR"):
            return False, raw.split(":", 1)[-1].strip() if ":" in raw else raw
        # Some endpoints return a bare value without a SUCCESS/ERROR prefix
        return True, raw

    @staticmethod
    def _parse_pipe_response(raw: str) -> list[dict[str, str]]:
        """Parse a pipe-delimited VICIdial response into a list of dicts.

        VICIdial non_agent_api.php responses look like:
          field1|field2|field3
          val1  |val2  |val3

        The first line is always a header row.
        Subsequent lines are data rows.
        Extra whitespace around pipe characters is stripped.
        """
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        if not lines:
            return []
        headers = [h.strip() for h in lines[0].split("|")]
        result: list[dict[str, str]] = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split("|")]
            row = dict(zip(headers, values))
            result.append(row)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────

    async def login(self) -> bool:
        """Log the bot agent into VICIdial.

        VICIdial tracks which agent is on a call; this must be called
        before making dispositions or transfers.

        Returns ``True`` on success.
        """
        url = self._build_agent_url({"function": "agent_login"})
        logger.info("VICIdial agent login: %s", self._config.agent_user)
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if ok:
                self._logged_in = True
                self._agent_session_id = val
                logger.info("VICIdial login OK  session=%s", val)
            else:
                logger.error("VICIdial login failed: %s", val)
            return ok
        except Exception:
            logger.exception("VICIdial login error")
            return False

    async def logout(self) -> bool:
        """Log out the agent session from VICIdial."""
        url = self._build_agent_url({"function": "agent_logout"})
        try:
            raw = await self._retry_request(url)
            ok, _ = self._parse_response(raw)
            self._logged_in = False
            logger.info("VICIdial logout %s", "OK" if ok else "failed")
            return ok
        except Exception:
            logger.exception("VICIdial logout error")
            self._logged_in = False
            return False

    async def pause(self, pause_code: str = "PAUSE") -> bool:
        """Pause the agent (no new calls assigned while paused)."""
        url = self._build_agent_url({
            "function":   "agent_pause",
            "pause_code": pause_code,
        })
        try:
            raw = await self._retry_request(url)
            ok, _ = self._parse_response(raw)
            return ok
        except Exception:
            logger.exception("VICIdial pause error")
            return False

    async def unpause(self) -> bool:
        """Resume the agent after a pause."""
        url = self._build_agent_url({
            "function":   "agent_resume",
        })
        try:
            raw = await self._retry_request(url)
            ok, _ = self._parse_response(raw)
            return ok
        except Exception:
            logger.exception("VICIdial unpause error")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Lead information
    # ─────────────────────────────────────────────────────────────────────

    async def get_lead_info(self, lead_id: str) -> Optional[LeadInfo]:
        """Retrieve all fields for a VICIdial lead.

        Uses non_agent_api.php ``function=get_lead_info``.

        Returns a populated ``LeadInfo`` or ``None`` on error.
        """
        url = self._build_admin_url({
            "function": "get_lead_info",
            "lead_id":  lead_id,
        })
        try:
            raw = await self._retry_request(url)
            # Response is pipe-delimited (header + data row)
            rows = self._parse_pipe_response(raw)
            if not rows:
                logger.warning("No lead found for lead_id=%s", lead_id)
                return None
            r = rows[0]
            return LeadInfo(
                lead_id     = r.get("lead_id",     lead_id),
                first_name  = r.get("first_name",  ""),
                last_name   = r.get("last_name",   ""),
                phone       = r.get("phone_number",""),
                alt_phone   = r.get("alt_phone",   ""),
                email       = r.get("email",       ""),
                address1    = r.get("address1",    ""),
                city        = r.get("city",        ""),
                state       = r.get("state",       ""),
                zip_code    = r.get("zip_code",    ""),
                dob         = r.get("date_of_birth",""),
                comments    = r.get("comments",    ""),
                list_id     = r.get("list_id",     ""),
                status      = r.get("status",      ""),
                custom_field1 = r.get("custom1",   ""),
                custom_field2 = r.get("custom2",   ""),
                custom_field3 = r.get("custom3",   ""),
                custom_field4 = r.get("custom4",   ""),
                custom_field5 = r.get("custom5",   ""),
                raw         = r,
            )
        except Exception:
            logger.exception("get_lead_info error for lead_id=%s", lead_id)
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Lead field update
    # ─────────────────────────────────────────────────────────────────────

    async def update_lead_fields(
        self,
        lead_id: str,
        fields: dict[str, str],
    ) -> bool:
        """Update one or more fields on a VICIdial lead record.

        Supported standard fields:
            first_name, last_name, email, address1, city, state, zip_code,
            phone_number, alt_phone, date_of_birth, comments

        Supported custom fields:
            custom1 … custom20

        Example::

            await api.update_lead_fields("12345", {
                "comments": "Interested in Plan B",
                "custom4":  "1234",   # SSN last 4
            })

        Uses non_agent_api.php ``function=update_lead``.
        Returns ``True`` on success.
        """
        if not fields:
            return True
        params: dict[str, Any] = {"function": "update_lead", "lead_id": lead_id}
        params.update(fields)
        url = self._build_admin_url(params)
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if not ok:
                logger.error("update_lead_fields failed: %s  (lead=%s)", val, lead_id)
            return ok
        except Exception:
            logger.exception("update_lead_fields error for lead_id=%s", lead_id)
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Call notes
    # ─────────────────────────────────────────────────────────────────────

    async def add_call_notes(
        self,
        lead_id: str,
        notes: str,
        *,
        append: bool = True,
    ) -> bool:
        """Add or replace the comments field on a lead.

        By default (``append=True``), new notes are appended to any
        existing comments with a timestamp prefix.  Set ``append=False``
        to replace the comments field entirely.

        VICIdial stores up to 65 535 characters in the ``comments`` field.

        Example::

            await api.add_call_notes("12345",
                "Interested in Plan B $49/mo, beneficiary = spouse, "
                "has diabetes, SSN last 4 = 1234")
        """
        if append:
            # Fetch existing comments first so we can prepend new notes
            lead = await self.get_lead_info(lead_id)
            existing = (lead.comments if lead else "").strip()
            ts = time.strftime("%Y-%m-%d %H:%M")
            notes = f"[{ts}] {notes}\n{existing}".strip()[:65_535]

        return await self.update_lead_fields(lead_id, {"comments": notes})

    # ─────────────────────────────────────────────────────────────────────
    # List / queue management
    # ─────────────────────────────────────────────────────────────────────

    async def move_lead_to_list(
        self,
        lead_id: str,
        list_id: str,
    ) -> bool:
        """Move a lead to a different VICIdial list (e.g. closer's queue).

        When the bot qualifies a lead and transfers, the lead should be
        moved to the closer's list so VICIdial assigns it correctly in
        the CRM.

        Uses non_agent_api.php ``function=update_lead`` with ``list_id``.
        """
        return await self.update_lead_fields(lead_id, {"list_id": list_id})

    # ─────────────────────────────────────────────────────────────────────
    # Disposition
    # ─────────────────────────────────────────────────────────────────────

    async def set_disposition(
        self,
        lead_id:     str,
        call_id:     str,
        disposition: str,
    ) -> CallDisposition:
        """Set the VICIdial disposition code after a call ends.

        Common disposition codes:
          QUALIFIED  — transferred to closer
          NOT_INT    — not interested
          CALLBACK   — requested call back
          DNC        — do not call (same as add_to_dnc, but in campaign)
          NO_ANS     — no answer / voicemail
          DISC       — disconnected

        Parameters
        ----------
        lead_id:     VICIdial lead_id (as string)
        call_id:     VICIdial uniqueid / call_id for the current call
        disposition: Disposition code string (must exist in campaign set-up)
        """
        url = self._build_agent_url({
            "function":    "disposition_log",
            "lead_id":     lead_id,
            "uniqueid":    call_id,
            "disposition": disposition,
            "campaign_id": self._config.campaign_id,
        })
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if not ok:
                logger.error(
                    "set_disposition failed: %s  lead=%s disp=%s",
                    val, lead_id, disposition,
                )
            return CallDisposition(
                success=ok,
                disposition=disposition,
                lead_id=lead_id,
                message=val,
            )
        except Exception:
            logger.exception("set_disposition error  lead=%s", lead_id)
            return CallDisposition(
                success=False,
                disposition=disposition,
                lead_id=lead_id,
                message="Exception during API call",
            )

    # ─────────────────────────────────────────────────────────────────────
    # Transfer helpers
    # ─────────────────────────────────────────────────────────────────────

    async def transfer_call(
        self,
        call_id:   str,
        extension: str,
        *,
        transfer_type: str = "BLIND",
    ) -> bool:
        """Tell VICIdial to transfer the call to *extension*.

        This is distinct from the SIP REFER that ``SIPHandler.blind_transfer``
        performs.  Use this to notify VICIdial's UI so the agent panel
        updates correctly.

        Parameters
        ----------
        call_id:
            VICIdial uniqueid for the call.
        extension:
            Destination extension (e.g. "8300") or in-group (e.g.
            "CLOSERS").
        transfer_type:
            "BLIND" (default) or "WARM".
        """
        url = self._build_agent_url({
            "function":        f"{transfer_type.lower()}_transfer",
            "uniqueid":        call_id,
            "xferextension":   extension,
            "campaign_id":     self._config.campaign_id,
        })
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if not ok:
                logger.error("transfer_call failed: %s", val)
            return ok
        except Exception:
            logger.exception("transfer_call error  call=%s  ext=%s", call_id, extension)
            return False

    async def blind_transfer(
        self,
        call_id:   str,
        extension: Optional[str] = None,
    ) -> bool:
        """Convenience wrapper for a blind transfer to the closer extension."""
        ext = extension or self._config.transfer_extension
        return await self.transfer_call(call_id, ext, transfer_type="BLIND")

    # ─────────────────────────────────────────────────────────────────────
    # Atomic qualify + transfer operation
    # ─────────────────────────────────────────────────────────────────────

    async def note_disposition_and_transfer(
        self,
        lead_id:       str,
        call_id:       str,
        disposition:   str,
        notes:         str,
        closer_list_id: Optional[str] = None,
    ) -> bool:
        """Combined atomic "qualify and hand off" operation.

        Performs in order:
          1.  Set VICIdial disposition (e.g. "QUALIFIED")
          2.  Append call notes capturing everything the bot collected
          3.  Move lead to closer's list (if *closer_list_id* is given)

        Returns ``True`` only if all three steps succeed.

        Example::

            await api.note_disposition_and_transfer(
                lead_id="12345",
                call_id="1234567890.001",
                disposition="QUALIFIED",
                notes=(
                    "Age 67, NC, no major health issues. "
                    "Interested in Plan B $49/mo. "
                    "Beneficiary: wife Susan. SSN last 4: 1234."
                ),
                closer_list_id="1002",
            )
        """
        results: list[bool] = []

        # Step 1 — disposition
        disp_result = await self.set_disposition(lead_id, call_id, disposition)
        results.append(disp_result.success)

        # Step 2 — notes (always attempt even if disposition failed)
        notes_ok = await self.add_call_notes(lead_id, notes)
        results.append(notes_ok)

        # Step 3 — move to closer list (only if list_id provided)
        if closer_list_id:
            move_ok = await self.move_lead_to_list(lead_id, closer_list_id)
            results.append(move_ok)

        all_ok = all(results)
        if not all_ok:
            logger.warning(
                "note_disposition_and_transfer partially failed  lead=%s  results=%s",
                lead_id, results,
            )
        return all_ok

    # ─────────────────────────────────────────────────────────────────────
    # Callback scheduling
    # ─────────────────────────────────────────────────────────────────────

    async def schedule_callback(
        self,
        lead_id:   str,
        call_id:   str,
        callback_datetime: str,
        *,
        notes:     str = "",
        timezone:  str = "US/Eastern",
    ) -> bool:
        """Schedule a future callback for a lead.

        Parameters
        ----------
        callback_datetime:
            ISO-8601 format: "YYYY-MM-DD HH:MM:SS" (24-hour, agent TZ).
        timezone:
            Timezone string recognised by VICIdial (default US/Eastern).
        """
        url = self._build_agent_url({
            "function":          "schedule_callback",
            "lead_id":           lead_id,
            "uniqueid":          call_id,
            "callback_datetime": callback_datetime,
            "callback_comments": notes[:255],
            "user":              self._config.agent_user,
            "calltype":          "ANYONE",
            "timezone":          timezone,
        })
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if not ok:
                logger.error("schedule_callback failed: %s", val)
            return ok
        except Exception:
            logger.exception("schedule_callback error  lead=%s", lead_id)
            return False

    # ─────────────────────────────────────────────────────────────────────
    # DNC
    # ─────────────────────────────────────────────────────────────────────

    async def add_to_dnc(self, phone_number: str) -> bool:
        """Add *phone_number* to the system-wide VICIdial DNC list.

        VICIdial will automatically skip this number in future campaigns.
        Phone number should be 10 digits, digits only (no dashes/spaces).

        Returns ``True`` on success.
        """
        # Sanitise: keep digits only
        digits_only = "".join(c for c in phone_number if c.isdigit())
        if len(digits_only) not in (10, 11):
            logger.warning("add_to_dnc: unexpected phone length: %s", phone_number)

        url = self._build_admin_url({
            "function": "add_dnc",
            "phone_number": digits_only,
        })
        try:
            raw = await self._retry_request(url)
            ok, val = self._parse_response(raw)
            if not ok:
                logger.error("add_to_dnc failed: %s  phone=%s", val, digits_only)
            else:
                logger.info("DNC added: %s", digits_only)
            return ok
        except Exception:
            logger.exception("add_to_dnc error  phone=%s", digits_only)
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Active call monitoring
    # ─────────────────────────────────────────────────────────────────────

    async def get_active_calls(self) -> list[dict[str, str]]:
        """Return a list of all currently active calls on the VICIdial system.

        Each entry in the returned list is a dict with keys matching the
        VICIdial non_agent_api.php ``get_active_calls`` field names, e.g.:
            lead_id, phone_number, campaign_id, start_epoch, uniqueid, status

        Returns an empty list on error.
        """
        url = self._build_admin_url({"function": "get_active_calls"})
        try:
            raw = await self._retry_request(url)
            return self._parse_pipe_response(raw)
        except Exception:
            logger.exception("get_active_calls error")
            return []

    # ─────────────────────────────────────────────────────────────────────    # Backward-compatible aliases
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_vicidial_response(raw: str) -> dict[str, str]:
        """Parse a pipe+equals-delimited VICIdial response into a dict.

        Handles the ``key=value|key=value`` format returned by some
        non_agent_api.php calls (e.g. get_lead_info compact mode).

        Example::

            "first_name=John|last_name=Doe|state=TX"
            → {"first_name": "John", "last_name": "Doe", "state": "TX"}
        """
        result: dict[str, str] = {}
        for part in raw.strip().split("|"):
            part = part.strip()
            if "=" in part:
                k, _, v = part.partition("=")
                result[k.strip()] = v.strip()
        return result

    # ─────────────────────────────────────────────────────────────────    # Introspection
    # ─────────────────────────────────────────────────────────────────────

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    @property
    def api_url(self) -> str:
        return self._config.api_url
