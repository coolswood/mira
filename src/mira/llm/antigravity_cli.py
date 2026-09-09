"""Antigravity CLI-backed provider using Google's ``agy`` (Gemini models).

Mirrors :class:`~mira.llm.codex_cli.CodexCLIProvider`: callers pass Mira's
existing chat messages/tool schemas and receive the same JSON strings an
OpenAI-compatible provider would have returned; only the execution backend
differs. The model runs through ``agy`` headless mode instead of an HTTP API
call.

Auth works in one of two ways (checked per call, no setup-time requirement):

- ``GEMINI_API_KEY`` (``llm.antigravity_api_key`` or the environment): the CLI
  requires ``~/.gemini/antigravity-cli/settings.json`` with
  ``{"modelProvider": "gemini"}`` for key auth — the provider writes that file
  into the ephemeral HOME automatically.
- ``llm.antigravity_home``: a trusted directory copied over the ephemeral
  ``~/.gemini`` (settings plus cached Google-account credentials from
  ``agy login``), analogous to ``llm.codex_home`` for Codex's auth.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import tempfile
from contextlib import suppress
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mira.config import LLMConfig
from mira.exceptions import LLMError

logger = logging.getLogger(__name__)


def _parse_stream_event(raw: bytes | str) -> dict | None:
    """Parse one ``agy --output-format stream-json`` NDJSON line.

    Returns None for blank lines, non-JSON noise, or non-object payloads —
    the stream is best-effort telemetry and must never break the call.
    """
    line = (raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw).strip()
    if not line.startswith("{"):
        return None
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }
)

# Model ids that mean "let the CLI pick" — no --model flag is passed.
_MODEL_SENTINELS = {"", "default", "antigravity-default", "agy-default"}

# Mira reasoning efforts accepted by the CLI's --effort flag; stronger levels
# clamp to "high" (the CLI maximum).
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class AntigravityCLIProvider:
    """LLM provider that shells out to Google Antigravity CLI (``agy``).

    Auth is handled by the CLI itself: either a Gemini API key
    (``GEMINI_API_KEY`` plus a settings.json selecting the ``gemini`` model
    provider, written automatically) or cached Google-account credentials
    copied from ``llm.antigravity_home``. No API key is read by Mira itself
    beyond forwarding it to the child process environment.
    """

    supports_json_mode: bool = True
    supports_tool_calling: bool = False
    supports_temperature: bool = False
    supports_max_tokens: bool = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Progress attribution: when set (by webhook handlers / indexing
        # callers) to a ProgressTracker job key, call-level events are
        # published to the dashboard's live progress view — same contract
        # as CodexCLIProvider.
        self.progress_key: str | None = None

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def _gemini_api_key(self) -> str:
        return self.config.antigravity_api_key or os.environ.get("GEMINI_API_KEY", "")

    def _env(self, runtime_home: str) -> dict[str, str]:
        """Build a minimal child environment without Mira service credentials."""
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
        env["HOME"] = runtime_home
        api_key = self._gemini_api_key()
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        return env

    def _prepare_gemini_home(self, invocation_root: str) -> None:
        """Populate the ephemeral HOME's ``.gemini`` directory for one call.

        With ``antigravity_home`` configured, its contents are trusted
        operator state (settings.json, cached login credentials) and are
        copied verbatim. With API-key auth, a minimal settings.json selecting
        the Gemini provider is written unless the copied home already has one.
        """
        gemini_dir = Path(invocation_root) / "gemini"
        gemini_dir.mkdir(parents=True, mode=0o700)
        source_home = self.config.antigravity_home
        if source_home:
            source = Path(source_home).expanduser()
            if not source.is_dir():
                raise LLMError("antigravity_home_missing", path=str(source))
            shutil.copytree(source, gemini_dir, dirs_exist_ok=True)
        if self._gemini_api_key():
            settings_dir = gemini_dir / "antigravity-cli"
            settings_dir.mkdir(exist_ok=True)
            settings = settings_dir / "settings.json"
            if not settings.exists():
                settings.write_text(json.dumps({"modelProvider": "gemini"}))
                settings.chmod(0o600)

    def _command(self) -> list[str]:
        agy_command = self.config.antigravity_command or "agy"
        if any(char in agy_command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid antigravity_command: {agy_command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        cmd = [agy_command, "--output-format", "stream-json"]
        if self.config.antigravity_sandbox:
            cmd.append("--sandbox")
        cmd.extend(["--print-timeout", f"{self.config.antigravity_timeout_seconds}s"])
        if self.config.model not in _MODEL_SENTINELS:
            cmd.extend(["--model", self.config.model])
        effort = _EFFORT_MAP.get(self.config.reasoning_effort or "")
        if effort:
            cmd.extend(["--effort", effort])
        # The prompt is fed on stdin: with piped stdin and no --print value the
        # CLI runs a single headless turn. (--print only accepts the prompt as
        # an argv value, which does not survive multi-hundred-KB review diffs.)
        return cmd

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate agy and model-spawned descendants before returning."""
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
        elif proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
        if proc.returncode is None:
            await proc.wait()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(LLMError),
        reraise=True,
    )
    async def _run_antigravity(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Run one agy headless call; return (final text, token usage).

        stdout is an NDJSON event stream (``--output-format stream-json``):
        ``init``, per-step ``step_update`` events, and a terminal ``result``
        envelope carrying status, the final response, and exact token usage.
        Events are parsed as they arrive — both for liveness logging and, when
        ``progress_key`` is set, for the dashboard's live progress view. The
        ``result`` envelope is the primary result channel; accumulated
        ``text_delta`` chunks are the fallback.
        """
        with tempfile.TemporaryDirectory(prefix="mira-antigravity-") as tmpdir:
            runtime_home = str(Path(tmpdir) / "runtime")
            Path(runtime_home).mkdir(mode=0o700)
            self._prepare_gemini_home(tmpdir)
            cmd = self._command()
            logger.debug("Running Antigravity CLI provider: %s", shlex.join(cmd + ["<stdin>"]))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._env(runtime_home),
                    cwd=runtime_home,
                    start_new_session=os.name == "posix",
                )
            except FileNotFoundError as exc:
                raise LLMError(
                    "antigravity_command_not_found", command=self.config.antigravity_command
                ) from exc

            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None
            # Local bindings: the asserts above don't narrow inside closures.
            proc_stdin = proc.stdin
            proc_stdout = proc.stdout
            proc_stderr = proc.stderr

            loop = asyncio.get_running_loop()
            started = loop.time()
            progress_key = self.progress_key
            if progress_key:
                from mira.core.progress import tracker

                tracker.call_started(progress_key)

            stderr_chunks: list[bytes] = []

            async def _drain_stderr() -> None:
                async for chunk in proc_stderr:
                    stderr_chunks.append(chunk)

            async def _write_stdin() -> None:
                """Feed the prompt and close stdin.

                Runs concurrently with stdout pumping: a multi-megabyte chunk
                prompt overflows the pipe buffer, so writing must not wait for
                the pump to finish first. The explicit close matters — the CLI
                reads the prompt until stdin EOF.
                """
                try:
                    proc_stdin.write(prompt.encode("utf-8"))
                    await proc_stdin.drain()
                    proc_stdin.close()
                    await proc_stdin.wait_closed()
                except Exception:
                    pass

            async def _pump_stdout() -> tuple[str, dict[str, int]]:
                """Consume stream-json events until EOF.

                Returns (final response text, normalized token usage).
                """
                text = ""
                usage: dict[str, int] = {}
                envelope_error = ""
                while True:
                    raw = await proc_stdout.readline()
                    if not raw:
                        break
                    event = _parse_stream_event(raw)
                    if event is None:
                        continue
                    kind = event.get("event", "")
                    if kind == "step_update":
                        step = event.get("step_update") or {}
                        step_type = str(step.get("step_type", ""))
                        if step_type == "agent_response" and step.get("text_delta"):
                            text += str(step["text_delta"])
                        if step_type:
                            logger.debug(
                                "antigravity step: %s (%s)",
                                step_type,
                                step.get("state", ""),
                            )
                            if progress_key:
                                tracker.call_item(progress_key, step_type)
                    elif kind == "result":
                        result = event.get("result") or {}
                        if result.get("response"):
                            # The terminal envelope is authoritative; deltas
                            # are only a fallback when it never arrives.
                            text = str(result["response"])
                        raw_usage = result.get("usage") or {}
                        usage = {
                            "input_tokens": int(raw_usage.get("input_tokens") or 0),
                            "cached_input_tokens": int(raw_usage.get("cache_read_tokens") or 0),
                            "output_tokens": int(raw_usage.get("output_tokens") or 0),
                            "reasoning_output_tokens": int(raw_usage.get("thinking_tokens") or 0),
                        }
                        if progress_key and usage:
                            tracker.call_usage(progress_key, **usage)
                        envelope_error = str(result.get("error") or "")
                        status = str(result.get("status", ""))
                        if status and status != "SUCCESS":
                            logger.warning(
                                "antigravity result status %s: %s",
                                status,
                                envelope_error[:500],
                            )
                    elif kind == "init":
                        logger.debug("antigravity session initialized")
                    # Unknown event kinds are skipped; the stream must never
                    # break the call.
                if envelope_error and not usage:
                    logger.warning("antigravity envelope error: %s", envelope_error[:500])
                return text, usage

            async def _heartbeat() -> None:
                """Emit a liveness line while the call runs — reasoning-heavy
                calls can go minutes between stream events."""
                while True:
                    await asyncio.sleep(30)
                    logger.info(
                        "antigravity call in progress: %ds elapsed (timeout %ds)",
                        int(loop.time() - started),
                        self.config.antigravity_timeout_seconds,
                    )

            stderr_task = asyncio.ensure_future(_drain_stderr())
            stdin_task = asyncio.ensure_future(_write_stdin())
            heartbeat = asyncio.ensure_future(_heartbeat())
            try:
                agent_text, usage = await asyncio.wait_for(
                    _pump_stdout(),
                    timeout=self.config.antigravity_timeout_seconds,
                )
                try:
                    exit_code = await asyncio.wait_for(proc.wait(), timeout=30)
                except TimeoutError:
                    await self._terminate_process_tree(proc)
                    exit_code = await proc.wait()
            except TimeoutError as exc:
                heartbeat.cancel()
                await self._terminate_process_tree(proc)
                if progress_key:
                    tracker.call_finished(
                        progress_key, ok=False, duration_s=loop.time() - started, error="timeout"
                    )
                raise LLMError(
                    "antigravity_timeout", seconds=self.config.antigravity_timeout_seconds
                ) from exc
            except BaseException as exc:
                heartbeat.cancel()
                await self._terminate_process_tree(proc)
                if progress_key:
                    tracker.call_finished(
                        progress_key,
                        ok=False,
                        duration_s=loop.time() - started,
                        error=type(exc).__name__,
                    )
                raise
            finally:
                heartbeat.cancel()
                # A task that never got scheduled completes via cancel() with
                # CancelledError, which is BaseException — suppress it too.
                for task in (stderr_task, stdin_task):
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

            duration = loop.time() - started
            if progress_key:
                tracker.call_finished(progress_key, ok=exit_code == 0, duration_s=duration)
            self._log_call_finished(duration, usage)

            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

            if exit_code != 0:
                detail = (stderr_text or agent_text).strip()
                raise LLMError(
                    "antigravity_exit_failed",
                    exit_code=exit_code,
                    detail=detail[-2000:],
                )

            return agent_text.strip(), usage

    @staticmethod
    def _log_call_finished(duration_s: float, usage: dict[str, int]) -> None:
        """One log line per finished call: duration plus real token spend."""
        if usage:
            logger.info(
                "antigravity call finished in %ds (tokens: %d in / %d cached / %d out / %d reasoning)",
                int(duration_s),
                usage.get("input_tokens", 0),
                usage.get("cached_input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("reasoning_output_tokens", 0),
            )
        else:
            logger.info("antigravity call finished in %ds", int(duration_s))

    def _messages_prompt(self, messages: list[dict]) -> str:
        parts = [
            "You are running as Mira's model backend through Antigravity CLI.",
            "Follow the Mira review instructions exactly. Do not mention Antigravity CLI.",
            "Return only the requested final answer; no prose wrappers unless explicitly requested.",
            "",
            "## Mira messages",
        ]
        for i, message in enumerate(messages, 1):
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"\n### Message {i}: {role}\n{content}")
        return "\n".join(parts)

    def _tool_prompt(self, messages: list[dict], tools: list[dict]) -> str:
        tool = tools[0].get("function", {}) if tools else {}
        tool_name = tool.get("name") or "submit_result"
        schema = tool.get("parameters") or {"type": "object"}
        return (
            self._messages_prompt(messages)
            + "\n\n## Required output\n"
            + f"Return ONLY a JSON object containing the arguments for `{tool_name}`.\n"
            + "Do not wrap the JSON in markdown fences. Do not include explanatory text.\n"
            + "The JSON object must conform to this schema:\n"
            + json.dumps(schema, indent=2, sort_keys=True)
        )

    def _extract_json_object(self, text: str) -> str:
        candidate = text.strip()
        if not candidate:
            raise LLMError("antigravity_empty_response")

        def is_object(value: str) -> bool:
            try:
                return isinstance(json.loads(value), dict)
            except Exception:
                return False

        if "```" in candidate:
            blocks = candidate.split("```")
            for block in blocks[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if is_object(block):
                    return block

        try:
            parsed_candidate = json.loads(candidate)
        except Exception:
            parsed_candidate = None
        else:
            if isinstance(parsed_candidate, dict):
                return candidate
            raise LLMError("antigravity_non_object_json", type=type(parsed_candidate).__name__)

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = candidate[start : end + 1]
            try:
                parsed = json.loads(obj)
            except Exception as exc:
                raise LLMError(
                    "antigravity_malformed_json",
                    error=str(exc),
                    excerpt=candidate[:1000],
                ) from exc
            if isinstance(parsed, dict):
                return obj

        raise LLMError("antigravity_no_json_object", excerpt=candidate[:1000])

    def _account_usage(self, prompt: str, result: str, usage: dict[str, int]) -> None:
        """Track token spend. Prefers the exact usage reported by the agy
        event stream; falls back to the chars/4 heuristic when a call died
        before the ``result`` envelope."""
        if usage:
            self.total_prompt_tokens += usage.get("input_tokens", 0)
            self.total_completion_tokens += usage.get("output_tokens", 0)
        else:
            self.total_prompt_tokens += self.count_tokens(prompt)
            self.total_completion_tokens += self.count_tokens(result)

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self._messages_prompt(messages)
        if json_mode:
            prompt += (
                "\n\n## Required output\n"
                "Return ONLY one valid JSON object. No markdown fences or explanatory text."
            )
        raw, usage = await self._run_antigravity(prompt)
        result = self._extract_json_object(raw) if json_mode else raw
        self._account_usage(prompt, result, usage)
        return result

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        prompt = self._tool_prompt(messages, tools)
        raw, usage = await self._run_antigravity(prompt)
        result = self._extract_json_object(raw)
        self._account_usage(prompt, result, usage)
        return result

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        # The CLI does not expose Mira's incremental OpenAI-style tool calls.
        # Defer immediately so the caller performs exactly one forced review.
        return {"content": "", "tool_calls": []}

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

        return await self.complete_with_tools(
            messages, tools=[SUBMIT_REVIEW_TOOL], temperature=temperature
        )

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        from mira.llm.tool_schemas import SUBMIT_WALKTHROUGH_TOOL

        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])
