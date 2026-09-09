"""Codex CLI-backed provider using the local Codex OAuth session.

This provider intentionally keeps Mira's review contract unchanged: callers pass
Mira's existing chat messages/tool schemas and receive the same JSON strings the
OpenAI-compatible provider would have returned. The only difference is that the
model execution happens through ``codex exec`` and ``CODEX_HOME`` instead of an
HTTP API key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import tempfile
from contextlib import suppress
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mira.config import LLMConfig
from mira.exceptions import LLMError

logger = logging.getLogger(__name__)

# Mira reasoning efforts understood by Codex's model_reasoning_effort config;
# stronger levels clamp to "high" (the widest codex-wide value).
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def _parse_exec_event(raw: bytes | str) -> dict | None:
    """Parse one ``codex exec --json`` JSONL line into an event dict.

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


class CodexCLIProvider:
    """LLM provider that shells out to OpenAI Codex CLI.

    Auth is provided by Codex itself, normally via ``$CODEX_HOME/auth.json`` from
    ``codex login``. No OpenAI API key is read or sent by this provider.
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
        # published to the dashboard's live progress view. Callers using
        # other providers may set the same attribute; only this provider
        # consumes it.
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

    def _env(self, runtime_home: str, runtime_codex_home: str) -> dict[str, str]:
        """Build a minimal child environment without Mira service credentials."""
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS}
        env["HOME"] = runtime_home
        env["CODEX_HOME"] = runtime_codex_home
        return env

    def _prepare_codex_home(self, invocation_root: str) -> str:
        """Create writable ephemeral Codex state containing only OAuth auth."""
        destination = Path(invocation_root) / "codex-home"
        destination.mkdir(parents=True, mode=0o700)
        source_home = self.config.codex_home or os.environ.get("CODEX_HOME")
        if source_home:
            source_auth = Path(source_home).expanduser() / "auth.json"
            if not source_auth.is_file():
                raise LLMError("codex_auth_file_missing", path=str(source_auth))
            destination_auth = destination / "auth.json"
            destination_auth.write_bytes(source_auth.read_bytes())
            destination_auth.chmod(0o600)
        return str(destination)

    def _command(self, output_path: str) -> list[str]:
        codex_command = self.config.codex_command or "codex"
        if any(char in codex_command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid codex_command: {codex_command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        cmd = [
            codex_command,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            self.config.codex_sandbox,
            "--output-last-message",
            output_path,
            "--ephemeral",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "-c",
            'shell_environment_policy.inherit="none"',
        ]
        if self.config.model not in {"", "default", "codex-default"}:
            cmd.extend(["-m", self.config.model])
        effort = _EFFORT_MAP.get(self.config.reasoning_effort or "")
        if effort:
            # Codex reads its reasoning level from config, not a flag; a `-c`
            # override lands in the ephemeral CODEX_HOME the CLI consults.
            # Stronger levels clamp to "high" (the widest codex-wide value —
            # "xhigh" exists only on codex-max model variants).
            cmd.extend(["-c", f"model_reasoning_effort={effort}"])
        cmd.append("-")
        return cmd

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate Codex and model-spawned descendants before returning."""
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
    async def _run_codex(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Run one codex exec call; return (final text, token usage).

        stdout is a JSONL event stream (``--json``): thread/turn/item events
        ending in ``turn.completed`` with exact token usage. Events are parsed
        as they arrive — both for liveness logging and, when ``progress_key``
        is set, for the dashboard's live progress view. The final message file
        (``--output-last-message``) stays the primary result channel.
        """
        with tempfile.TemporaryDirectory(prefix="mira-codex-") as tmpdir:
            output_path = str(Path(tmpdir) / "last-message.txt")
            runtime_home = str(Path(tmpdir) / "runtime")
            Path(runtime_home).mkdir(mode=0o700)
            runtime_codex_home = self._prepare_codex_home(tmpdir)
            cmd = self._command(output_path)
            logger.debug("Running Codex CLI provider: %s", shlex.join(cmd[:-1] + ["<stdin>"]))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._env(runtime_home, runtime_codex_home),
                    cwd=runtime_home,
                    start_new_session=os.name == "posix",
                )
            except FileNotFoundError as exc:
                raise LLMError(
                    "codex_command_not_found", command=self.config.codex_command
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
                the pump to finish first. The explicit close matters — without
                it codex waits for stdin EOF forever and the call dies at the
                timeout.
                """
                try:
                    proc_stdin.write(prompt.encode("utf-8"))
                    await proc_stdin.drain()
                    proc_stdin.close()
                    await proc_stdin.wait_closed()
                except Exception:
                    pass

            async def _pump_stdout() -> tuple[str, dict[str, int]]:
                """Consume exec --json events until EOF.

                Returns (last agent_message text, usage from turn.completed).
                """
                agent_text = ""
                usage: dict[str, int] = {}
                while True:
                    raw = await proc_stdout.readline()
                    if not raw:
                        break
                    event = _parse_exec_event(raw)
                    if event is None:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "item.completed":
                        item = event.get("item") or {}
                        item_type = item.get("type", "")
                        if item_type == "agent_message" and item.get("text"):
                            agent_text = item["text"]
                        if item_type:
                            logger.debug("codex item completed: %s", item_type)
                            if progress_key:
                                tracker.call_item(progress_key, str(item_type))
                    elif event_type == "turn.completed":
                        raw_usage = event.get("usage") or {}
                        usage = {
                            key: int(raw_usage.get(key) or 0)
                            for key in (
                                "input_tokens",
                                "cached_input_tokens",
                                "output_tokens",
                                "reasoning_output_tokens",
                            )
                        }
                        if progress_key:
                            tracker.call_usage(progress_key, **usage)
                    elif event_type in ("turn.failed", "thread.failed", "error"):
                        logger.warning(
                            "codex stream error event (%s): %s",
                            event_type,
                            str(event.get("message") or event.get("error") or event)[:500],
                        )
                    elif event_type == "thread.started":
                        logger.debug("codex thread started: %s", event.get("thread_id", ""))
                return agent_text, usage

            async def _heartbeat() -> None:
                """Emit a liveness line while the call runs — reasoning-heavy
                calls can go minutes between stream events."""
                while True:
                    await asyncio.sleep(30)
                    logger.info(
                        "codex call in progress: %ds elapsed (timeout %ds)",
                        int(loop.time() - started),
                        self.config.codex_timeout_seconds,
                    )

            stderr_task = asyncio.ensure_future(_drain_stderr())
            stdin_task = asyncio.ensure_future(_write_stdin())
            heartbeat = asyncio.ensure_future(_heartbeat())
            try:
                agent_text, usage = await asyncio.wait_for(
                    _pump_stdout(),
                    timeout=self.config.codex_timeout_seconds,
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
                raise LLMError("codex_timeout", seconds=self.config.codex_timeout_seconds) from exc
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
            output_file = Path(output_path)
            last_message = output_file.read_text(encoding="utf-8") if output_file.exists() else ""

            if exit_code != 0:
                detail = (stderr_text or agent_text or last_message).strip()
                raise LLMError(
                    "codex_exit_failed",
                    exit_code=exit_code,
                    detail=detail[-2000:],
                )

            return ((last_message or agent_text) or "").strip(), usage

    @staticmethod
    def _log_call_finished(duration_s: float, usage: dict[str, int]) -> None:
        """One log line per finished call: duration plus real token spend."""
        if usage:
            logger.info(
                "codex call finished in %ds (tokens: %d in / %d cached / %d out / %d reasoning)",
                int(duration_s),
                usage.get("input_tokens", 0),
                usage.get("cached_input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("reasoning_output_tokens", 0),
            )
        else:
            logger.info("codex call finished in %ds", int(duration_s))

    def _messages_prompt(self, messages: list[dict]) -> str:
        parts = [
            "You are running as Mira's model backend through Codex CLI.",
            "Follow the Mira review instructions exactly. Do not mention Codex CLI.",
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
            raise LLMError("codex_empty_response")

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
            raise LLMError("codex_non_object_json", type=type(parsed_candidate).__name__)

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = candidate[start : end + 1]
            try:
                parsed = json.loads(obj)
            except Exception as exc:
                raise LLMError(
                    "codex_malformed_json",
                    error=str(exc),
                    excerpt=candidate[:1000],
                ) from exc
            if isinstance(parsed, dict):
                return obj

        raise LLMError("codex_no_json_object", excerpt=candidate[:1000])

    def _account_usage(self, prompt: str, result: str, usage: dict[str, int]) -> None:
        """Track token spend. Prefers the exact usage reported by the codex
        event stream; falls back to the chars/4 heuristic when a call died
        before ``turn.completed``."""
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
        raw, usage = await self._run_codex(prompt)
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
        raw, usage = await self._run_codex(prompt)
        result = self._extract_json_object(raw)
        self._account_usage(prompt, result, usage)
        return result

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        # Codex CLI does not expose Mira's incremental OpenAI-style tool calls.
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
