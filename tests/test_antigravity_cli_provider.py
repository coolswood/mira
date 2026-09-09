from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig
from mira.exceptions import LLMError
from mira.llm import create_llm
from mira.llm.antigravity_cli import AntigravityCLIProvider


class TestAntigravityCLIProvider:
    def test_factory_selects_antigravity_provider(self):
        provider = create_llm(LLMConfig(provider="antigravity-cli", model="gemini-3-pro"))
        assert isinstance(provider, AntigravityCLIProvider)

    @pytest.mark.parametrize("alias", ["antigravity", "antigravity-cli", "antigravity_cli", "agy"])
    def test_factory_accepts_provider_aliases(self, alias: str):
        provider = create_llm(LLMConfig(provider=alias))
        assert isinstance(provider, AntigravityCLIProvider)

    def test_command_uses_stream_json_with_stdin_prompt(self):
        provider = AntigravityCLIProvider(
            LLMConfig(
                provider="antigravity-cli",
                model="antigravity-default",
                antigravity_command="/usr/local/bin/agy",
            )
        )
        cmd = provider._command()
        assert cmd == [
            "/usr/local/bin/agy",
            "--output-format",
            "stream-json",
            "--print-timeout",
            "900s",
        ]

    def test_command_passes_model_and_effort(self):
        provider = AntigravityCLIProvider(
            LLMConfig(
                provider="antigravity-cli",
                model="gemini-3-pro",
                reasoning_effort="max",
                antigravity_sandbox=True,
            )
        )
        cmd = provider._command()
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gemini-3-pro"
        # Stronger efforts clamp to the CLI maximum.
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert "--sandbox" in cmd
        assert "--print-timeout" in cmd

    @pytest.mark.parametrize("sentinel", ["", "default", "antigravity-default", "agy-default"])
    def test_command_omits_model_and_effort_for_defaults(self, sentinel: str):
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", model=sentinel)
        )
        cmd = provider._command()
        assert "--model" not in cmd
        assert "--effort" not in cmd
        assert "--sandbox" not in cmd

    def test_command_omits_effort_for_off(self):
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", reasoning_effort="off")
        )
        assert "--effort" not in provider._command()

    def test_command_rejects_shell_metacharacters_or_arguments(self):
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", antigravity_command="agy --dangerous-flag")
        )

        with pytest.raises(ValueError, match="Invalid antigravity_command"):
            provider._command()

    def test_subprocess_environment_excludes_service_secrets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.setenv("GITHUB_TOKEN", "do-not-inherit")
        monkeypatch.setenv("DATABASE_URL", "postgres://secret")
        monkeypatch.setenv("MIRA_WEBHOOK_SECRET", "do-not-inherit")
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", antigravity_home="/trusted/gemini-home")
        )

        env = provider._env(str(tmp_path))

        assert env["PATH"] == "/usr/local/bin:/usr/bin"
        assert env["HOME"] == str(tmp_path)
        assert env["GEMINI_API_KEY"] == "env-key"
        assert "GITHUB_TOKEN" not in env
        assert "DATABASE_URL" not in env
        assert "MIRA_WEBHOOK_SECRET" not in env

    def test_subprocess_environment_omits_key_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        env = provider._env(str(tmp_path))

        assert "GEMINI_API_KEY" not in env

    def test_configured_api_key_wins_over_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", antigravity_api_key="config-key")
        )

        assert provider._env(str(tmp_path))["GEMINI_API_KEY"] == "config-key"

    def test_runtime_home_copies_trusted_gemini_state(self, tmp_path):
        source = tmp_path / "mounted-gemini"
        (source / "antigravity-cli").mkdir(parents=True)
        (source / "antigravity-cli" / "settings.json").write_text('{"modelProvider": "gemini"}')
        (source / "cached-credentials.json").write_text('{"token": "secret"}')
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", antigravity_home=str(source))
        )

        runtime_home = tmp_path / "runtime"
        runtime_home.mkdir()
        provider._prepare_gemini_home(str(runtime_home))

        gemini = runtime_home / ".gemini"
        assert (gemini / "antigravity-cli" / "settings.json").read_text() == (
            '{"modelProvider": "gemini"}'
        )
        assert (gemini / "cached-credentials.json").read_text() == '{"token": "secret"}'

    def test_missing_antigravity_home_fails_fast(self, tmp_path):
        provider = AntigravityCLIProvider(
            LLMConfig(
                provider="antigravity-cli", antigravity_home=str(tmp_path / "does-not-exist")
            )
        )

        runtime_home = tmp_path / "runtime"
        runtime_home.mkdir()
        with pytest.raises(LLMError, match="home directory not found"):
            provider._prepare_gemini_home(str(runtime_home))

    def test_api_key_auth_writes_gemini_provider_settings(self, tmp_path):
        provider = AntigravityCLIProvider(
            LLMConfig(provider="antigravity-cli", antigravity_api_key="key")
        )

        runtime_home = tmp_path / "runtime"
        runtime_home.mkdir()
        provider._prepare_gemini_home(str(runtime_home))

        settings = runtime_home / ".gemini" / "antigravity-cli" / "settings.json"
        assert json.loads(settings.read_text()) == {"modelProvider": "gemini"}

    def test_api_key_auth_does_not_override_copied_settings(self, tmp_path):
        source = tmp_path / "mounted-gemini"
        (source / "antigravity-cli").mkdir(parents=True)
        (source / "antigravity-cli" / "settings.json").write_text('{"modelProvider": "gemini"}')
        provider = AntigravityCLIProvider(
            LLMConfig(
                provider="antigravity-cli",
                antigravity_home=str(source),
                antigravity_api_key="key",
            )
        )

        runtime_home = tmp_path / "runtime"
        runtime_home.mkdir()
        provider._prepare_gemini_home(str(runtime_home))

        settings = runtime_home / ".gemini" / "antigravity-cli" / "settings.json"
        assert settings.read_text() == '{"modelProvider": "gemini"}'

    def test_no_auth_configured_leaves_home_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        runtime_home = tmp_path / "runtime"
        runtime_home.mkdir()
        provider._prepare_gemini_home(str(runtime_home))

        assert list((runtime_home / ".gemini").iterdir()) == []

    def test_extracts_fenced_json(self):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        assert provider._extract_json_object('Here you go:\n```json\n{"comments": []}\n```') == (
            '{"comments": []}'
        )

    def test_rejects_non_object_json_output(self):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        with pytest.raises(LLMError, match="JSON object"):
            provider._extract_json_object('[{"comments": []}]')

    @pytest.mark.asyncio
    async def test_complete_json_mode_returns_json_only(self):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        provider._run_antigravity = AsyncMock(  # type: ignore[method-assign]
            return_value=('text before {"ok": true} text after', {})
        )

        result = await provider.complete([{"role": "user", "content": "return json"}])

        assert result == '{"ok": true}'
        provider._run_antigravity.assert_awaited_once()
        assert provider.total_prompt_tokens > 0
        assert provider.total_completion_tokens > 0

    @pytest.mark.asyncio
    async def test_complete_with_tools_prompts_for_tool_arguments_json(self):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        provider._run_antigravity = AsyncMock(  # type: ignore[method-assign]
            return_value=('{"comments": [], "summary": "ok"}', {})
        )
        tool = {
            "type": "function",
            "function": {
                "name": "submit_review",
                "parameters": {
                    "type": "object",
                    "properties": {"comments": {"type": "array"}, "summary": {"type": "string"}},
                    "required": ["comments", "summary"],
                },
            },
        }

        result = await provider.complete_with_tools(
            [{"role": "user", "content": "review this"}], tools=[tool]
        )

        await_args = provider._run_antigravity.await_args
        assert await_args is not None
        prompt = await_args.args[0]
        assert "Return ONLY a JSON object containing the arguments for `submit_review`" in prompt
        assert '"required": [' in prompt
        assert result == '{"comments": [], "summary": "ok"}'

    @pytest.mark.asyncio
    async def test_subprocess_starts_in_an_isolated_process_group(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = self._fake_agy_proc(
            events=[self._result_event(response="", usage={"total_tokens": 0})]
        )
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        result, usage = await provider._run_antigravity("return JSON")

        assert result == ""
        # Usage is normalized to the full four-key shape the tracker uses.
        assert usage == {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        await_args = spawn.await_args
        assert await_args is not None
        assert await_args.kwargs["start_new_session"] is True

    # ── stream-json event stream ───────────────────────────────────

    @staticmethod
    def _result_event(
        response: str = "",
        usage: dict | None = None,
        status: str = "SUCCESS",
        error: str = "",
    ) -> dict:
        return {
            "event": "result",
            "result": {
                "conversation_id": "c-1",
                "status": status,
                "response": response,
                "error": error,
                "duration_seconds": 1.5,
                "num_turns": 1,
                "usage": usage or {},
            },
        }

    @staticmethod
    def _reader(lines: list[bytes]) -> asyncio.StreamReader:
        reader = asyncio.StreamReader()
        for line in lines:
            reader.feed_data(line)
        reader.feed_eof()
        return reader

    @staticmethod
    def _fake_agy_proc(
        events: list[dict],
        exit_code: int = 0,
    ) -> MagicMock:
        """A fake agy process whose stdout carries a stream-json event feed."""
        proc = MagicMock()
        proc.stdout = TestAntigravityCLIProvider._reader(
            [json.dumps(e).encode() + b"\n" for e in events]
        )
        stderr = asyncio.StreamReader()
        stderr.feed_eof()
        proc.stderr = stderr

        class _Stdin:
            def __init__(self) -> None:
                self.data = b""
                self.eof = False
                self.closed = False

            def write(self, data: bytes) -> None:
                self.data += data

            async def drain(self) -> None:
                pass

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                self.eof = True

        proc.stdin = _Stdin()
        proc.wait = AsyncMock(return_value=exit_code)
        proc.returncode = exit_code
        return proc

    @pytest.mark.asyncio
    async def test_run_parses_stream_and_normalizes_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        events = [
            {"event": "init", "session": "s-1"},
            {
                "event": "step_update",
                "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"},
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "text before ",
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": '{"ok": true}',
                },
            },
            self._result_event(
                response="envelope response",
                usage={
                    "input_tokens": 100,
                    "cache_read_tokens": 40,
                    "output_tokens": 7,
                    "thinking_tokens": 3,
                    "total_tokens": 150,
                },
            ),
        ]
        proc = self._fake_agy_proc(events)
        monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        text, usage = await provider._run_antigravity("prompt")

        # The result envelope is the authoritative answer channel.
        assert text == "envelope response"
        # agy usage keys are normalized to the tracker's codex-shaped keys.
        assert usage == {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 7,
            "reasoning_output_tokens": 3,
        }
        # stdin carried the prompt and was closed.
        assert proc.stdin.data == b"prompt"
        assert proc.stdin.closed

    @pytest.mark.asyncio
    async def test_deltas_fallback_when_result_envelope_has_no_response(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        events = [
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "text_delta": '{"ok":',
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "text_delta": " true}",
                },
            },
            self._result_event(response=""),
        ]
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=self._fake_agy_proc(events)),
        )
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        text, usage = await provider._run_antigravity("prompt")

        assert text == '{"ok": true}'
        assert usage == {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_exact_usage_feeds_token_accounting(self, monkeypatch: pytest.MonkeyPatch):
        events = [
            self._result_event(
                response='{"ok": true}',
                usage={"input_tokens": 500, "output_tokens": 25},
            ),
        ]
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=self._fake_agy_proc(events))
        )
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        await provider.complete([{"role": "user", "content": "hi"}])

        assert provider.total_prompt_tokens == 500
        assert provider.total_completion_tokens == 25

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_with_stderr_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = self._fake_agy_proc([], exit_code=1)
        stderr = asyncio.StreamReader()
        stderr.feed_data(b"Error: authentication required.\n")
        stderr.feed_eof()
        proc.stderr = stderr
        monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        with pytest.raises(LLMError, match="exit 1"):
            await provider._run_antigravity("prompt")

    @pytest.mark.asyncio
    async def test_stream_noise_and_non_json_lines_are_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = self._fake_agy_proc([])
        # Prepend garbage lines to the stream.
        noisy = self._reader([b"not json\n", b'{"truncated\n', b"  \n"])
        proc.stdout = noisy
        monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))

        text, usage = await provider._run_antigravity("prompt")

        assert text == ""
        assert usage == {}

    @pytest.mark.asyncio
    async def test_call_events_are_published_to_progress_tracker(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from mira.core import progress as progress_module

        events = [
            {
                "event": "step_update",
                "step_update": {"step_type": "tool", "state": "DONE"},
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "text_delta": '{"ok": true}',
                },
            },
            self._result_event(
                response='{"ok": true}',
                usage={"input_tokens": 10, "output_tokens": 2},
            ),
        ]
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=self._fake_agy_proc(events))
        )
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        provider.progress_key = "acme/widget#7"
        progress_module.tracker.begin(
            "acme/widget#7", progress_module.REVIEW, "acme/widget", pr_number=7
        )

        await provider._run_antigravity("prompt")

        job = progress_module.tracker.get("acme/widget#7")
        assert job is not None
        assert job.calls_started == 1
        assert job.calls_in_flight == 0
        assert job.tokens_input == 10
        assert job.tokens_output == 2
        assert job.items == {"tool": 1, "agent_response": 1}

    @pytest.mark.asyncio
    async def test_process_group_exit_race_does_not_rekill_direct_child(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 123
        proc.wait = AsyncMock()
        monkeypatch.setattr("os.killpg", MagicMock(side_effect=ProcessLookupError))

        await provider._terminate_process_tree(proc)

        proc.kill.assert_not_called()
        proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exited_group_leader_still_triggers_descendant_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        proc = MagicMock()
        proc.returncode = 0
        proc.pid = 456
        proc.wait = AsyncMock()
        killpg = MagicMock()
        monkeypatch.setattr("os.killpg", killpg)

        await provider._terminate_process_tree(proc)

        killpg.assert_called_once_with(456, signal.SIGKILL)
        proc.kill.assert_not_called()
        proc.wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complete_agentic_defers_without_starting_a_second_cli_review(self):
        provider = AntigravityCLIProvider(LLMConfig(provider="antigravity-cli"))
        provider._run_antigravity = AsyncMock(  # type: ignore[method-assign]
            return_value=('{"comments": []}', {})
        )

        msg = await provider.complete_agentic([{"role": "user", "content": "hi"}], tools=[])

        assert msg == {"content": "", "tool_calls": []}
        provider._run_antigravity.assert_not_awaited()
