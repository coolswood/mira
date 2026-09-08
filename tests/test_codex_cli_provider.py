from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig
from mira.exceptions import LLMError
from mira.llm import create_llm
from mira.llm.codex_cli import CodexCLIProvider


class TestCodexCLIProvider:
    def test_factory_selects_codex_cli_provider(self):
        provider = create_llm(LLMConfig(provider="codex-cli", model="gpt-5-codex"))
        assert isinstance(provider, CodexCLIProvider)

    def test_command_uses_stdin_oauth_cli_not_http_api(self):
        provider = CodexCLIProvider(
            LLMConfig(
                provider="codex-cli",
                model="gpt-5-codex",
                codex_command="/usr/local/bin/codex",
                codex_sandbox="read-only",
            )
        )
        cmd = provider._command("/tmp/out.txt")
        assert cmd == [
            "/usr/local/bin/codex",
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-last-message",
            "/tmp/out.txt",
            "--ephemeral",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "-c",
            'shell_environment_policy.inherit="none"',
            "-m",
            "gpt-5-codex",
            "-",
        ]

    def test_command_omits_model_for_codex_default(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli", model="codex-default"))
        cmd = provider._command("/tmp/out.txt")
        assert "-m" not in cmd
        assert cmd[-1] == "-"

    def test_command_rejects_shell_metacharacters_or_arguments(self):
        provider = CodexCLIProvider(
            LLMConfig(provider="codex-cli", codex_command="codex --dangerous-flag")
        )

        with pytest.raises(ValueError, match="Invalid codex_command"):
            provider._command("/tmp/out.txt")

    def test_subprocess_environment_excludes_service_secrets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.setenv("GITHUB_TOKEN", "do-not-inherit")
        monkeypatch.setenv("DATABASE_URL", "postgres://secret")
        monkeypatch.setenv("MIRA_WEBHOOK_SECRET", "do-not-inherit")
        monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
        provider = CodexCLIProvider(
            LLMConfig(provider="codex-cli", codex_home="/trusted/codex-home")
        )

        env = provider._env(str(tmp_path), "/private/runtime-codex-home")

        assert env["PATH"] == "/usr/local/bin:/usr/bin"
        assert env["HOME"] == str(tmp_path)
        assert env["CODEX_HOME"] == "/private/runtime-codex-home"
        assert "GITHUB_TOKEN" not in env
        assert "DATABASE_URL" not in env
        assert "MIRA_WEBHOOK_SECRET" not in env

    def test_runtime_codex_home_copies_only_auth_file(self, tmp_path):
        source = tmp_path / "mounted-codex-home"
        source.mkdir()
        (source / "auth.json").write_text('{"tokens": "secret"}')
        (source / "config.toml").write_text("dangerous_user_config = true")
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli", codex_home=str(source)))

        runtime_home = Path(provider._prepare_codex_home(str(tmp_path / "invocation")))

        assert (runtime_home / "auth.json").read_text() == '{"tokens": "secret"}'
        assert not (runtime_home / "config.toml").exists()

    def test_command_disables_user_rules_and_shell_environment_inheritance(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        cmd = provider._command("/tmp/out.txt")

        assert "--ignore-user-config" in cmd
        assert "--ignore-rules" in cmd
        assert 'shell_environment_policy.inherit="none"' in cmd

    def test_extracts_fenced_json(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        assert provider._extract_json_object('Here you go:\n```json\n{"comments": []}\n```') == (
            '{"comments": []}'
        )

    def test_rejects_non_object_json_output(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        with pytest.raises(LLMError, match="JSON object"):
            provider._extract_json_object('[{"comments": []}]')

    @pytest.mark.asyncio
    async def test_complete_json_mode_returns_json_only(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        provider._run_codex = AsyncMock(  # type: ignore[method-assign]
            return_value=('text before {"ok": true} text after', {})
        )

        result = await provider.complete([{"role": "user", "content": "return json"}])

        assert result == '{"ok": true}'
        provider._run_codex.assert_awaited_once()
        assert provider.total_prompt_tokens > 0
        assert provider.total_completion_tokens > 0

    @pytest.mark.asyncio
    async def test_complete_json_mode_counts_extracted_response_tokens(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        provider._run_codex = AsyncMock(  # type: ignore[method-assign]
            return_value=('text before {"ok": true} text after', {})
        )
        provider.count_tokens = lambda text: len(text)  # type: ignore[method-assign]

        result = await provider.complete([{"role": "user", "content": "return json"}])

        assert result == '{"ok": true}'
        assert provider.total_completion_tokens == len('{"ok": true}')

    @pytest.mark.asyncio
    async def test_complete_with_tools_prompts_for_tool_arguments_json(self):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        provider._run_codex = AsyncMock(  # type: ignore[method-assign]
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

        await_args = provider._run_codex.await_args
        assert await_args is not None
        prompt = await_args.args[0]
        assert "Return ONLY a JSON object containing the arguments for `submit_review`" in prompt
        assert '"required": [' in prompt
        assert result == '{"comments": [], "summary": "ok"}'

    @pytest.mark.asyncio
    async def test_subprocess_starts_in_an_isolated_process_group(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = self._fake_codex_proc(
            events=[{"type": "turn.completed", "usage": {"output_tokens": 1}}]
        )
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        result, usage = await provider._run_codex("return JSON")

        assert result == ""
        # Usage is normalized to the full four-key shape the tracker uses.
        assert usage == {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 1,
            "reasoning_output_tokens": 0,
        }
        await_args = spawn.await_args
        assert await_args is not None
        assert await_args.kwargs["start_new_session"] is True

    # ── exec --json event stream ───────────────────────────────────

    @staticmethod
    def _reader(lines: list[bytes]) -> asyncio.StreamReader:
        reader = asyncio.StreamReader()
        for line in lines:
            reader.feed_data(line)
        reader.feed_eof()
        return reader

    @staticmethod
    def _fake_codex_proc(
        events: list[dict],
        last_message: str = "",
        exit_code: int = 0,
    ) -> MagicMock:
        """A fake codex exec process whose stdout carries a --json stream."""
        import json as _json

        proc = MagicMock()
        proc.stdout = TestCodexCLIProvider._reader(
            [_json.dumps(e).encode() + b"\n" for e in events]
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
        proc.last_message = last_message
        return proc

    @pytest.mark.asyncio
    async def test_run_codex_parses_stream_and_reports_exact_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        events = [
            {"type": "thread.started", "thread_id": "t-1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "reasoning"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "not final"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"ok": true}'},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 7,
                    "reasoning_output_tokens": 3,
                },
            },
        ]
        proc = self._fake_codex_proc(events)
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        text, usage = await provider._run_codex("prompt")

        # No last-message file exists in the fake run — the stream fallback wins.
        assert text == '{"ok": true}'
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
    async def test_exact_usage_feeds_token_accounting(self, monkeypatch: pytest.MonkeyPatch):
        events = [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"ok": true}'},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 500, "output_tokens": 25},
            },
        ]
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=self._fake_codex_proc(events))
        )
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        await provider.complete([{"role": "user", "content": "hi"}])

        assert provider.total_prompt_tokens == 500
        assert provider.total_completion_tokens == 25

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_with_stderr_detail(self, monkeypatch: pytest.MonkeyPatch):
        proc = self._fake_codex_proc([{"type": "turn.started"}], exit_code=1)
        monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        with pytest.raises(LLMError, match="exit 1"):
            await provider._run_codex("prompt")

    @pytest.mark.asyncio
    async def test_call_events_are_published_to_progress_tracker(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from mira.core import progress as progress_module

        events = [
            {"type": "item.completed", "item": {"type": "reasoning"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"ok": true}'},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        ]
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=self._fake_codex_proc(events))
        )
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        provider.progress_key = "acme/widget#7"
        progress_module.tracker.begin(
            "acme/widget#7", progress_module.REVIEW, "acme/widget", pr_number=7
        )

        await provider._run_codex("prompt")

        job = progress_module.tracker.get("acme/widget#7")
        assert job is not None
        assert job.calls_started == 1
        assert job.calls_in_flight == 0
        assert job.tokens_input == 10
        assert job.tokens_output == 2
        assert job.items == {"reasoning": 1, "agent_message": 1}

    @pytest.mark.asyncio
    async def test_stream_noise_and_non_json_lines_are_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        proc = self._fake_codex_proc([])
        # Prepend garbage lines to the stream.
        noisy = self._reader([b"not json\n", b'{"truncated\n', b"  \n"])
        proc.stdout = noisy
        monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))

        text, usage = await provider._run_codex("prompt")

        assert text == ""
        assert usage == {}

    @pytest.mark.asyncio
    async def test_process_group_exit_race_does_not_rekill_direct_child(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
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
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
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
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli"))
        provider._run_codex = AsyncMock(  # type: ignore[method-assign]
            return_value=('{"comments": []}', {})
        )

        msg = await provider.complete_agentic([{"role": "user", "content": "hi"}], tools=[])

        assert msg == {"content": "", "tool_calls": []}
        provider._run_codex.assert_not_awaited()
