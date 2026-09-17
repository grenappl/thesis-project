"""Runs Pipeline 2 (api/pipelines/demo/) for one uploaded audio file.

Pipeline 2 only runs under WSL2 (Essentia has no Windows wheel), while this
API runs natively on Windows — so this service shells out to `wsl.exe` per
request rather than importing the pipeline directly. See
PIPELINE_SETUP.md#why-pipeline-2-needs-wsl2 for why the pipeline itself
isn't split, and why a per-request subprocess call (rather than a
persistent WSL2 service) was chosen for this demo/test endpoint: simplicity
over per-request latency, since this isn't a high-traffic path.
"""

from __future__ import annotations

import asyncio
import json
import queue as queue_module
import shlex
import subprocess
import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import UploadFile

from api.core.config import Settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPLOAD_DIR = _PROJECT_ROOT / "api" / "pipelines" / "demo" / "_uploads"

_KNOWN_ERROR_MARKERS = (
    "UnsupportedAudioError",
    "AudioTooShortError",
    "RuntimeError",
    "ValueError",
    "TypeError",
)

_STREAM_END = object()  # sentinel meaning "no more events, stop reading the queue"


class DemoExtractionError(Exception):
    """A feature-extraction failure with a message safe to show the caller."""


class DemoFeatureExtractionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def extract_features(self, upload: UploadFile, lyrics: str | None = None) -> dict:
        async with self._saved_upload(upload, lyrics) as (audio_path, lyrics_path):
            return await asyncio.to_thread(self._run_pipeline, audio_path, lyrics_path)

    async def stream_extract_features(
        self, upload: UploadFile, lyrics: str | None = None
    ) -> AsyncIterator[dict]:
        """Yields {"type": "log", ...} events live, then one final
        {"type": "result", ...} or {"type": "error", ...} event.

        Never raises — every failure becomes an {"type": "error"} event,
        since HTTP status/headers are already committed once streaming
        starts (see _stream_pipeline).
        """
        try:
            async with self._saved_upload(upload, lyrics) as (audio_path, lyrics_path):
                async for event in self._stream_pipeline(audio_path, lyrics_path):
                    yield event
        except Exception as exc:  # last-resort guard, see docstring
            yield {"type": "error", "message": f"Unexpected error: {exc}"}

    def _saved_upload(self, upload: UploadFile, lyrics: str | None):
        return _SavedUpload(upload, lyrics)

    def _build_command(self, audio_path: Path, lyrics_path: Path | None) -> list[str]:
        settings = self.settings
        relative_audio_path = audio_path.relative_to(_PROJECT_ROOT).as_posix()

        inner_command = (
            "export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/thesis-project && "
            f"cd {shlex.quote(settings.wsl_project_path)} && "
            "uv run python -m api.pipelines.demo.pipeline "
            f"{shlex.quote(relative_audio_path)} "
            f"--panns-checkpoint {shlex.quote(settings.demo_panns_checkpoint)} "
            f"--vggish-ridge-models-dir {shlex.quote(settings.demo_vggish_ridge_models_dir)}"
        )
        if lyrics_path is not None:
            relative_lyrics_path = lyrics_path.relative_to(_PROJECT_ROOT).as_posix()
            inner_command += f" --lyrics-file {shlex.quote(relative_lyrics_path)}"
        return ["wsl.exe", "-d", settings.wsl_distro, "--", "bash", "-lc", inner_command]

    def _run_pipeline(self, audio_path: Path, lyrics_path: Path | None) -> dict:
        cmd = self._build_command(audio_path, lyrics_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.demo_pipeline_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise DemoExtractionError(
                f"Feature extraction timed out after {self.settings.demo_pipeline_timeout_seconds}s"
            ) from exc
        except FileNotFoundError as exc:
            raise DemoExtractionError("wsl.exe not found — is WSL2 installed on this machine?") from exc

        if result.returncode != 0:
            raise DemoExtractionError(_extract_error_message(result.stderr, result.returncode))

        return _extract_json(result.stdout)

    async def _stream_pipeline(self, audio_path: Path, lyrics_path: Path | None) -> AsyncIterator[dict]:
        # Once the StreamingResponse starts, HTTP headers/status are already
        # committed — every failure mode from here on has to surface as an
        # {"type": "error"} event, never as a raised exception, or it'd just
        # look like a truncated/broken connection to the client.
        try:
            cmd = self._build_command(audio_path, lyrics_path)
        except DemoExtractionError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        # Deliberately NOT asyncio.create_subprocess_exec: under uvicorn
        # --reload on Windows, the reload worker doesn't reliably get the
        # ProactorEventLoop that Windows needs for asyncio subprocess pipes —
        # it silently raises a bare NotImplementedError. A background thread
        # running a plain blocking subprocess.Popen sidesteps that entirely,
        # regardless of which event loop policy is active.
        q: queue_module.Queue = queue_module.Queue()
        thread = threading.Thread(target=self._run_pipeline_in_thread, args=(cmd, q), daemon=True)
        thread.start()

        while True:
            item = await asyncio.to_thread(q.get)
            if item is _STREAM_END:
                return
            yield item

    def _run_pipeline_in_thread(self, cmd: list[str], q: queue_module.Queue) -> None:
        timeout = self.settings.demo_pipeline_timeout_seconds

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
        except FileNotFoundError:
            q.put({"type": "error", "message": "wsl.exe not found — is WSL2 installed on this machine?"})
            q.put(_STREAM_END)
            return

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def pump(stream, name: str, sink: list[str]) -> None:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\n")
                sink.append(line)
                q.put({"type": "log", "stream": name, "line": line})
            stream.close()

        readers = [
            threading.Thread(target=pump, args=(process.stdout, "stdout", stdout_lines)),
            threading.Thread(target=pump, args=(process.stderr, "stderr", stderr_lines)),
        ]
        for reader in readers:
            reader.start()

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            for reader in readers:
                reader.join(timeout=5)
            q.put({"type": "error", "message": f"Feature extraction timed out after {timeout}s"})
            q.put(_STREAM_END)
            return

        for reader in readers:
            reader.join(timeout=5)

        if returncode != 0:
            q.put({"type": "error", "message": _extract_error_message_from_lines(stderr_lines, returncode)})
            q.put(_STREAM_END)
            return

        try:
            features = _extract_json("\n".join(stdout_lines))
        except DemoExtractionError as exc:
            q.put({"type": "error", "message": str(exc)})
            q.put(_STREAM_END)
            return

        q.put({"type": "result", "data": features})
        q.put(_STREAM_END)


class _SavedUpload:
    """Async context manager: writes an UploadFile (and optional lyrics text)
    to temp paths under a shared UUID prefix, deletes both on exit.
    """

    def __init__(self, upload: UploadFile, lyrics: str | None = None) -> None:
        self._upload = upload
        self._lyrics = lyrics
        self._audio_path: Path | None = None
        self._lyrics_path: Path | None = None

    async def __aenter__(self) -> tuple[Path, Path | None]:
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        suffix = Path(self._upload.filename or "").suffix or ".audio"
        self._audio_path = _UPLOAD_DIR / f"{stem}{suffix}"
        contents = await self._upload.read()
        self._audio_path.write_bytes(contents)

        if self._lyrics is not None and self._lyrics.strip():
            self._lyrics_path = _UPLOAD_DIR / f"{stem}.lyrics.txt"
            self._lyrics_path.write_text(self._lyrics, encoding="utf-8")

        return self._audio_path, self._lyrics_path

    async def __aexit__(self, *exc_info: object) -> None:
        if self._audio_path is not None:
            self._audio_path.unlink(missing_ok=True)
        if self._lyrics_path is not None:
            self._lyrics_path.unlink(missing_ok=True)


def _extract_json(stdout: str) -> dict:
    # panns_inference prints "Checkpoint path: ..." / "Using CPU." straight to
    # stdout before the pipeline's own JSON output, so this isn't a plain
    # json.loads(stdout) — find the pretty-printed JSON object's opening line.
    lines = stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "{"), None)
    if start is None:
        raise DemoExtractionError(f"Pipeline produced no JSON output. Raw output:\n{stdout.strip()}")
    try:
        return json.loads("\n".join(lines[start:]))
    except json.JSONDecodeError as exc:
        raise DemoExtractionError(f"Could not parse pipeline output as JSON:\n{stdout.strip()}") from exc


def _extract_error_message(stderr: str, returncode: int) -> str:
    lines = [line for line in stderr.splitlines() if line.strip()]
    return _extract_error_message_from_lines(lines, returncode)


def _extract_error_message_from_lines(lines: list[str], returncode: int) -> str:
    non_empty = [line for line in lines if line.strip()]
    for line in reversed(non_empty):
        if any(marker in line for marker in _KNOWN_ERROR_MARKERS):
            return line.strip()
    tail = "\n".join(non_empty[-5:]) if non_empty else "(no output)"
    return f"Feature extraction failed (exit code {returncode}):\n{tail}"
