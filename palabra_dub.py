import argparse
import asyncio
import base64
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import websockets


INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
INPUT_SAMPLE_WIDTH_BYTES = 2
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH_BYTES = 2
AUDIO_OUTPUT_EXTENSIONS = {".mp3", ".wav"}
VIDEO_OUTPUT_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
ANIMATION_WIDTH = 80
_ANIMATION_ACTIVE = False


@dataclasses.dataclass
class SegmentState:
    transcription_id: str
    source_text: str = ""
    translated_text: str = ""
    source_end_sec: float | None = None
    last_chunk_received: bool = False


@dataclasses.dataclass
class ReceiveState:
    stream_start_monotonic: float
    segments: dict[str, SegmentState] = dataclasses.field(default_factory=dict)
    continuous_output_pcm: bytearray = dataclasses.field(default_factory=bytearray)
    chunk_events: list[tuple[str, bytes, bool]] = dataclasses.field(default_factory=list)

    def get_segment(self, transcription_id: str) -> SegmentState:
        segment = self.segments.get(transcription_id)
        if segment is None:
            segment = SegmentState(transcription_id=transcription_id)
            self.segments[transcription_id] = segment
        return segment


@dataclasses.dataclass
class SubtitleCue:
    index: int
    start_sec: float
    end_sec: float
    text: str


def output_frame_size_bytes() -> int:
    return OUTPUT_CHANNELS * OUTPUT_SAMPLE_WIDTH_BYTES


def output_bytes_per_second() -> int:
    return OUTPUT_SAMPLE_RATE * output_frame_size_bytes()


def seconds_to_output_frame_bytes(duration_sec: float) -> int:
    frame_size = output_frame_size_bytes()
    frame_count = int(round(duration_sec * OUTPUT_SAMPLE_RATE))
    return frame_count * frame_size


def pcm_duration_seconds(pcm_data: bytes | bytearray) -> float:
    return len(pcm_data) / float(output_bytes_per_second())


def normalize_alignment_mode(config: dict) -> str:
    mode = str(config.get("alignment_mode", "")).strip().lower()
    if mode not in {"raw", "inline", "ffmpeg_segments"}:
        raise RuntimeError("alignment_mode must be 'raw', 'inline', or 'ffmpeg_segments'")
    return mode


def normalize_mp4_audio_bitrate(config: dict) -> str | None:
    bitrate = str(config.get("mp4_audio_bitrate", "auto")).strip().lower()
    if bitrate == "auto":
        return None
    if bitrate not in {"64k", "80k", "96k", "128k", "160k"}:
        raise RuntimeError("mp4_audio_bitrate must be 'auto', '64k', '80k', '96k', '128k', or '160k'")
    return bitrate


def parse_sbv_timestamp(value: str) -> float:
    pieces = value.strip().split(":")
    if len(pieces) != 3:
        raise ValueError(f"Invalid SBV timestamp: {value}")
    hours = int(pieces[0])
    minutes = int(pieces[1])
    seconds = float(pieces[2].replace(",", "."))
    return hours * 3600 + minutes * 60 + seconds


def parse_sbv_file(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    blocks = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if "," not in lines[0]:
            raise ValueError(f"Invalid SBV cue timing line: {lines[0]}")
        start_raw, end_raw = lines[0].split(",", 1)
        start_sec = parse_sbv_timestamp(start_raw)
        end_sec = parse_sbv_timestamp(end_raw)
        text = " ".join(lines[1:]).strip()
        if not text:
            continue
        if end_sec < start_sec:
            raise ValueError(f"SBV cue ends before it starts: {lines[0]}")
        cues.append(SubtitleCue(index=len(cues) + 1, start_sec=start_sec, end_sec=end_sec, text=text))

    if not cues:
        raise ValueError(f"No usable subtitle cues found in: {path}")
    cues.sort(key=lambda cue: cue.start_sec)
    return cues


def subtitle_tts_language(config: dict) -> str:
    return str(config.get("subtitle_tts_language") or config.get("target_language") or config.get("source_language") or "").strip()


def subtitle_tts_voice_options(config: dict) -> dict:
    voice_id = config.get("voice_id")
    if not voice_id:
        raise RuntimeError("voice_id is required for subtitle TTS mode")
    return {
        "voice_id": voice_id,
        "speed": float(config.get("subtitle_tts_speed", 0.5)),
        "deaccent_strength": float(config.get("subtitle_tts_deaccent_strength", 1.0)),
    }


def trim_pcm_to_frame_boundary(pcm_data: bytes) -> bytes:
    frame_size = output_frame_size_bytes()
    extra_bytes = len(pcm_data) % frame_size
    if extra_bytes:
        return pcm_data[:-extra_bytes]
    return pcm_data


def pcm_sample_values(pcm_data: bytes) -> list[int]:
    frame_size = output_frame_size_bytes()
    pcm_data = trim_pcm_to_frame_boundary(pcm_data)
    return [int.from_bytes(pcm_data[i : i + frame_size], "little", signed=True) for i in range(0, len(pcm_data), frame_size)]


def trim_pcm_silence(pcm_data: bytes, threshold: int = 160, padding_sec: float = 0.03) -> tuple[bytes, float, float]:
    pcm_data = trim_pcm_to_frame_boundary(pcm_data)
    samples = pcm_sample_values(pcm_data)
    if not samples:
        return b"", 0.0, 0.0

    first = 0
    while first < len(samples) and abs(samples[first]) <= threshold:
        first += 1
    if first == len(samples):
        return b"", len(samples) / float(OUTPUT_SAMPLE_RATE), 0.0

    last = len(samples) - 1
    while last > first and abs(samples[last]) <= threshold:
        last -= 1

    padding_frames = int(round(padding_sec * OUTPUT_SAMPLE_RATE))
    start_frame = max(0, first - padding_frames)
    end_frame = min(len(samples), last + 1 + padding_frames)
    trimmed = pcm_data[start_frame * output_frame_size_bytes() : end_frame * output_frame_size_bytes()]
    leading_trim_sec = start_frame / float(OUTPUT_SAMPLE_RATE)
    trailing_trim_sec = (len(samples) - end_frame) / float(OUTPUT_SAMPLE_RATE)
    return trimmed, leading_trim_sec, trailing_trim_sec


def overlay_pcm_at(output_pcm: bytearray, start_sec: float, segment_pcm: bytes) -> None:
    start_byte = seconds_to_output_frame_bytes(start_sec)
    end_byte = start_byte + len(segment_pcm)
    if len(output_pcm) < end_byte:
        output_pcm.extend(b"\x00" * (end_byte - len(output_pcm)))
    output_pcm[start_byte:end_byte] = segment_pcm


def trim_pcm_duration(pcm_data: bytes, duration_sec: float) -> bytes:
    if duration_sec <= 0:
        return b""
    target_bytes = seconds_to_output_frame_bytes(duration_sec)
    return pcm_data[:target_bytes]


def trim_subtitle_cues_to_duration(cues: list[SubtitleCue], duration_sec: float) -> tuple[list[SubtitleCue], bool, int]:
    usable: list[SubtitleCue] = []
    skipped = 0
    clipped = False

    for cue in cues:
        if cue.start_sec >= duration_sec:
            skipped += 1
            clipped = True
            continue
        end_sec = min(cue.end_sec, duration_sec)
        if end_sec != cue.end_sec:
            clipped = True
        usable.append(SubtitleCue(index=cue.index, start_sec=cue.start_sec, end_sec=end_sec, text=cue.text))

    return usable, clipped, skipped


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def apply_env_overrides(config: dict) -> dict:
    merged = dict(config)
    env_mappings = {
        "client_id": "PALABRA_CLIENT_ID",
        "client_secret": "PALABRA_CLIENT_SECRET",
    }
    for config_key, env_key in env_mappings.items():
        env_value = os.environ.get(env_key)
        if env_value:
            merged[config_key] = env_value
    return merged


def ffmpeg_command(ffmpeg_path: str, *args: str) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        *args,
    ]


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        return

    if completed.stdout:
        print(completed.stdout, file=sys.stderr, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    raise subprocess.CalledProcessError(completed.returncode, command)


def extract_audio_to_wav(input_video: Path, output_wav: Path, ffmpeg_path: str) -> None:
    command = ffmpeg_command(
        ffmpeg_path,
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(INPUT_SAMPLE_RATE),
        "-ac",
        str(INPUT_CHANNELS),
        str(output_wav),
    )
    run_command(command)


def mux_audio_back_to_video(
    input_video: Path,
    input_audio: Path,
    output_video: Path,
    ffmpeg_path: str,
    audio_bitrate: str | None,
) -> None:
    audio_options = ["-c:a", "aac"]
    if audio_bitrate:
        audio_options.extend(["-b:a", audio_bitrate])

    command = ffmpeg_command(
        ffmpeg_path,
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        *audio_options,
        "-shortest",
        str(output_video),
    )
    run_command(command)


def convert_wav_to_mp3(input_wav: Path, output_mp3: Path, ffmpeg_path: str) -> None:
    command = ffmpeg_command(
        ffmpeg_path,
        "-y",
        "-i",
        str(input_wav),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_mp3),
    )
    run_command(command)


def run_ffmpeg(command: list[str]) -> None:
    run_command(command)


_LIVE_STATUS_LINES: dict[str, str] = {}
_LIVE_STATUS_ORDER = ("received", "sent", "finished")
_LIVE_STATUS_RENDERED = 0


def _clear_animation() -> None:
    global _ANIMATION_ACTIVE
    if _ANIMATION_ACTIVE:
        sys.stdout.write("\r" + (" " * ANIMATION_WIDTH) + "\r")
        _ANIMATION_ACTIVE = False


def _live_status_values() -> list[str]:
    return [line for key in _LIVE_STATUS_ORDER if (line := _LIVE_STATUS_LINES.get(key))]


def _render_live_status() -> None:
    global _LIVE_STATUS_RENDERED
    lines = _live_status_values()
    line_count = max(_LIVE_STATUS_RENDERED, len(lines))

    if _LIVE_STATUS_RENDERED:
        sys.stdout.write(f"\x1b[{_LIVE_STATUS_RENDERED}A")

    for index in range(line_count):
        line = lines[index] if index < len(lines) else ""
        sys.stdout.write("\r\x1b[2K" + line)
        if index < line_count - 1:
            sys.stdout.write("\n")

    if lines:
        sys.stdout.write("\n")

    sys.stdout.flush()
    _LIVE_STATUS_RENDERED = len(lines)


def update_live_status(key: str, message: str) -> None:
    _clear_animation()
    _LIVE_STATUS_LINES[key] = message
    _render_live_status()


def finish_live_status() -> None:
    global _LIVE_STATUS_RENDERED
    if _LIVE_STATUS_RENDERED:
        _LIVE_STATUS_RENDERED = 0


def log(message: str) -> None:
    _clear_animation()
    finish_live_status()
    print(message, flush=True)


async def animate_progress(label: str, stop_event: asyncio.Event) -> None:
    global _ANIMATION_ACTIVE
    frames = [".  ", ".. ", "..."]
    index = 0
    while not stop_event.is_set():
        _ANIMATION_ACTIVE = True
        text = f"{label}{frames[index % len(frames)]}"
        sys.stdout.write("\r" + text.ljust(ANIMATION_WIDTH))
        sys.stdout.flush()
        index += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

    if _ANIMATION_ACTIVE:
        sys.stdout.write("\r" + (" " * ANIMATION_WIDTH) + "\r")
        sys.stdout.flush()
        _ANIMATION_ACTIVE = False


def create_session(client_id: str, client_secret: str) -> dict:
    url = "https://api.palabra.ai/session-storage/session"
    payload = json.dumps({"data": {"subscriber_count": 0}}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "ClientID": client_id,
            "ClientSecret": client_secret,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to create session: HTTP {exc.code} {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to create session: {exc}") from exc

    data = json.loads(body)
    if not data.get("ok") or "data" not in data:
        raise RuntimeError(f"Unexpected session response: {data}")
    return data["data"]


def build_tts_init(config: dict) -> dict:
    language = subtitle_tts_language(config)
    if not language:
        raise RuntimeError("target_language or subtitle_tts_language is required for subtitle TTS mode")

    return {
        "type": "init",
        "language": language,
        "model": str(config.get("subtitle_tts_model", "auto")),
        "voice_options": subtitle_tts_voice_options(config),
        "output": {
            "format": "pcm",
            "sample_rate": OUTPUT_SAMPLE_RATE,
        },
    }


def split_tts_text(text: str, max_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if max_chars <= 0 or len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def build_tts_text_messages(text_chunks: list[str], generation_id: str) -> list[dict]:
    messages: list[dict] = []
    for index, chunk in enumerate(text_chunks):
        messages.append(
            {
                "type": "text",
                "text": chunk,
                "generation_id": generation_id,
                "is_eos": index == len(text_chunks) - 1,
            }
        )
    return messages


def build_set_task(config: dict) -> dict:
    auto_tempo = bool(config.get("auto_tempo", False))
    min_tempo = float(config.get("min_tempo", 1.0))
    max_tempo = float(config.get("max_tempo", 1.0))
    speech_mode = str(config.get("speech_mode", "voice_id")).strip().lower()

    if speech_mode not in {"voice_id", "voice_cloning"}:
        raise RuntimeError("speech_mode must be 'voice_id' or 'voice_cloning'")

    speech_generation = {
        "voice_cloning": speech_mode == "voice_cloning",
        "voice_timbre_detection": {
            "enabled": False,
            "high_timbre_voices": ["default_high"],
            "low_timbre_voices": ["default_low"],
        },
    }

    if speech_mode == "voice_id":
        voice_id = config.get("voice_id")
        if not voice_id:
            raise RuntimeError("voice_id is required when speech_mode = 'voice_id'")
        speech_generation["voice_id"] = voice_id
    else:
        speech_generation["voice_id"] = None

    return {
        "message_type": "set_task",
        "data": {
            "input_stream": {
                "content_type": "audio",
                "source": {
                    "type": "ws",
                    "format": "pcm_s16le",
                    "sample_rate": INPUT_SAMPLE_RATE,
                    "channels": INPUT_CHANNELS,
                },
            },
            "output_stream": {
                "content_type": "audio",
                "target": {
                    "type": "ws",
                    "format": "pcm_s16le",
                },
            },
            "pipeline": {
                "transcription": {
                    "source_language": config["source_language"],
                    "detectable_languages": [],
                    "segment_confirmation_silence_threshold": float(
                        config.get("segment_confirmation_silence_threshold", 0.7)
                    ),
                    "sentence_splitter": {
                        "enabled": bool(config.get("sentence_splitter_enabled", True)),
                    },
                    "verification": {
                        "auto_transcription_correction": False,
                        "transcription_correction_style": None,
                    },
                },
                "translations": [
                    {
                        "target_language": config["target_language"],
                        "translate_partial_transcriptions": False,
                        "speech_generation": speech_generation,
                    }
                ],
                "translation_queue_configs": {
                    "global": {
                        "desired_queue_level_ms": int(config.get("desired_queue_level_ms", 5000)),
                        "max_queue_level_ms": int(config.get("max_queue_level_ms", 20000)),
                        "auto_tempo": auto_tempo,
                        "min_tempo": min_tempo,
                        "max_tempo": max_tempo,
                    }
                },
                "allowed_message_types": [
                    "translated_transcription",
                    "partial_transcription",
                    "validated_transcription",
                    "partial_translated_transcription",
                ],
            },
        },
    }


def pcm_chunk_size_bytes(chunk_ms: int) -> int:
    if chunk_ms <= 0:
        raise RuntimeError("audio_chunk_ms must be greater than 0")
    frames = int(INPUT_SAMPLE_RATE * (chunk_ms / 1000.0))
    if frames <= 0:
        raise RuntimeError("audio_chunk_ms is too small for the configured input sample rate")
    return frames * INPUT_CHANNELS * INPUT_SAMPLE_WIDTH_BYTES


async def send_audio(ws, input_wav: Path, chunk_ms: int) -> None:
    chunk_size = pcm_chunk_size_bytes(chunk_ms)
    with wave.open(str(input_wav), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        total_frames = wav_file.getnframes()
        total_seconds = total_frames / float(sample_rate)
        frames_per_chunk = chunk_size // INPUT_SAMPLE_WIDTH_BYTES
        sent_frames = 0
        last_progress_print = 0.0

        if sample_rate != INPUT_SAMPLE_RATE or channels != INPUT_CHANNELS or sample_width != INPUT_SAMPLE_WIDTH_BYTES:
            raise RuntimeError(
                f"Unexpected input WAV format: rate={sample_rate}, channels={channels}, sample_width={sample_width}"
            )

        update_live_status("sent", f"Streaming audio to Palabra in real time ({total_seconds / 60:.1f} minutes total)...")

        while True:
            chunk = wav_file.readframes(frames_per_chunk)
            if not chunk:
                break
            await ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_data",
                        "data": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                        },
                    }
                )
            )
            sent_frames += len(chunk) // INPUT_SAMPLE_WIDTH_BYTES
            sent_seconds = sent_frames / float(sample_rate)
            if sent_seconds - last_progress_print >= 30.0:
                percent = min(100.0, (sent_seconds / total_seconds) * 100.0) if total_seconds else 100.0
                update_live_status(
                    "sent",
                    f"Sent {sent_seconds / 60:.1f}/{total_seconds / 60:.1f} minutes "
                    f"({percent:.1f}%). Waiting for translated audio as it streams back...",
                )
                last_progress_print = sent_seconds
            await asyncio.sleep(chunk_ms / 1000.0)

        update_live_status("finished", "Finished sending source audio.")


async def receive_audio(
    ws,
    receive_state: ReceiveState,
    inactivity_timeout_sec: float,
    sender_done: asyncio.Event,
) -> None:
    received_chunks = 0
    received_bytes = 0
    last_report = time.monotonic()
    last_audio_at = time.monotonic()
    while True:
        if sender_done.is_set() and time.monotonic() - last_audio_at >= inactivity_timeout_sec:
            log("No more translated audio received within timeout window. Finalizing WAV file...")
            return

        timeout = inactivity_timeout_sec
        if sender_done.is_set():
            remaining_audio_wait = inactivity_timeout_sec - (time.monotonic() - last_audio_at)
            timeout = max(0.1, min(1.0, remaining_audio_wait))

        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosedOK:
            log("Palabra WebSocket closed cleanly. Finalizing output files...")
            return
        except websockets.exceptions.ConnectionClosed as exc:
            if sender_done.is_set() and exc.code == 1000:
                log("Palabra WebSocket closed cleanly. Finalizing output files...")
                return
            raise

        payload = json.loads(message)
        message_type = payload.get("message_type")
        data = payload.get("data", {})

        if message_type == "output_audio_data":
            transcription = data.get("transcription", data)
            transcription_id = transcription.get("transcription_id")
            encoded = transcription.get("data")
            if transcription_id and encoded:
                segment = receive_state.get_segment(transcription_id)
                decoded = base64.b64decode(encoded)
                receive_state.continuous_output_pcm.extend(decoded)
                last_audio_at = time.monotonic()
                is_last_chunk = transcription.get("last_chunk") is True
                if is_last_chunk:
                    segment.last_chunk_received = True
                receive_state.chunk_events.append((transcription_id, decoded, is_last_chunk))
                received_chunks += 1
                received_bytes += len(decoded)
                now = time.monotonic()
                if now - last_report >= 30.0:
                    approx_seconds = received_bytes / float(output_bytes_per_second())
                    update_live_status(
                        "received",
                        f"Received {received_chunks} translated chunks "
                        f"(about {approx_seconds / 60:.1f} minutes of output audio so far)...",
                    )
                    last_report = now
        elif message_type == "validated_transcription":
            transcription = data.get("transcription", {})
            transcription_id = transcription.get("transcription_id")
            if transcription_id:
                segment = receive_state.get_segment(transcription_id)
                segment.source_text = transcription.get("text", "")
                segment.source_end_sec = time.monotonic() - receive_state.stream_start_monotonic
        elif message_type == "translated_transcription":
            transcription = data.get("transcription", {})
            transcription_id = transcription.get("transcription_id")
            if transcription_id:
                segment = receive_state.get_segment(transcription_id)
                segment.translated_text = transcription.get("text", "")
        elif message_type == "error":
            raise RuntimeError(f"Palabra API error: {json.dumps(data)}")


async def receive_tts_generation(ws, generation_id: str, inactivity_timeout_sec: float) -> bytes:
    chunks: list[bytes] = []
    last_audio_at = time.monotonic()

    while True:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=inactivity_timeout_sec)
        except asyncio.TimeoutError as exc:
            if chunks:
                raise RuntimeError(f"Timed out waiting for final TTS chunk for {generation_id}") from exc
            raise RuntimeError(f"Timed out waiting for TTS audio for {generation_id}") from exc
        except websockets.exceptions.ConnectionClosedOK as exc:
            raise RuntimeError(f"Palabra WebSocket closed before TTS generation finished: {generation_id}") from exc

        payload = json.loads(message)
        message_type = payload.get("type") or payload.get("message_type")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

        if message_type == "error":
            raise RuntimeError(f"Palabra API error: {json.dumps(data)}")
        if message_type != "audio_chunk":
            continue

        payload_generation_id = data.get("generation_id")
        if payload_generation_id and payload_generation_id != generation_id:
            continue

        encoded = data.get("audio") or data.get("data")
        if encoded:
            chunks.append(trim_pcm_to_frame_boundary(base64.b64decode(encoded)))
            last_audio_at = time.monotonic()

        if data.get("last_chunk") is True:
            return b"".join(chunks)

        if chunks and time.monotonic() - last_audio_at >= inactivity_timeout_sec:
            return b"".join(chunks)


def write_output_wav(output_path: Path, pcm_data: bytes | bytearray) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(OUTPUT_CHANNELS)
        wav_file.setsampwidth(OUTPUT_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(OUTPUT_SAMPLE_RATE)
        wav_file.writeframes(pcm_data)


def get_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def build_segment_manifest(config: dict, receive_state: ReceiveState) -> tuple[list[dict], float]:
    manifest: list[dict] = []
    cumulative_output_sec = 0.0
    segment_audio_totals: dict[str, int] = {}

    for transcription_id, chunk_bytes, is_last_chunk in receive_state.chunk_events:
        cumulative_output_sec += pcm_duration_seconds(chunk_bytes)
        segment_audio_totals[transcription_id] = segment_audio_totals.get(transcription_id, 0) + len(chunk_bytes)

        if not is_last_chunk:
            continue

        segment = receive_state.segments.get(transcription_id)
        if segment is None or segment.source_end_sec is None:
            continue

        output_audio_bytes = segment_audio_totals[transcription_id]
        output_audio_duration_sec = output_audio_bytes / float(output_bytes_per_second())
        manifest.append(
            {
                "transcription_id": transcription_id,
                "source_end_sec_raw": segment.source_end_sec,
                "raw_timeline_end_sec": cumulative_output_sec,
                "output_audio_duration_sec": output_audio_duration_sec,
                "output_audio_bytes": output_audio_bytes,
                "source_text": segment.source_text,
                "translated_text": segment.translated_text,
                "last_chunk_received": segment.last_chunk_received,
            }
        )

    if not manifest:
        return [], 0.0

    if "timing_latency_offset_sec" in config:
        latency_offset_sec = max(0.0, float(config.get("timing_latency_offset_sec", 0.0)))
    elif bool(config.get("normalize_source_timestamps", True)):
        latency_offset_sec = max(0.0, manifest[0]["source_end_sec_raw"] - manifest[0]["output_audio_duration_sec"])
    else:
        latency_offset_sec = 0.0

    previous_source_end_sec = 0.0
    for item in manifest:
        source_end_sec = max(0.0, item["source_end_sec_raw"] - latency_offset_sec)
        source_end_sec = max(source_end_sec, previous_source_end_sec)
        source_duration_sec = source_end_sec - previous_source_end_sec
        item["source_end_sec"] = source_end_sec
        item["source_duration_sec"] = source_duration_sec
        previous_source_end_sec = source_end_sec

    return manifest, latency_offset_sec


def build_inline_aligned_output(config: dict, receive_state: ReceiveState, manifest: list[dict]) -> tuple[bytearray, list[dict]]:
    aligned_output = bytearray()
    if not manifest:
        return bytearray(receive_state.continuous_output_pcm), []

    manifest_by_id = {item["transcription_id"]: item for item in manifest}
    timeline_cursor_sec = 0.0
    total_padding_sec = 0.0
    drift_warnings = 0
    used_manifest: list[dict] = []

    for transcription_id, chunk_bytes, is_last_chunk in receive_state.chunk_events:
        aligned_output.extend(chunk_bytes)
        timeline_cursor_sec += pcm_duration_seconds(chunk_bytes)

        if not is_last_chunk:
            continue

        item = manifest_by_id.get(transcription_id)
        if item is None:
            continue

        padding_after_sec = 0.0
        overrun_sec = 0.0
        if bool(config.get("pad_segments_to_source_timing", True)) and timeline_cursor_sec < item["source_end_sec"]:
            padding_after_sec = item["source_end_sec"] - timeline_cursor_sec
            padding_bytes = seconds_to_output_frame_bytes(padding_after_sec)
            aligned_output.extend(b"\x00" * padding_bytes)
            padding_after_sec = padding_bytes / float(output_bytes_per_second())
            timeline_cursor_sec += padding_after_sec
            total_padding_sec += padding_after_sec
        elif timeline_cursor_sec > item["source_end_sec"]:
            overrun_sec = timeline_cursor_sec - item["source_end_sec"]
            drift_warnings += 1

        debug_item = dict(item)
        debug_item["timeline_end_sec"] = timeline_cursor_sec
        debug_item["padding_after_sec"] = padding_after_sec
        debug_item["overrun_sec"] = overrun_sec
        used_manifest.append(debug_item)

    log(
        f"Inline alignment inserted {total_padding_sec:.2f} seconds of silence, "
        f"used latency compensation, and saw {drift_warnings} boundary overruns."
    )
    return aligned_output, used_manifest


def fit_output_duration_to_input(config: dict, input_wav: Path, output_pcm: bytearray) -> None:
    target_duration = get_wav_duration_seconds(input_wav)
    output_duration = pcm_duration_seconds(output_pcm)

    if bool(config.get("pad_output_to_input_duration", True)) and output_duration < target_duration:
        missing_seconds = target_duration - output_duration
        missing_bytes = seconds_to_output_frame_bytes(missing_seconds)
        output_pcm.extend(b"\x00" * missing_bytes)
        missing_seconds = missing_bytes / float(output_bytes_per_second())
        log(
            f"Padded translated audio with {missing_seconds:.2f} seconds of silence "
            f"to match the source duration."
        )

    if bool(config.get("trim_output_to_input_duration", False)):
        target_bytes = seconds_to_output_frame_bytes(target_duration)
        if len(output_pcm) > target_bytes:
            del output_pcm[target_bytes:]
            log("Trimmed translated audio to match the source duration.")


def write_alignment_manifest(output_wav: Path, manifest: list[dict], latency_offset_sec: float) -> None:
    if not manifest:
        return
    manifest_path = output_wav.with_suffix(".segments.json")
    payload = {
        "latency_offset_sec": round(latency_offset_sec, 3),
        "segments": [
            {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in item.items()
                if key != "output_audio_bytes"
            }
            for item in manifest
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote segment alignment manifest: {manifest_path.name}")


def write_palabra_input_audio_debug(config: dict, input_wav: Path, output_wav: Path) -> None:
    if not bool(config.get("write_alignment_debug_json", True)):
        return
    debug_path = output_wav.with_suffix(".palabra-input.wav")
    shutil.copyfile(input_wav, debug_path)
    log(f"Wrote Palabra input audio debug file: {debug_path.name}")


def write_palabra_tts_input_debug(config: dict, output_wav: Path, tts_init: dict, cue_messages: list[dict]) -> None:
    if not bool(config.get("write_alignment_debug_json", True)):
        return
    debug_path = output_wav.with_suffix(".palabra-input.json")
    payload = {
        "mode": "subtitle_tts",
        "init": tts_init,
        "messages": cue_messages,
    }
    debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote Palabra TTS input debug file: {debug_path.name}")


def write_silence_wav(path: Path, duration_sec: float) -> None:
    silence_bytes = seconds_to_output_frame_bytes(duration_sec)
    write_output_wav(path, b"\x00" * silence_bytes)


def build_atempo_filter(tempo: float) -> str:
    if tempo <= 0:
        raise RuntimeError("Tempo must be positive.")

    factors: list[float] = []
    remaining = tempo
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.5f}" for factor in factors)


def load_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        if channels != OUTPUT_CHANNELS or sample_width != OUTPUT_SAMPLE_WIDTH_BYTES or sample_rate != OUTPUT_SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected WAV format in {path.name}: rate={sample_rate}, channels={channels}, sample_width={sample_width}"
            )
        return wav_file.readframes(wav_file.getnframes())


def fit_output_wav_file_to_input(config: dict, input_wav: Path, output_wav: Path) -> None:
    pcm = bytearray(load_wav_pcm(output_wav))
    fit_output_duration_to_input(config, input_wav, pcm)
    write_output_wav(output_wav, pcm)


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def output_language_suffix(target_language: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in target_language.strip())
    return cleaned.upper() or "OUTPUT"


def output_stem(input_video: Path, target_language: str) -> str:
    return f"{input_video.stem}_dubbed_{output_language_suffix(target_language)}"


def default_audio_output_path(input_video: Path, audio_format: str, target_language: str) -> Path:
    return input_video.with_name(f"{output_stem(input_video, target_language)}.{audio_format}")


def default_video_output_path(input_video: Path, target_language: str) -> Path:
    suffix = input_video.suffix or ".mp4"
    return input_video.with_name(f"{output_stem(input_video, target_language)}{suffix}")


def classify_output_path(output_path: Path) -> tuple[str, str | None]:
    suffix = output_path.suffix.lower()
    if suffix == ".mp3":
        return "audio", "mp3"
    if suffix == ".wav":
        return "audio", "wav"
    if suffix in VIDEO_OUTPUT_EXTENSIONS:
        return "video", None
    supported = ", ".join(sorted(AUDIO_OUTPUT_EXTENSIONS | VIDEO_OUTPUT_EXTENSIONS))
    raise ValueError(f"Unsupported output extension '{output_path.suffix}'. Use one of: {supported}")


def build_ffmpeg_segment_output(
    config: dict,
    input_wav: Path,
    output_wav: Path,
    raw_output_wav: Path,
    work_dir: Path,
    raw_pcm: bytes | bytearray,
    manifest: list[dict],
) -> list[dict]:
    if not manifest:
        shutil.copyfile(raw_output_wav, output_wav)
        log("No usable segment metadata found. Copied raw translated WAV as final output.")
        return []

    work_dir.mkdir(exist_ok=True)
    concat_entries: list[Path] = []
    enriched_manifest: list[dict] = []
    raw_offset = 0
    timeline_cursor_sec = 0.0
    total_padding_sec = 0.0
    drift_warnings = 0
    speedup_count = 0
    max_speedup = max(1.0, float(config.get("segment_max_speedup", 1.15)))
    speedup_threshold_sec = max(0.0, float(config.get("segment_speedup_threshold_sec", 0.15)))
    alignment_strategy = str(config.get("segment_alignment_strategy", "pad_or_speedup")).strip().lower()
    if alignment_strategy not in {"pad_only", "pad_or_speedup"}:
        raise RuntimeError("segment_alignment_strategy must be 'pad_only' or 'pad_or_speedup'")

    for index, item in enumerate(manifest, start=1):
        segment_bytes = int(item["output_audio_bytes"])
        segment_pcm = raw_pcm[raw_offset : raw_offset + segment_bytes]
        raw_offset += len(segment_pcm)

        segment_duration_sec = pcm_duration_seconds(segment_pcm)
        source_duration_sec = float(item.get("source_duration_sec", 0.0))
        processed_duration_sec = segment_duration_sec
        tempo_applied = 1.0

        if segment_pcm:
            segment_path = work_dir / f"segment_{index:04d}.wav"
            write_output_wav(segment_path, segment_pcm)
            final_segment_path = segment_path

            if (
                alignment_strategy == "pad_or_speedup"
                and source_duration_sec > 0
                and segment_duration_sec - source_duration_sec >= speedup_threshold_sec
            ):
                required_speedup = segment_duration_sec / source_duration_sec
                if required_speedup <= max_speedup:
                    sped_path = work_dir / f"segment_{index:04d}.sped.wav"
                    run_ffmpeg(
                        ffmpeg_command(
                            config.get("ffmpeg_path", "ffmpeg"),
                            "-y",
                            "-i",
                            str(segment_path),
                            "-filter:a",
                            build_atempo_filter(required_speedup),
                            str(sped_path),
                        )
                    )
                    final_segment_path = sped_path
                    processed_duration_sec = get_wav_duration_seconds(sped_path)
                    tempo_applied = required_speedup
                    speedup_count += 1

            concat_entries.append(final_segment_path)
            timeline_cursor_sec += processed_duration_sec

        padding_after_sec = 0.0
        overrun_sec = 0.0
        if bool(config.get("pad_segments_to_source_timing", True)) and timeline_cursor_sec < item["source_end_sec"]:
            padding_after_sec = item["source_end_sec"] - timeline_cursor_sec
            silence_path = work_dir / f"silence_{index:04d}.wav"
            write_silence_wav(silence_path, padding_after_sec)
            concat_entries.append(silence_path)
            timeline_cursor_sec = item["source_end_sec"]
            total_padding_sec += padding_after_sec
        elif timeline_cursor_sec > item["source_end_sec"]:
            overrun_sec = timeline_cursor_sec - item["source_end_sec"]
            drift_warnings += 1

        debug_item = dict(item)
        debug_item["tempo_applied"] = round(tempo_applied, 5)
        debug_item["processed_audio_duration_sec"] = round(processed_duration_sec, 3)
        debug_item["timeline_end_sec"] = timeline_cursor_sec
        debug_item["padding_after_sec"] = padding_after_sec
        debug_item["overrun_sec"] = overrun_sec
        enriched_manifest.append(debug_item)

    if raw_offset < len(raw_pcm):
        tail_path = work_dir / "tail_unassigned.wav"
        write_output_wav(tail_path, raw_pcm[raw_offset:])
        concat_entries.append(tail_path)
        log("Kept trailing translated audio that had no completed phrase metadata.")

    combined_pcm = bytearray()
    for path in concat_entries:
        combined_pcm.extend(load_wav_pcm(path))
    write_output_wav(output_wav, combined_pcm)

    fit_output_wav_file_to_input(config, input_wav, output_wav)
    log(
        f"FFmpeg segment alignment inserted {total_padding_sec:.2f} seconds of silence, "
        f"sped up {speedup_count} segments, used latency compensation, and saw {drift_warnings} boundary overruns."
    )
    return enriched_manifest


def speedup_wav_file(config: dict, input_wav: Path, output_wav: Path, tempo: float) -> None:
    run_ffmpeg(
        ffmpeg_command(
            config.get("ffmpeg_path", "ffmpeg"),
            "-y",
            "-i",
            str(input_wav),
            "-filter:a",
            build_atempo_filter(tempo),
            str(output_wav),
        )
    )


def build_subtitle_timed_output(
    config: dict,
    cues: list[SubtitleCue],
    cue_audio: dict[int, bytes],
    output_wav: Path,
    work_dir: Path,
    target_duration_sec: float,
) -> list[dict]:
    work_dir.mkdir(exist_ok=True)
    output_pcm = bytearray()
    manifest: list[dict] = []
    total_padding_sec = 0.0
    speedup_count = 0
    delayed_count = 0
    leading_trim_total_sec = 0.0
    max_late_start_sec = 0.0
    max_speedup = max(1.0, float(config.get("subtitle_max_speedup", 2.0)))
    speedup_threshold_sec = max(0.0, float(config.get("segment_speedup_threshold_sec", 0.15)))
    silence_threshold = int(config.get("subtitle_silence_trim_threshold", 160))
    silence_padding_sec = max(0.0, float(config.get("subtitle_silence_trim_padding_sec", 0.0)))

    for cue_number, cue in enumerate(cues):
        requested_start_sec = cue.start_sec
        next_start_sec = cues[cue_number + 1].start_sec if cue_number + 1 < len(cues) else target_duration_sec
        next_start_sec = min(next_start_sec, target_duration_sec)
        timeline_cursor_sec = pcm_duration_seconds(output_pcm)
        actual_start_sec = max(requested_start_sec, timeline_cursor_sec)
        late_start_sec = max(0.0, actual_start_sec - requested_start_sec)
        if late_start_sec > 0:
            delayed_count += 1
            max_late_start_sec = max(max_late_start_sec, late_start_sec)

        if timeline_cursor_sec < actual_start_sec:
            padding_sec = actual_start_sec - timeline_cursor_sec
            output_pcm.extend(b"\x00" * seconds_to_output_frame_bytes(padding_sec))
            total_padding_sec += padding_sec

        original_pcm = cue_audio.get(cue.index, b"")
        original_duration_sec = pcm_duration_seconds(original_pcm)
        segment_pcm, leading_trim_sec, trailing_trim_sec = trim_pcm_silence(
            original_pcm, threshold=silence_threshold, padding_sec=silence_padding_sec
        )
        leading_trim_total_sec += leading_trim_sec
        trimmed_duration_sec = pcm_duration_seconds(segment_pcm)
        processed_duration_sec = trimmed_duration_sec
        tempo_applied = 1.0

        available_until_next_sec = max(0.0, next_start_sec - actual_start_sec)
        if (
            segment_pcm
            and available_until_next_sec > 0
            and trimmed_duration_sec - available_until_next_sec >= speedup_threshold_sec
        ):
            required_speedup = trimmed_duration_sec / available_until_next_sec
            tempo_applied = min(required_speedup, max_speedup)
            if tempo_applied > 1.0:
                segment_path = work_dir / f"subtitle_{cue.index:04d}.wav"
                sped_path = work_dir / f"subtitle_{cue.index:04d}.sped.wav"
                write_output_wav(segment_path, segment_pcm)
                speedup_wav_file(config, segment_path, sped_path, tempo_applied)
                segment_pcm = load_wav_pcm(sped_path)
                processed_duration_sec = pcm_duration_seconds(segment_pcm)
                speedup_count += 1

        output_pcm.extend(segment_pcm)
        timeline_end_sec = pcm_duration_seconds(output_pcm)
        overrun_sec = max(0.0, timeline_end_sec - next_start_sec) if cue_number + 1 < len(cues) else 0.0

        manifest.append(
            {
                "cue_index": cue.index,
                "source_start_sec": cue.start_sec,
                "source_end_sec": cue.end_sec,
                "requested_timeline_start_sec": requested_start_sec,
                "timeline_start_sec": actual_start_sec,
                "late_start_sec": late_start_sec,
                "next_start_sec": next_start_sec,
                "available_duration_sec": available_until_next_sec,
                "output_audio_duration_sec": original_duration_sec,
                "trimmed_audio_duration_sec": trimmed_duration_sec,
                "processed_audio_duration_sec": processed_duration_sec,
                "leading_silence_trimmed_sec": leading_trim_sec,
                "trailing_silence_trimmed_sec": trailing_trim_sec,
                "tempo_applied": round(tempo_applied, 5),
                "timeline_end_sec": timeline_end_sec,
                "clipped_sec": 0.0,
                "overrun_sec": overrun_sec,
                "text": cue.text,
            }
        )

    if bool(config.get("pad_output_to_input_duration", True)) and pcm_duration_seconds(output_pcm) < target_duration_sec:
        padding_sec = target_duration_sec - pcm_duration_seconds(output_pcm)
        output_pcm.extend(b"\x00" * seconds_to_output_frame_bytes(padding_sec))
        total_padding_sec += padding_sec

    if bool(config.get("trim_output_to_input_duration", False)) and pcm_duration_seconds(output_pcm) > target_duration_sec:
        del output_pcm[seconds_to_output_frame_bytes(target_duration_sec):]

    write_output_wav(output_wav, output_pcm)
    log(
        f"Subtitle alignment trimmed {leading_trim_total_sec:.2f} seconds of leading generated silence, "
        f"inserted {total_padding_sec:.2f} seconds of silence, sped up {speedup_count} phrases, "
        f"and delayed {delayed_count} phrase starts by up to {max_late_start_sec:.2f} seconds without clipping speech."
    )
    return manifest


async def run_subtitle_pipeline(config: dict) -> None:
    input_video = Path(config["input_video"]).resolve()
    subtitle_path = Path(config["subtitle_path"]).resolve()
    audio_format = config.get("audio_format")
    output_wav = Path(config["working_wav"]).resolve()
    output_audio = Path(config["output_audio"]).resolve() if config.get("output_audio") else None
    output_video = Path(config["output_video"]).resolve() if config.get("output_video") else None
    run_work_dir = Path(tempfile.mkdtemp(prefix=f".{output_wav.stem}.", suffix=".subtitle.work", dir=output_wav.parent))
    temp_input_wav = run_work_dir / "input_16k_mono.wav"
    subtitle_work_dir = run_work_dir / "subtitle_segments"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    try:
        cues = parse_sbv_file(subtitle_path)
        original_cue_count = len(cues)
        log(f"Loaded {len(cues)} subtitle cues from: {subtitle_path.name}")
        log(f"Extracting audio duration from: {input_video.name}")
        extract_audio_to_wav(input_video, temp_input_wav, config.get("ffmpeg_path", "ffmpeg"))
        input_duration_sec = get_wav_duration_seconds(temp_input_wav)
        cues, clipped_to_video, skipped_cue_count = trim_subtitle_cues_to_duration(cues, input_duration_sec)
        if clipped_to_video:
            log(
                f"Warning: subtitle timings extend past the {input_duration_sec:.2f}s video duration. "
                f"Using {len(cues)}/{original_cue_count} cues and trimming voiceover to the video length."
            )
            if skipped_cue_count:
                log(f"Skipped {skipped_cue_count} subtitle cues that start after the video ends.")
        if not cues:
            raise RuntimeError("No subtitle cues start before the video ends.")

        log("Creating Palabra TTS session...")
        session = create_session(config["client_id"], config["client_secret"])
        ws_tts_url = session.get("ws_tts_url")
        if not ws_tts_url:
            raise RuntimeError("Palabra session response did not include ws_tts_url for realtime TTS")
        publisher = session["publisher"]
        delimiter = "&" if "?" in ws_tts_url else "?"
        endpoint = f"{ws_tts_url}{delimiter}token={urllib.parse.quote(publisher)}"

        cue_audio: dict[int, bytes] = {}
        tts_init_debug: dict | None = None
        tts_input_messages: list[dict] = []
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(animate_progress("Palabra subtitle TTS running", heartbeat_stop))
        try:
            async with websockets.connect(endpoint, max_size=None) as ws:
                log("Connected. Sending TTS init configuration...")
                tts_init = build_tts_init(config)
                tts_init_debug = tts_init
                log(
                    f"Using Palabra TTS voice_id={tts_init['voice_options']['voice_id']} "
                    f"language={tts_init['language']} model={tts_init['model']}"
                )
                if tts_init["language"] != str(config.get("source_language", "")).strip():
                    log(
                        "Note: realtime TTS voices may be language-specific. If this voice_id belongs to another "
                        "language, Palabra may substitute a generic voice for subtitle TTS."
                    )
                await ws.send(json.dumps(tts_init))
                max_chars = int(config.get("subtitle_tts_max_chars", 256))
                chunk_delay_sec = max(0.0, float(config.get("subtitle_tts_text_chunk_delay_ms", 50)) / 1000.0)
                phrase_delay_sec = max(0.0, float(config.get("subtitle_tts_phrase_delay_ms", 0)) / 1000.0)
                if chunk_delay_sec > 0:
                    log(f"Using {chunk_delay_sec * 1000:.0f} ms delay between TTS text chunks.")
                if phrase_delay_sec > 0:
                    log(f"Using {phrase_delay_sec * 1000:.0f} ms delay between subtitle TTS phrases.")
                for position, cue in enumerate(cues, start=1):
                    text_chunks = split_tts_text(cue.text, max_chars)
                    if len(text_chunks) > 1:
                        log(f"Cue {cue.index} is streamed to TTS in {len(text_chunks)} text chunks because it is long.")
                    update_live_status("sent", f"Generating subtitle phrase {position}/{len(cues)}...")
                    generation_id = f"subtitle-{cue.index:04d}"
                    messages = build_tts_text_messages(text_chunks, generation_id)
                    for message_index, message in enumerate(messages):
                        tts_input_messages.append(
                            {
                                "cue_index": cue.index,
                                "message": message,
                            }
                        )
                        await ws.send(json.dumps(message))
                        if chunk_delay_sec > 0 and message_index + 1 < len(messages):
                            await asyncio.sleep(chunk_delay_sec)
                    cue_audio[cue.index] = await receive_tts_generation(
                        ws, generation_id, float(config.get("output_inactivity_timeout_sec", 8.0))
                    )
                    if phrase_delay_sec > 0 and position < len(cues):
                        await asyncio.sleep(phrase_delay_sec)
        finally:
            heartbeat_stop.set()
            await heartbeat_task

        log(f"Writing subtitle-timed WAV: {output_wav.name}")
        subtitle_output_config = dict(config)
        subtitle_output_config["trim_output_to_input_duration"] = True
        manifest = build_subtitle_timed_output(
            subtitle_output_config, cues, cue_audio, output_wav, subtitle_work_dir, input_duration_sec
        )
        if bool(config.get("write_alignment_debug_json", True)):
            write_alignment_manifest(output_wav, manifest, 0.0)
            if tts_init_debug is not None:
                write_palabra_tts_input_debug(config, output_wav, tts_init_debug, tts_input_messages)
        if output_video:
            log(f"Muxing subtitle voiceover into video: {output_video.name}")
            mux_audio_back_to_video(
                input_video,
                output_wav,
                output_video,
                config.get("ffmpeg_path", "ffmpeg"),
                normalize_mp4_audio_bitrate(config),
            )
        if output_audio and audio_format == "mp3":
            log(f"Writing subtitle voiceover MP3: {output_audio.name}")
            convert_wav_to_mp3(output_wav, output_audio, config.get("ffmpeg_path", "ffmpeg"))
    finally:
        cleanup_candidates = [run_work_dir]
        if audio_format != "wav":
            cleanup_candidates.append(output_wav)
        for path in cleanup_candidates:
            try:
                cleanup_path(path)
            except OSError as exc:
                log(f"Could not delete temporary file {path}: {exc}")


async def run_pipeline(config: dict) -> None:
    input_video = Path(config["input_video"]).resolve()
    audio_format = config.get("audio_format")
    output_wav = Path(config["working_wav"]).resolve()
    output_audio = Path(config["output_audio"]).resolve() if config.get("output_audio") else None
    output_video = Path(config["output_video"]).resolve() if config.get("output_video") else None
    run_work_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_wav.stem}.", suffix=".work", dir=output_wav.parent)
    )
    temp_input_wav = run_work_dir / "input_16k_mono.wav"
    raw_output_wav = run_work_dir / "raw.wav"
    segment_work_dir = run_work_dir / "ffmpeg_segments"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    try:
        log(f"Extracting audio from: {input_video.name}")
        extract_audio_to_wav(input_video, temp_input_wav, config.get("ffmpeg_path", "ffmpeg"))
        write_palabra_input_audio_debug(config, temp_input_wav, output_wav)
        log("Creating Palabra streaming session...")
        session = create_session(config["client_id"], config["client_secret"])

        ws_url = session["ws_url"]
        publisher = session["publisher"]
        delimiter = "&" if "?" in ws_url else "?"
        endpoint = f"{ws_url}{delimiter}token={urllib.parse.quote(publisher)}"

        receive_state = ReceiveState(stream_start_monotonic=time.monotonic())
        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(animate_progress("Palabra pipeline running", heartbeat_stop))

        log("Connecting to Palabra WebSocket...")
        try:
            async with websockets.connect(endpoint, max_size=None) as ws:
                log("Connected. Sending translation task configuration...")
                await ws.send(json.dumps(build_set_task(config)))
                sender_done = asyncio.Event()
                sender = asyncio.create_task(send_audio(ws, temp_input_wav, int(config.get("audio_chunk_ms", 320))))
                receiver = asyncio.create_task(
                    receive_audio(
                        ws,
                        receive_state,
                        float(config.get("output_inactivity_timeout_sec", 8.0)),
                        sender_done,
                    )
                )

                try:
                    done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                    if receiver in done:
                        receiver.result()
                        raise RuntimeError("Translated audio receiver stopped before source audio finished streaming.")

                    sender.result()
                    sender_done.set()
                    try:
                        await ws.send(json.dumps({"message_type": "end_task", "data": {}}))
                    except Exception:
                        pass
                    await receiver
                except Exception:
                    sender_done.set()
                    if not sender.done():
                        sender.cancel()
                    if not receiver.done():
                        receiver.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)
                    raise

        finally:
            heartbeat_stop.set()
            await heartbeat_task

        raw_pcm = bytearray(receive_state.continuous_output_pcm)
        log(f"Writing raw translated WAV: {raw_output_wav.name}")
        write_output_wav(raw_output_wav, raw_pcm)

        manifest, latency_offset_sec = build_segment_manifest(config, receive_state)
        alignment_mode = normalize_alignment_mode(config)

        if alignment_mode == "raw":
            output_pcm = bytearray(raw_pcm)
            used_manifest = manifest
        elif alignment_mode == "inline":
            output_pcm, used_manifest = build_inline_aligned_output(config, receive_state, manifest)
        else:
            used_manifest = build_ffmpeg_segment_output(
                config,
                temp_input_wav,
                output_wav,
                raw_output_wav,
                segment_work_dir,
                raw_pcm,
                manifest,
            )
            if bool(config.get("write_alignment_debug_json", True)):
                write_alignment_manifest(output_wav, used_manifest, latency_offset_sec)
            if output_video:
                log(f"Muxing translated audio into video: {output_video.name}")
                mux_audio_back_to_video(
                    input_video,
                    output_wav,
                    output_video,
                    config.get("ffmpeg_path", "ffmpeg"),
                    normalize_mp4_audio_bitrate(config),
                )
            if output_audio and audio_format == "mp3":
                log(f"Writing translated MP3: {output_audio.name}")
                convert_wav_to_mp3(output_wav, output_audio, config.get("ffmpeg_path", "ffmpeg"))
            return

        fit_output_duration_to_input(config, temp_input_wav, output_pcm)
        log(f"Writing translated WAV: {output_wav.name}")
        write_output_wav(output_wav, output_pcm)
        if bool(config.get("write_alignment_debug_json", True)):
            write_alignment_manifest(output_wav, used_manifest, latency_offset_sec)
        if output_video:
            log(f"Muxing translated audio into video: {output_video.name}")
            mux_audio_back_to_video(
                input_video,
                output_wav,
                output_video,
                config.get("ffmpeg_path", "ffmpeg"),
                normalize_mp4_audio_bitrate(config),
            )
        if output_audio and audio_format == "mp3":
            log(f"Writing translated MP3: {output_audio.name}")
            convert_wav_to_mp3(output_wav, output_audio, config.get("ffmpeg_path", "ffmpeg"))
    finally:
        cleanup_candidates = [run_work_dir]
        if audio_format != "wav":
            cleanup_candidates.extend([output_wav, output_wav.with_suffix(".segments.json")])
        for path in cleanup_candidates:
            try:
                cleanup_path(path)
            except OSError as exc:
                log(f"Could not delete temporary file {path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate translated speech audio from a video via Palabra.")
    parser.add_argument("input_video", help="Input video file to translate.")
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Optional explicit output path. The extension selects mp3, wav, or video output.",
    )
    parser.add_argument(
        "--audio",
        nargs="?",
        const="mp3",
        choices=("mp3", "wav"),
        help="Save translated audio as mp3 or wav. Defaults to mp3 when no output switch is provided.",
    )
    parser.add_argument("--video", action="store_true", help="Save a video with the translated audio muxed in.")
    parser.add_argument("--subtitles", help="Create the voiceover from a timed .sbv subtitle file instead of source audio transcription.")
    parser.add_argument("--voice-id", help="Override the configured Palabra voice_id.")
    args = parser.parse_args()
    if args.output_path and (args.audio is not None or args.video):
        parser.error("output_path cannot be combined with --audio or --video")
    return args


def main() -> int:
    args = parse_args()
    config_path = Path("config.toml").resolve()
    load_dotenv(config_path.with_name(".env"))
    config = apply_env_overrides(load_config(config_path))
    input_video = Path(args.input_video).resolve()
    target_language = str(config.get("target_language", ""))
    output_audio = None
    output_video = None

    if args.output_path:
        explicit_output = Path(args.output_path).resolve()
        try:
            output_kind, audio_format = classify_output_path(explicit_output)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if output_kind == "audio":
            output_audio = explicit_output
        else:
            output_video = explicit_output
    else:
        audio_format = args.audio
        if args.subtitles and audio_format is None and not args.video:
            output_video = default_video_output_path(input_video, target_language)
        elif audio_format is None and not args.video:
            audio_format = "mp3"
        if audio_format:
            output_audio = default_audio_output_path(input_video, audio_format, target_language)
        if args.video:
            output_video = default_video_output_path(input_video, target_language)

    working_wav = output_audio if audio_format == "wav" and output_audio else default_audio_output_path(
        input_video, "wav", target_language
    )
    if audio_format != "wav":
        working_parent = output_video.parent if output_video else (output_audio.parent if output_audio else input_video.parent)
        working_stem = output_video.stem if output_video else (output_audio.stem if output_audio else output_stem(input_video, target_language))
        working_wav = working_parent / f"{working_stem}.working.wav"

    config["input_video"] = str(input_video)
    config["audio_format"] = audio_format
    config["working_wav"] = str(working_wav)
    config["output_audio"] = str(output_audio) if output_audio else None
    config["output_video"] = str(output_video) if output_video else None
    config["subtitle_path"] = str(Path(args.subtitles).resolve()) if args.subtitles else None

    if args.voice_id:
        config["voice_id"] = args.voice_id

    required_keys = [
        "client_id",
        "client_secret",
        "source_language",
        "target_language",
    ]
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(missing)}")

    try:
        if args.subtitles:
            asyncio.run(run_subtitle_pipeline(config))
        else:
            asyncio.run(run_pipeline(config))
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if config.get("output_audio"):
        print(f"Translated audio saved to: {Path(config['output_audio']).resolve()}")
    if config.get("output_video"):
        print(f"Translated video saved to: {Path(config['output_video']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
