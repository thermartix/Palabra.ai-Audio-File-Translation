import argparse
import asyncio
import base64
import dataclasses
import json
import os
import shutil
import subprocess
import sys
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


def extract_audio_to_wav(input_video: Path, output_wav: Path, ffmpeg_path: str) -> None:
    command = [
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
    ]
    subprocess.run(command, check=True)


def mux_audio_back_to_video(input_video: Path, input_audio: Path, output_video: Path, ffmpeg_path: str) -> None:
    command = [
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
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(command, check=True)


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def log(message: str) -> None:
    global _ANIMATION_ACTIVE
    if _ANIMATION_ACTIVE:
        sys.stdout.write("\r" + (" " * ANIMATION_WIDTH) + "\r")
        _ANIMATION_ACTIVE = False
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
    frames = int(INPUT_SAMPLE_RATE * (chunk_ms / 1000.0))
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

        log(f"Streaming audio to Palabra in real time ({total_seconds / 60:.1f} minutes total)...")

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
                log(
                    f"Sent {sent_seconds / 60:.1f}/{total_seconds / 60:.1f} minutes "
                    f"({percent:.1f}%). Waiting for translated audio as it streams back..."
                )
                last_progress_print = sent_seconds
            await asyncio.sleep(chunk_ms / 1000.0)

        log("Finished sending source audio.")


async def receive_audio(ws, receive_state: ReceiveState, inactivity_timeout_sec: float) -> None:
    received_chunks = 0
    received_bytes = 0
    last_report = time.monotonic()
    while True:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=inactivity_timeout_sec)
        except asyncio.TimeoutError:
            log("No more translated audio received within timeout window. Finalizing WAV file...")
            return

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
                is_last_chunk = transcription.get("last_chunk") is True
                if is_last_chunk:
                    segment.last_chunk_received = True
                receive_state.chunk_events.append((transcription_id, decoded, is_last_chunk))
                received_chunks += 1
                received_bytes += len(decoded)
                now = time.monotonic()
                if now - last_report >= 30.0:
                    approx_seconds = received_bytes / float(output_bytes_per_second())
                    log(
                        f"Received {received_chunks} translated chunks "
                        f"(about {approx_seconds / 60:.1f} minutes of output audio so far)..."
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

    previous_end_sec = 0.0
    previous_source_end_sec = 0.0
    for item in manifest:
        source_end_sec = max(0.0, item["source_end_sec_raw"] - latency_offset_sec)
        source_end_sec = max(source_end_sec, previous_end_sec)
        source_duration_sec = max(0.0, source_end_sec - previous_source_end_sec)
        item["source_end_sec"] = source_end_sec
        item["source_duration_sec"] = source_duration_sec
        previous_end_sec = source_end_sec
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


def build_ffmpeg_segment_output(
    config: dict,
    input_wav: Path,
    output_wav: Path,
    raw_output_wav: Path,
    raw_pcm: bytes | bytearray,
    manifest: list[dict],
) -> list[dict]:
    if not manifest:
        shutil.copyfile(raw_output_wav, output_wav)
        log("No usable segment metadata found. Copied raw translated WAV as final output.")
        return []

    work_dir = output_wav.parent / f"{output_wav.stem}_ffmpeg_segments"
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
                        [
                            config.get("ffmpeg_path", "ffmpeg"),
                            "-y",
                            "-i",
                            str(segment_path),
                            "-filter:a",
                            build_atempo_filter(required_speedup),
                            str(sped_path),
                        ]
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

    concat_list_path = work_dir / "concat.txt"
    concat_lines = [f"file '{path.resolve().as_posix()}'" for path in concat_entries]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    run_ffmpeg(
        [
            config.get("ffmpeg_path", "ffmpeg"),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_wav),
        ]
    )
    fit_output_wav_file_to_input(config, input_wav, output_wav)
    log(
        f"FFmpeg segment alignment inserted {total_padding_sec:.2f} seconds of silence, "
        f"sped up {speedup_count} segments, used latency compensation, and saw {drift_warnings} boundary overruns."
    )
    return enriched_manifest


async def run_pipeline(config: dict) -> None:
    input_video = Path(config["input_video"]).resolve()
    output_wav = Path(config["output_wav"]).resolve()
    output_video = Path(config["output_video"]).resolve() if config.get("output_video") else None
    temp_input_wav = output_wav.with_name(f"{output_wav.stem}.input_16k_mono.wav")
    raw_output_wav = output_wav.with_name(f"{output_wav.stem}.raw.wav")

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    log(f"Extracting audio from: {input_video.name}")
    extract_audio_to_wav(input_video, temp_input_wav, config.get("ffmpeg_path", "ffmpeg"))
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
            sender = asyncio.create_task(send_audio(ws, temp_input_wav, int(config.get("audio_chunk_ms", 320))))
            receiver = asyncio.create_task(
                receive_audio(ws, receive_state, float(config.get("output_inactivity_timeout_sec", 8.0)))
            )

            await sender
            await receiver

            try:
                await ws.send(json.dumps({"message_type": "end_task", "data": {}}))
            except Exception:
                pass
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
        used_manifest = build_ffmpeg_segment_output(config, temp_input_wav, output_wav, raw_output_wav, raw_pcm, manifest)
        if bool(config.get("write_alignment_debug_json", True)):
            write_alignment_manifest(output_wav, used_manifest, latency_offset_sec)
        if output_video:
            log(f"Muxing translated audio into video: {output_video.name}")
            mux_audio_back_to_video(input_video, output_wav, output_video, config.get("ffmpeg_path", "ffmpeg"))
        return

    fit_output_duration_to_input(config, temp_input_wav, output_pcm)
    log(f"Writing translated WAV: {output_wav.name}")
    write_output_wav(output_wav, output_pcm)
    if bool(config.get("write_alignment_debug_json", True)):
        write_alignment_manifest(output_wav, used_manifest, latency_offset_sec)
    if output_video:
        log(f"Muxing translated audio into video: {output_video.name}")
        mux_audio_back_to_video(input_video, output_wav, output_video, config.get("ffmpeg_path", "ffmpeg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate translated speech audio from a video via Palabra.")
    parser.add_argument("input_video", help="Input video file to translate.")
    parser.add_argument("output_wav", help="Final translated WAV file to generate.")
    parser.add_argument("output_video", nargs="?", help="Optional output video file with the translated audio muxed in.")
    parser.add_argument("--voice-id", help="Override the configured Palabra voice_id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path("config.toml").resolve()
    load_dotenv(config_path.with_name(".env"))
    config = apply_env_overrides(load_config(config_path))
    config["input_video"] = args.input_video
    config["output_wav"] = args.output_wav
    config["output_video"] = args.output_video

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
        asyncio.run(run_pipeline(config))
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Translated WAV saved to: {Path(config['output_wav']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
