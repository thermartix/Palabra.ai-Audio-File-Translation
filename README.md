# Palabra Audio Dub Pipeline

This project uses Python to:

1. Extract mono PCM audio from a source video with `ffmpeg`
2. Stream that audio to the Palabra WebSocket API
3. Receive translated speech audio back from Palabra
4. Save the result as a WAV file for later muxing back into the video

This implementation supports either a fixed `voice_id` or Palabra `voice_cloning`.

## What "voice cloning" means

In Palabra, voice cloning means the translated speech tries to sound like the original speaker.

If you want a fixed voice, set:

- `voice_cloning: false`
- `voice_id: "<your configured voice id>"`

If you want speaker-matching behavior, set:

- `speech_mode = "voice_cloning"`

For the most predictable dubbing output, `voice_id` is usually the better choice.

## Requirements

- Python 3.10+
- `ffmpeg` available on `PATH`
- Palabra API credentials

Install Python dependency:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` and fill in your Palabra credentials.
2. Copy `config.example.toml` to `config.toml` and fill in the non-secret settings.

Secrets are read from `.env`:

- `PALABRA_CLIENT_ID`
- `PALABRA_CLIENT_SECRET`

Important TOML fields:

- `client_id`
- `client_secret`
- `source_language`
- `target_language`
- `speech_mode`
- `voice_id`
- `alignment_mode`

Speech mode options:

- `speech_mode = "voice_id"`: use the fixed configured `voice_id`
- `speech_mode = "voice_cloning"`: mimic the original speaker instead

When `speech_mode = "voice_cloning"`, `voice_id` is ignored.

Dubbing-oriented timing fields:

- `auto_tempo = false`
- `min_tempo = 1.0`
- `max_tempo = 1.0`
- `pad_output_to_input_duration = true`
- `alignment_mode = "ffmpeg_segments"`

These help keep output timing closer to the source and pad trailing silence when the translated speech is shorter.

Phrase-alignment fields:

- `segment_confirmation_silence_threshold = 0.7`
- `sentence_splitter_enabled = true`
- `pad_segments_to_source_timing = true`
- `normalize_source_timestamps = true`
- `segment_alignment_strategy = "pad_or_speedup"`
- `segment_max_speedup = 1.15`
- `segment_speedup_threshold_sec = 0.15`
- `write_alignment_debug_json = true`

Alignment mode options:

- `alignment_mode = "raw"`: keep Palabra's continuous stream untouched
- `alignment_mode = "inline"`: insert silence directly into the in-memory stream at phrase boundaries
- `alignment_mode = "ffmpeg_segments"`: save the raw stream untouched, then do an offline phrase-aware rebuild with `ffmpeg`

`normalize_source_timestamps = true` compensates for the fact that Palabra phrase events arrive later than the original source speech, which helps reduce artificial leading silence.

With `segment_alignment_strategy = "pad_or_speedup"`, the offline `ffmpeg_segments` mode will:

- pad short sections with silence
- speed up slightly too-long sections up to `segment_max_speedup`
- leave very overlong sections mostly intact rather than introducing extreme compression

## Usage

Default MP3 audio output:

```bash
python palabra_dub.py input_file.mp4
```

Explicit audio output:

```bash
python palabra_dub.py input_file.mp4 --audio mp3
python palabra_dub.py input_file.mp4 --audio wav
python palabra_dub.py input_file.mp4 custom_output.mp3
python palabra_dub.py input_file.mp4 custom_output.wav
```

Video output only:

```bash
python palabra_dub.py input_file.mp4 --video
python palabra_dub.py input_file.mp4 custom_output.mp4
```

Audio and video output together:

```bash
python palabra_dub.py input_file.mp4 --audio wav --video
```

Optional override:

```bash
python palabra_dub.py input_file.mp4 --audio --voice-id YOUR_OTHER_VOICE_ID
```

Without an explicit output path, outputs are saved next to the input as `input_file_dubbed_DE.mp3`, `input_file_dubbed_DE.wav`, and/or `input_file_dubbed_DE.mp4`, using the configured `target_language` code. With an explicit `output.xxx`, the extension selects audio or video output and the provided path is used as-is.

## Repo Safety

- `.env` is ignored by `.gitignore` and should not be committed.
- `config.toml` is now safe to commit because credentials are no longer stored there.
- Because the credentials were previously stored in `config.toml`, rotating those Palabra keys is a good idea before publishing the repo.

## Notes

- The script extracts audio as `pcm_s16le`, `16 kHz`, mono, which matches the Palabra WebSocket input configuration used here.
- Palabra returns translated audio chunks. This script rebuilds them into a WAV file at `24 kHz` mono PCM, matching the documented default output.
- By default, the script disables Palabra auto-tempo and pads trailing silence if the translated audio is shorter than the input.
- The script creates temporary `*.raw.wav`, extracted input WAV, and segment working files during processing, then deletes them when the run finishes.
- The default `ffmpeg_segments` mode is safer than inline rewriting because it leaves the raw translated stream untouched and does the timing work afterward.
- If you pass `--video`, the script muxes the translated audio back into the video automatically with `ffmpeg`.

## Example mux command

```bash
ffmpeg -i input.mp4 -i translated.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest output.mp4
```

## Sources

- https://docs.palabra.ai/docs/streaming_api
- https://docs.palabra.ai/docs/streaming_api/translation_settings_breakdown
- https://docs.palabra.ai/docs/streaming_api/publishing_and_receiving_audio
- https://docs.palabra.ai/docs/streaming_api/session
