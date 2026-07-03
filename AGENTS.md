## Maintaining this file

When working on this project, update this AGENTS.md file if you discover durable information that will help future Codex sessions.

## Coding rules
- Keep changes minimal and focused.
- Do not edit generated files unless explicitly asked.
- Do not modify .env files.
- Before finishing, run the relevant test/lint command when possible.
- In the Palabra dubbing path, keep the no-drop safeguards unless intentionally testing alternatives: `segment_confirmation_silence_threshold = 0.7`, `desired_queue_level_ms = 5000`, `max_queue_level_ms = 20000`, and `auto_tempo = true`. The current active test uses Palabra tempo `min_tempo = 1.1` and `max_tempo = 1.3`; Palabra's published defaults list `min_tempo = 1.15` and `max_tempo = 1.45`.
- Palabra Zoom is the local reference for Speech-to-Speech `set_task` voice wiring: fixed voices go in `translations[].speech_generation.voice_id`. Subtitle TTS uses Palabra's realtime TTS WebSocket instead, where the voice is sent as `voice_options.voice_id`.

# Codex workspace instructions

- For Python commands in this workspace, use `& "C:\Users\marti\.venvs\palabra_dub\Scripts\python.exe" ...` from PowerShell.
- Install or update Python dependencies with `& "C:\Users\marti\.venvs\palabra_dub\Scripts\python.exe" -m pip ...`.
- Do not create or use a project-local `.venv` unless explicitly requested.
