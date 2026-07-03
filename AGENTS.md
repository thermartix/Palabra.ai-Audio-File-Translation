## Maintaining this file

When working on this project, update this AGENTS.md file if you discover durable information that will help future Codex sessions.

## Coding rules
- Keep changes minimal and focused.
- Do not edit generated files unless explicitly asked.
- Do not modify .env files.
- Before finishing, run the relevant test/lint command when possible.
- In the Palabra dubbing path, keep Palabra's no-drop defaults unless intentionally testing alternatives: `segment_confirmation_silence_threshold = 0.7`, `desired_queue_level_ms = 5000`, `max_queue_level_ms = 20000`, `auto_tempo = true`, `min_tempo = 1.0`, and `max_tempo = 1.45`.

# Codex workspace instructions

- For Python commands in this workspace, use `& "C:\Users\marti\.venvs\palabra_dub\Scripts\python.exe" ...` from PowerShell.
- Install or update Python dependencies with `& "C:\Users\marti\.venvs\palabra_dub\Scripts\python.exe" -m pip ...`.
- Do not create or use a project-local `.venv` unless explicitly requested.
