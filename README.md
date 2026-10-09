# RogerVGB

Roger Very Good Bot — your AI, your rules.

A modular, local-first AI workspace. Version 0.1 focuses on three useful features:

1. **Chat:** streamed conversations with Ollama and saved chat history.
2. **Custom assistants:** save names, descriptions, instructions, and preferred models.
3. **Model arena:** run the same prompt against two different local models and compare answers.

## Run locally

Requirements: a modern browser and [Ollama](https://ollama.com/) with at least one model downloaded.

1. Start Ollama and download a model, e.g. `ollama pull qwen3:4b`.
2. In this repository directory, run `python3 -m http.server 8000`.
3. Open http://localhost:8000.
4. Open settings, enter `http://localhost:11434`, choose a model, and test the connection.

### Browser access / CORS

This static frontend sends requests directly from your browser to your configured Ollama server. Depending on your Ollama version and browser configuration, cross-origin requests may be blocked. If you see a CORS/network error, configure Ollama to allow the page origin `http://localhost:8000` and restart Ollama. Do not expose Ollama to the public internet.

## Modules

- `index.html`: app shell and view containers
- `styles.css`: responsive visual system
- `js/api.js`: Ollama HTTP API and streaming parser
- `js/storage.js`: localStorage persistence
- `js/ui.js`: DOM helpers, escaped Markdown renderer, notifications
- `js/chat.js`: conversations, history, streamed chat
- `js/assistants.js`: custom assistant CRUD
- `js/arena.js`: parallel model comparisons
- `js/main.js`: navigation and app wiring

## Privacy and limitations

Chats, assistants, and settings are stored in this browser's localStorage (not encrypted). Prompts are sent to the Ollama URL you configure. This version includes no RogerVGB server, account system, cloud model API, or telemetry. Do not put sensitive information into saved chats on a shared device. The arena displays responses side by side without inventing a winner or score.

## Iterate before expanding

Test with actual models, note bugs, fix them, then add the next feature. Planned follow-ups include editing/regenerating messages, branching, import/export, provider adapters, and automated tests.
