# openchat TUI

Terminal version of the openchat iOS app: a ChatGPT-style chat with **NVIDIA NIM** models, built with [Textual](https://textual.textualize.io/). It has a chat history sidebar, streaming replies, optional "thinking" (reasoning) text, and markdown-rendered answers.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

Put your API key in `config.py`:

```python
NVIDIA_API_KEY = "nvapi-..."
```

(If it's left as `""`, the `NVIDIA_API_KEY` environment variable is used instead.)

## Run

Install once as a global command (editable, so edits to `config.py` apply immediately):

```bash
cd openchat_tui
uv tool install --editable .
```

Then from anywhere:

```bash
openchat
```

Or, without installing, from inside this folder: `uv run openchat`.

To add a dependency: `uv add <package>`, then `uv tool install --editable . --force` to refresh the global command. To remove the command: `uv tool uninstall openchat-tui`.

## Keys

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+J` / `Shift+Enter` | Newline in message |
| `Esc` | Stop generating |
| `Ctrl+N` | New chat |
| `Ctrl+D` | Delete current chat |
| `Ctrl+O` | Pick model |
| `Ctrl+T` | Toggle thinking mode |
| `Ctrl+Y` | Copy last reply |
| `Ctrl+L` | Copy your last message |
| `Ctrl+C` | Copy highlighted text |
| `Ctrl+B` | Show/hide the history sidebar |
| `Ctrl+Q` | Quit |

Click a chat in **History** (or use the arrow keys and press Enter) to open it.

**Copying text:** drag with the mouse over any message (yours or the model's) to highlight it. The highlighted text is copied to the clipboard as soon as you let go, so you can paste it anywhere with `Cmd+V`. Double-click copies a paragraph and triple-click copies a whole message. On macOS, copying uses `pbcopy`, so it also works in Terminal.app, which ignores the usual terminal clipboard escape code. Terminal.app doesn't pass `Cmd+C` through to apps, so use copy-on-highlight or `Ctrl+C` there. (To use Terminal's own selection instead, hold `Fn` while dragging.)

## Files

| File | Role |
|------|------|
| `config.py` | API key, models, request params, data paths |
| `nim_client.py` | Async SSE streaming client (port of `NVIDIAChatClient.swift`) |
| `storage.py` | `ChatMessage` / `ChatThread`, JSON persistence, token estimate |
| `app.py` | Textual UI |
| `app.tcss` | Styles |

## Behavior (same as iOS)

- Models: Nemotron 3 Ultra 550B A55B (default) and Nemotron 3 Super 120B A12B, picked with `Ctrl+O`. (Step 3.7 Flash and DeepSeek V4 Flash from the iOS app are omitted; NVIDIA has retired them.)
- `temperature=1`, `top_p=0.95`, `max_tokens=16384`, `stream=true`.
- Thinking off: sends the "instant mode" system prompt plus `chat_template_kwargs: {"enable_thinking": false}`. If the API returns HTTP 422, the request is retried once without `chat_template_kwargs`.
- Chat titles come from the first user message (up to 36 characters).
- Soft limit of about 200,000 tokens per chat, estimated as 0.75 × words.
- Chats and settings are saved in `~/.openchat_tui/` (`threads.json`, `settings.json`). Set `OPENCHAT_TUI_HOME` to use a different folder.
