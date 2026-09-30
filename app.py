"""openchat TUI — a terminal ChatGPT-style client for NVIDIA NIM models."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from datetime import datetime

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option
from textual.worker import WorkerCancelled

import config
import storage
from nim_client import stream_completion
from storage import ChatMessage, ChatThread


# --------------------------------------------------------------------------- widgets


class UserBubble(Static):
    def __init__(self, message: ChatMessage) -> None:
        super().__init__(Text(message.content))
        self.border_title = "You"


class AssistantBubble(Vertical):
    def __init__(self, message: ChatMessage) -> None:
        super().__init__()
        self.message = message
        self.border_title = "Assistant"
        self.reasoning = Static(Text(message.reasoning), classes="reasoning")
        self.reasoning.display = bool(message.reasoning)
        self.markdown = Markdown(message.content)

    def compose(self) -> ComposeResult:
        yield self.reasoning
        yield self.markdown

    def set_reasoning(self, text: str) -> None:
        self.reasoning.display = bool(text)
        self.reasoning.update(Text(text))


class HistoryItem(ListItem):
    def __init__(self, thread: ChatThread) -> None:
        self.thread_id = thread.id
        when = datetime.fromisoformat(thread.updated_at).astimezone()
        super().__init__(
            Label(Text(thread.title, overflow="ellipsis", no_wrap=True), classes="title"),
            Label(when.strftime("%b %d, %H:%M"), classes="date"),
        )


class Composer(TextArea):
    """Multiline input: Enter sends, Ctrl+J / Shift+Enter inserts a newline."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
        elif event.key in ("ctrl+j", "shift+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
        else:
            await super()._on_key(event)


class ModelPicker(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, current: str) -> None:
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        options = [
            Option(("● " if m.id == self.current else "  ") + m.title, id=m.id)
            for m in config.MODELS
        ]
        with Vertical(classes="dialog"):
            yield Label("Select model", classes="dialog-title")
            yield OptionList(*options)

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        ol.highlighted = next(i for i, m in enumerate(config.MODELS) if m.id == self.current)

    @on(OptionList.OptionSelected)
    def picked(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class ConfirmDelete(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "dismiss(True)", "Delete"),
        Binding("n,escape", "dismiss(False)", "Cancel"),
    ]

    def __init__(self, title: str) -> None:
        super().__init__()
        self.chat_title = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(f"Delete “{self.chat_title}”?"), classes="dialog-title")
            with Horizontal(classes="buttons"):
                yield Button("Delete (y)", variant="error", id="yes")
                yield Button("Cancel (n)", id="no")

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


# --------------------------------------------------------------------------- app


class OpenChatApp(App):
    TITLE = "openchat"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+n", "new_chat", "New chat", priority=True),
        Binding("ctrl+d", "delete_chat", "Delete chat", priority=True),
        Binding("ctrl+o", "pick_model", "Model", priority=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking", priority=True),
        Binding("ctrl+y", "copy_reply", "Copy reply", priority=True),
        Binding("ctrl+l", "copy_prompt", "Copy my msg", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        Binding("escape", "stop", "Stop", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = storage.load_settings()
        self.threads = storage.load_threads()
        if not self.threads:
            self.threads.append(ChatThread())
        self.current: ChatThread = max(self.threads, key=lambda t: t.updated_at)
        self.streaming = False
        self._stream_worker = None
        self._selection_timer = None
        self._pbcopy = shutil.which("pbcopy") if sys.platform == "darwin" else None

    # ----- layout

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Button("+ New chat", id="new-chat", variant="primary")
                yield Label("History", id="history-label")
                yield ListView(id="history")
            with Vertical(id="main"):
                yield VerticalScroll(id="messages")
                yield Static(id="status")
                yield Composer(id="composer", soft_wrap=True, show_line_numbers=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#composer", Composer).border_title = "Message (Enter to send, Ctrl+J for newline)"
        await self.refresh_history()
        await self.show_thread()
        self.query_one("#composer").focus()
        if not config.api_key():
            self.notify(
                "No NVIDIA API key. Set NVIDIA_API_KEY in openchat_tui/config.py "
                "or export NVIDIA_API_KEY.",
                severity="warning",
                timeout=10,
            )

    # ----- helpers

    def persist(self) -> None:
        storage.save_threads(self.threads)

    def copy_to_clipboard(self, text: str) -> None:
        # Textual copies via OSC 52, which macOS Terminal.app ignores; pbcopy works everywhere on macOS.
        super().copy_to_clipboard(text)
        if self._pbcopy:
            try:
                subprocess.run([self._pbcopy], input=text.encode("utf-8"), check=False, timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass

    def copy_selection(self) -> None:
        self._selection_timer = None
        text = (self.screen.get_selected_text() or "").strip("\n")
        if text:
            self.copy_to_clipboard(text)
            self.notify("Copied selection to clipboard.", timeout=2)

    def schedule_selection_copy(self) -> None:
        # Debounced so a double- then triple-click copies (and notifies) once.
        if self._selection_timer is not None:
            self._selection_timer.stop()
        self._selection_timer = self.set_timer(0.2, self.copy_selection)

    def update_status(self) -> None:
        s = self.settings
        thinking = "thinking on" if s.thinking_enabled else "thinking off"
        draft = self.query_one("#composer", Composer).text
        tokens = storage.thread_tokens(self.current, draft)
        parts = [s.model.title, thinking, f"~{tokens:,} / {config.THREAD_TOKEN_LIMIT:,} est. tokens"]
        if not config.api_key():
            parts.append("[b red]no API key[/]")
        if self.streaming:
            parts.append("[b yellow]generating… (Esc to stop)[/]")
        self.query_one("#status", Static).update(" · ".join(parts))

    async def refresh_history(self) -> None:
        lv = self.query_one("#history", ListView)
        ordered = sorted(self.threads, key=lambda t: t.updated_at, reverse=True)
        await lv.clear()
        await lv.extend(HistoryItem(t) for t in ordered)
        lv.index = next(i for i, t in enumerate(ordered) if t.id == self.current.id)

    async def show_thread(self) -> None:
        container = self.query_one("#messages", VerticalScroll)
        await container.remove_children()
        widgets = [
            UserBubble(m) if m.role == "user" else AssistantBubble(m)
            for m in self.current.messages
            if m.role in ("user", "assistant")
        ]
        if widgets:
            await container.mount_all(widgets)
        else:
            await container.mount(
                Static(
                    f"Start a conversation with [b]{self.settings.model.title}[/].",
                    classes="empty",
                )
            )
        container.scroll_end(animate=False)
        self.update_status()

    async def cancel_stream(self) -> None:
        if self._stream_worker is not None and self.streaming:
            self._stream_worker.cancel()
            try:
                await self._stream_worker.wait()  # let the worker's cleanup run
            except WorkerCancelled:
                pass
        self._stream_worker = None

    async def select_thread(self, thread: ChatThread) -> None:
        if thread is self.current:
            return
        await self.cancel_stream()
        self.current = thread
        await self.refresh_history()
        await self.show_thread()

    # ----- actions

    async def action_new_chat(self) -> None:
        await self.cancel_stream()
        if self.current.messages:
            # Reuse an existing empty chat rather than piling up blank ones.
            empty = next((t for t in self.threads if not t.messages), None)
            if empty is None:
                empty = ChatThread()
                self.threads.insert(0, empty)
            empty.touch()
            self.current = empty
            self.persist()
            await self.refresh_history()
            await self.show_thread()
        self.query_one("#composer").focus()

    def action_delete_chat(self) -> None:
        async def done(confirmed: bool | None) -> None:
            if not confirmed:
                return
            await self.cancel_stream()
            self.threads = [t for t in self.threads if t.id != self.current.id]
            if not self.threads:
                self.threads.append(ChatThread())
            self.current = max(self.threads, key=lambda t: t.updated_at)
            self.persist()
            await self.refresh_history()
            await self.show_thread()

        self.push_screen(ConfirmDelete(self.current.title), done)

    def action_pick_model(self) -> None:
        def done(model_id: str | None) -> None:
            if model_id:
                self.settings.model_id = model_id
                storage.save_settings(self.settings)
                self.update_status()
                self.notify(f"Model: {self.settings.model.title}")

        self.push_screen(ModelPicker(self.settings.model_id), done)

    def action_toggle_thinking(self) -> None:
        self.settings.thinking_enabled = not self.settings.thinking_enabled
        storage.save_settings(self.settings)
        self.update_status()
        self.notify("Thinking mode " + ("on" if self.settings.thinking_enabled else "off"))

    def action_copy_reply(self) -> None:
        last = next(
            (m for m in reversed(self.current.messages) if m.role == "assistant" and m.content),
            None,
        )
        if last is None:
            self.notify("Nothing to copy yet.")
            return
        text = f"{last.reasoning}\n\n{last.content}" if last.reasoning else last.content
        self.copy_to_clipboard(text)
        self.notify("Copied last reply to clipboard.")

    def action_copy_prompt(self) -> None:
        last = next(
            (m for m in reversed(self.current.messages) if m.role == "user" and m.content),
            None,
        )
        if last is None:
            self.notify("Nothing to copy yet.")
            return
        self.copy_to_clipboard(last.content)
        self.notify("Copied your last message to clipboard.")

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    async def action_stop(self) -> None:
        await self.cancel_stream()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "stop":
            return self.streaming
        return True

    # ----- events

    def on_text_selected(self, event: events.TextSelected) -> None:
        # Mouse drag released: copy whatever is highlighted, like a text editor.
        self.schedule_selection_copy()

    def on_click(self, event: events.Click) -> None:
        # Double-click selects a paragraph, triple-click the whole message.
        if event.chain >= 3 and event.widget is not None:
            # Textual's default triple-click selects the entire scroll area; narrow it to one bubble.
            bubble = next(
                (w for w in (event.widget, *event.widget.ancestors)
                 if isinstance(w, (UserBubble, AssistantBubble))),
                None,
            )
            if bubble is not None:
                bubble.text_select_all()
        if event.chain >= 2:
            self.schedule_selection_copy()

    @on(Button.Pressed, "#new-chat")
    async def new_chat_pressed(self) -> None:
        await self.action_new_chat()

    @on(ListView.Selected, "#history")
    async def history_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, HistoryItem):
            thread = next((t for t in self.threads if t.id == item.thread_id), None)
            if thread:
                await self.select_thread(thread)
                self.query_one("#composer").focus()

    @on(TextArea.Changed, "#composer")
    def draft_changed(self) -> None:
        self.update_status()

    @on(Composer.Submitted)
    async def send(self, event: Composer.Submitted) -> None:
        text = event.text.strip()
        if not text or self.streaming:
            return
        key = config.api_key()
        if not key:
            self.notify("Add your NVIDIA API key in openchat_tui/config.py first.", severity="error")
            return
        if storage.thread_tokens(self.current, text) > config.THREAD_TOKEN_LIMIT:
            self.notify("This chat is over the token budget. Start a new chat.", severity="error")
            return

        thread = self.current
        self.query_one("#composer", Composer).clear()
        thread.messages.append(ChatMessage(role="user", content=text))
        if thread.title == storage.NEW_CHAT_TITLE or not thread.title:
            thread.title = storage.title_preview(text)
        placeholder = ChatMessage(role="assistant")
        thread.messages.append(placeholder)
        thread.touch()
        self.persist()

        container = self.query_one("#messages", VerticalScroll)
        await container.query(".empty").remove()
        bubble = AssistantBubble(placeholder)
        await container.mount_all([UserBubble(thread.messages[-2]), bubble])
        container.scroll_end(animate=False)
        await self.refresh_history()

        self.streaming = True
        self.refresh_bindings()
        self.update_status()
        self._stream_worker = self.run_stream(key, thread, placeholder, bubble)

    @work(exclusive=True, group="stream")
    async def run_stream(
        self,
        key: str,
        thread: ChatThread,
        message: ChatMessage,
        bubble: AssistantBubble,
    ) -> None:
        container = self.query_one("#messages", VerticalScroll)
        md_stream = Markdown.get_stream(bubble.markdown)
        thinking = self.settings.thinking_enabled
        history = thread.messages[:-1]  # exclude the placeholder
        last_save = time.monotonic()
        try:
            async for reasoning, content in stream_completion(
                key, self.settings.model, history, thinking
            ):
                follow = container.max_scroll_y - container.scroll_y <= 2
                if reasoning and thinking:
                    message.reasoning += reasoning
                    bubble.set_reasoning(message.reasoning)
                if content:
                    message.content += content
                    await md_stream.write(content)
                if follow:
                    container.scroll_end(animate=False)
                if time.monotonic() - last_save > 1.0:
                    self.persist()
                    last_save = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # network / HTTP errors
            self.notify(str(e) or e.__class__.__name__, severity="error", timeout=15, markup=False)
        finally:
            await md_stream.stop()
            if not message.content and not message.reasoning:
                if message in thread.messages:
                    thread.messages.remove(message)
                await bubble.remove()
            thread.touch()
            self.persist()
            self.streaming = False
            self.refresh_bindings()
            self.update_status()
            container.scroll_end(animate=False)


def main() -> None:
    OpenChatApp().run()


if __name__ == "__main__":
    main()
