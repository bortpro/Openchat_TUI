"""Streaming client for NVIDIA NIM chat completions (port of NVIDIAChatClient.swift)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

import config
from storage import ChatMessage


class NIMError(Exception):
    pass


def _chat_template_kwargs(thinking: bool) -> dict | None:
    if not thinking:
        return {"enable_thinking": False}
    return None


def _request_body(
    model: config.ChatModel,
    messages: list[ChatMessage],
    thinking: bool,
    include_kwargs: bool,
) -> dict:
    payload: list[dict] = []
    if not thinking:
        payload.append({"role": "system", "content": config.INSTANT_MODE_SYSTEM_PROMPT})
    for m in messages:
        if m.role == "system":
            continue
        text = m.content
        if m.role == "assistant" and m.reasoning:
            text = m.reasoning + "\n\n" + m.content
        payload.append({"role": m.role, "content": text})

    body = {
        "model": model.id,
        "messages": payload,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "max_tokens": config.MAX_TOKENS,
        "stream": True,
    }
    kwargs = _chat_template_kwargs(thinking)
    if include_kwargs and kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    return body


async def stream_completion(
    api_key: str,
    model: config.ChatModel,
    messages: list[ChatMessage],
    thinking: bool,
) -> AsyncIterator[tuple[str | None, str | None]]:
    """Yields (reasoning_delta, content_delta) pairs as they arrive."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }
    include_kwargs = _chat_template_kwargs(thinking) is not None
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            body = _request_body(model, messages, thinking, include_kwargs)
            async with client.stream("POST", config.BASE_URL, headers=headers, json=body) as resp:
                if resp.status_code >= 300:
                    err = (await resp.aread()).decode("utf-8", "replace")
                    if resp.status_code == 422 and include_kwargs:
                        # Retry once without chat_template_kwargs.
                        include_kwargs = False
                        continue
                    raise NIMError(f"Request failed ({resp.status_code}): {err}")

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta") or {}
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    content = delta.get("content")
                    if reasoning or content:
                        yield reasoning, content
                return
