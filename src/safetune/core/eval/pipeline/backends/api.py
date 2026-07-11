"""OpenAI-compatible HTTP API inference backend.

Targets any endpoint that speaks the OpenAI ``/v1/chat/completions`` schema:

* OpenAI proper (``api_base=https://api.openai.com/v1``)
* vLLM's OpenAI-compatible server (``api_base=http://localhost:8000/v1``)
* Anthropic via an OpenAI compatibility shim
* Together / Groq / Fireworks
* Any in-house gateway with the same shape

We avoid hard-depending on the ``openai`` SDK to keep the install slim;
all calls go through ``requests`` directly. If ``openai`` is installed we
defer to it for retry handling.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .base import GenerationConfig, InferenceBackend

logger = logging.getLogger(__name__)


class ApiBackend(InferenceBackend):
    """Hit an OpenAI-compatible chat-completions endpoint.

    Args:
        model: model id as understood by the endpoint
            (e.g. ``"gpt-4o-mini"``, ``"meta-llama/Llama-3.1-8B-Instruct"``).
        api_base: base URL ending in ``/v1``.
        api_key: bearer token. If ``None``, read from env var named in
            ``api_key_env``.
        api_key_env: env-var to look up when ``api_key`` is None.
        max_retries: per-request retry budget for transient 5xx / 429.
        timeout: seconds to wait per request.
        extra_headers: additional HTTP headers (e.g. organization id).
    """

    def __init__(
        self,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        config: Optional[GenerationConfig] = None,
        chat_template: bool = False,  # endpoint formats messages itself
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model=model,
            config=config,
            chat_template=chat_template,
            system_prompt=system_prompt,
        )
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"ApiBackend: api_key not provided and env var {api_key_env} is unset."
            )
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _payload(self, prompt: str) -> Dict[str, Any]:
        cfg = self.config
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt),
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
        }
        if cfg.temperature > 0:
            body["top_p"] = cfg.top_p
        if cfg.stop_sequences:
            body["stop"] = cfg.stop_sequences
        if cfg.seed is not None:
            body["seed"] = cfg.seed
        return body

    def _call_once(self, body: Dict[str, Any]) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        url = f"{self.api_base}/chat/completions"
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"ApiBackend HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"ApiBackend: malformed response: {data}") from e

    def _generate_one(self, prompt: str) -> str:
        body = self._payload(prompt)
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(body)
            except RuntimeError as e:
                if attempt >= self.max_retries:
                    raise
                msg = str(e)
                # Retry on 5xx and 429.
                if "HTTP 5" in msg or "HTTP 429" in msg:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise
        raise RuntimeError("ApiBackend: exhausted retries")

    def generate(self, prompts: List[str]) -> List[str]:
        out: List[str] = []
        for p in prompts:
            text = self._generate_one(p)
            out.append(self._truncate_on_stop(text))
        return out
