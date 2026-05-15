"""Minimal LLM provider adapter for the astronomy PaperOrchestra layer.

Secrets are read from environment variables only. Do not hard-code API keys in
project files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ModelConfig:
    provider: str
    model: str
    reasoning_effort: str
    base_url: str
    api_key_env: str
    wire_api: str
    disable_response_storage: bool


PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent
REPO_ROOT = ASTRO_AGENT_DIR.parent


def load_dotenv_files(*paths: Path) -> list[str]:
    """Load simple KEY=VALUE .env files without echoing secrets."""
    loaded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(str(path))
    return loaded


def load_default_env() -> list[str]:
    kg_workspace = Path(os.getenv("ASTRO_AGENT_KG_WORKSPACE", str(REPO_ROOT / ".local_kg")))
    return load_dotenv_files(
        REPO_ROOT / ".env",
        PACKAGE_DIR / ".env",
        kg_workspace / ".env",
    )


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalise_model_name(provider: str, model: str) -> str:
    value = str(model or "").strip()
    if provider == "deepseek":
        aliases = {
            "deepseek-v4-pro": "deepseek-v4-pro",
            "deepseek-v4-flash": "deepseek-v4-flash",
            "deepseek v4 pro": "deepseek-v4-pro",
            "deepseek v4 flash": "deepseek-v4-flash",
            "deepseek-v4": "deepseek-v4-pro",
            "deepseekv4pro": "deepseek-v4-pro",
            "deepseekv4flash": "deepseek-v4-flash",
        }
        key = value.lower().replace("_", "-")
        return aliases.get(key, key)
    return value


def load_model_config(provider: Optional[str] = None) -> ModelConfig:
    load_default_env()
    selected = provider or os.getenv("ASTRO_AGENT_MODEL_PROVIDER", "fox")
    selected = selected.lower().strip()
    model = os.getenv("ASTRO_AGENT_MODEL", "gpt-5.5")
    effort = os.getenv("ASTRO_AGENT_REASONING_EFFORT", "high")
    disable_storage = _truthy(os.getenv("ASTRO_AGENT_DISABLE_RESPONSE_STORAGE", "true"))

    if selected in {"deepseek", "dp", "deepseek_v4", "deepseek-v4"}:
        deepseek_key_env = "DEEPSEEK_API_KEY"
        if not os.getenv(deepseek_key_env):
            deepseek_key_env = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"
        return ModelConfig(
            provider="deepseek",
            model=_normalise_model_name(
                "deepseek",
                os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-v4-pro"))),
            ),
            reasoning_effort=effort,
            base_url=os.getenv("DEEPSEEK_BASE_URL", os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))),
            api_key_env=deepseek_key_env,
            wire_api=os.getenv("DEEPSEEK_WIRE_API", "chat_completions"),
            disable_response_storage=disable_storage,
        )
    if selected in {"kimi", "moonshot", "moonshot_kimi", "moonshot-kimi"}:
        kimi_key_env = "KIMI_API_KEY"
        if not os.getenv(kimi_key_env):
            kimi_key_env = "MOONSHOT_API_KEY" if os.getenv("MOONSHOT_API_KEY") else "KIMI_API_KEY"
        return ModelConfig(
            provider="kimi",
            model=os.getenv("KIMI_MODEL", os.getenv("MOONSHOT_MODEL", "kimi-k2-latest")),
            reasoning_effort=effort,
            base_url=os.getenv("KIMI_BASE_URL", os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")),
            api_key_env=kimi_key_env,
            wire_api=os.getenv("KIMI_WIRE_API", os.getenv("MOONSHOT_WIRE_API", "chat_completions")),
            disable_response_storage=disable_storage,
        )
    if selected == "fox":
        return ModelConfig(
            provider="fox",
            model=model,
            reasoning_effort=effort,
            base_url=os.getenv("FOX_BASE_URL", "https://dm-fox.rjj.cc/codex/v1"),
            api_key_env="OPENAI_API_KEY",
            wire_api=os.getenv("FOX_WIRE_API", "responses"),
            disable_response_storage=disable_storage,
        )
    if selected in {"gemini", "google_gemini", "google"}:
        gemini_key_env = "GEMINI_API_KEY"
        if not os.getenv(gemini_key_env):
            gemini_key_env = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"
        return ModelConfig(
            provider="gemini",
            model=os.getenv(
                "GEMINI_MODEL",
                os.getenv("GOOGLE_GEMINI_MODEL", os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gemini-3-pro-preview"))),
            ),
            reasoning_effort=effort,
            base_url=os.getenv(
                "GOOGLE_GEMINI_BASE_URL",
                os.getenv("GEMINI_BASE_URL", os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com"))),
            ),
            api_key_env=gemini_key_env,
            wire_api=os.getenv("GEMINI_WIRE_API", "chat_completions"),
            disable_response_storage=disable_storage,
        )
    if selected in {"claude", "anthropic"}:
        claude_key_env = "ANTHROPIC_API_KEY"
        if not os.getenv(claude_key_env):
            # User may have set the Claude relay key under LLM_API_KEY.
            claude_key_env = "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "ANTHROPIC_API_KEY"
        return ModelConfig(
            provider="claude",
            model=os.getenv(
                "CLAUDE_MODEL",
                os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            ),
            reasoning_effort=effort,
            base_url=os.getenv(
                "CLAUDE_BASE_URL",
                os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            ),
            api_key_env=claude_key_env,
            wire_api=os.getenv("CLAUDE_WIRE_API", "anthropic_messages"),
            disable_response_storage=disable_storage,
        )
    raise ValueError(f"Unsupported model provider: {selected}")


class LLMClient:
    """Small wrapper around OpenAI-compatible Responses/Chat APIs."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()

    @property
    def available(self) -> bool:
        return bool(os.getenv(self.config.api_key_env))

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        cache_system: bool = True,
    ) -> str:
        """Run an LLM completion.

        cache_system: if True and the provider is Anthropic, the system prompt
            (which is static across all sections of the same paper run) is
            wrapped with ``cache_control: {"type": "ephemeral"}``. According
            to Anthropic's 2026 docs this gives up to 90% input-token cost
            reduction and 85% latency reduction for stable prefixes.
            DeepSeek/OpenAI providers ignore the flag silently.
        """
        if not self.available:
            raise RuntimeError(f"Missing API key environment variable: {self.config.api_key_env}")
        if self.config.wire_api == "anthropic_messages":
            return self._anthropic_messages(system, user, temperature, max_output_tokens, cache_system=cache_system)
        if self.config.wire_api == "responses":
            return self._responses(system, user, temperature, max_output_tokens, cache_system=cache_system)
        return self._chat_completions(system, user, temperature, max_output_tokens, cache_system=cache_system)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ[self.config.api_key_env]}",
            "Content-Type": "application/json",
        }

    def _responses(
        self,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        cache_system: bool = True,
    ) -> str:
        system_content: Any = system
        if cache_system and "anthropic" in self.config.base_url.lower():
            # Anthropic-style: cache the static system prefix to drop cost.
            system_content = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": self.config.reasoning_effort},
            "max_output_tokens": max_output_tokens,
        }
        if self.config.disable_response_storage:
            payload["store"] = False
        if temperature is not None:
            payload["temperature"] = temperature
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/responses",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("output_text"):
            return str(data["output_text"])
        chunks: List[str] = []
        for item in data.get("output", []):
            for part in item.get("content", []):
                if "text" in part:
                    chunks.append(str(part["text"]))
        return "\n".join(chunks).strip()

    def _anthropic_messages(
        self,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        cache_system: bool = True,
    ) -> str:
        """Native Anthropic /v1/messages API. Used when provider=claude."""
        system_block: Any = system
        if cache_system:
            system_block = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "system": system_block,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "x-api-key": os.environ[self.config.api_key_env],
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
        }
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/v1/messages",
            headers=headers,
            data=json.dumps(payload),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        chunks: List[str] = []
        for block in data.get("content", []):
            if block.get("type") == "text" and "text" in block:
                chunks.append(str(block["text"]))
        return "\n".join(chunks).strip()

    def _chat_completions(
        self,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        cache_system: bool = True,
    ) -> str:
        if cache_system and "anthropic" in self.config.base_url.lower():
            system_content: Any = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
            messages = [{"role": "system", "content": system_content},
                        {"role": "user", "content": user}]
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()
