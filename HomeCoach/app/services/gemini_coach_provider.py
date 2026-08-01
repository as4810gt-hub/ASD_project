"""Gemini Interactions API adapter for multimodal parent coaching.

Only a trusted catalog image and the privacy-filtered ContextBuilder payload
leave the local process. Camera frames, raw audio and diagnostic class scores
are never accepted by this provider.
"""

from __future__ import annotations

import base64
import json
import socket
import ssl
import threading
from pathlib import Path
from urllib import error, request

import certifi

from ..materials import allowed_filenames, get_material
from .ollama_coach_provider import OllamaCoachProvider


class GeminiCoachProvider(OllamaCoachProvider):
    """Generate structured coaching copy with Gemini and safe fallbacks."""

    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    API_REVISION = "2026-05-20"
    VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}

    def __init__(
        self,
        *,
        api_key,
        model,
        base_url,
        stimuli_dir,
        timeout_seconds=15,
        enabled=True,
        thinking_level="low",
        max_image_bytes=3 * 1024 * 1024,
        fallback_provider=None,
    ):
        self.api_key = str(api_key or "").strip()
        self.requested = bool(enabled)
        self.gemini_enabled = self.requested and bool(self.api_key)
        self.fallback_provider = fallback_provider
        self.stimuli_dir = (
            Path(stimuli_dir).expanduser().resolve(strict=False)
            if stimuli_dir
            else None
        )
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.ssl_context = self._verified_ssl_context()
        normalized_thinking = str(thinking_level or "low").strip().lower()
        self.thinking_level = (
            normalized_thinking
            if normalized_thinking in self.VALID_THINKING_LEVELS
            else "low"
        )
        self._state_lock = threading.Lock()
        self._last_error = None
        self._has_succeeded = False

        fallback_enabled = bool(
            fallback_provider
            and getattr(fallback_provider, "enabled", False)
        )
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            enabled=self.gemini_enabled or fallback_enabled,
        )

    def generate(self, context, fallback):
        # High-risk disclosures never wait for, or leave the device for, a
        # generative model. The deterministic safety card is immediate.
        if self._context_requires_safety(context):
            return self._fallback(self._safety_fallback(context, fallback))
        enriched_fallback = self._contextual_fallback(context, fallback)
        if not self.gemini_enabled:
            return self._generate_fallback(context, enriched_fallback)

        try:
            payload = self._build_payload(context)
            response = self._request_interaction(payload)
            suggestion = json.loads(self._extract_output_text(response))
            normalized = self._normalize(
                suggestion,
                enriched_fallback,
                context,
                source="gemini",
                model=self.model,
            )
            if normalized.get("source") != "gemini":
                with self._state_lock:
                    self._last_error = "unsafe_or_invalid_output"
                # Relationship repair fails closed to deterministic relational
                # copy. Delegating this particular rejection with the original
                # context could let a weaker fallback model pull the parent
                # back to the picture again, especially when Gemini was the
                # component that noticed a rule-engine miss.
                if normalized.get("response_mode") == "repair_connection":
                    return normalized
                return self._generate_fallback(context, enriched_fallback)
            with self._state_lock:
                self._last_error = None
                self._has_succeeded = True
            return normalized
        except (
            OSError,
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            with self._state_lock:
                self._last_error = self._safe_error_name(exc)
            return self._generate_fallback(context, enriched_fallback)

    def health(self):
        if not self.requested:
            status = "disabled"
        elif not self.api_key:
            status = "key_missing"
        with self._state_lock:
            last_error = self._last_error
            has_succeeded = self._has_succeeded

        if self.requested and self.api_key:
            if last_error:
                status = "degraded"
            elif has_succeeded:
                status = "ready"
            else:
                status = "configured"

        fallback_health = None
        if self.fallback_provider is not None:
            fallback_health_method = getattr(
                self.fallback_provider,
                "health",
                None,
            )
            fallback_health = (
                fallback_health_method()
                if callable(fallback_health_method)
                else {"status": "unknown"}
            )

        return {
            "status": status,
            "provider": "gemini",
            "model": self.model,
            "configured": self.gemini_enabled,
            "vision": self.stimuli_dir is not None,
            "thinking_level": self.thinking_level,
            "has_succeeded": has_succeeded,
            "last_error": last_error,
            "fallback": fallback_health,
        }

    def _build_payload(self, context):
        input_blocks = [
            {
                "type": "text",
                "text": (
                    "請閱讀目前教材圖片與以下 JSON 情境，依規則產生這一輪的"
                    "家長教練提示：\n"
                    + json.dumps(context, ensure_ascii=False)
                ),
            }
        ]
        image = self._trusted_material_image(context)
        if image is not None:
            input_blocks.append(
                {
                    "type": "image",
                    "data": image,
                    "mime_type": "image/png",
                }
            )

        return {
            "model": self.model,
            "store": False,
            "system_instruction": self.SYSTEM_PROMPT,
            "input": input_blocks,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": self.OUTPUT_SCHEMA,
            },
            "generation_config": {
                "thinking_level": self.thinking_level,
            },
        }

    def _request_interaction(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/interactions",
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "Api-Revision": self.API_REVISION,
            },
            method="POST",
        )
        with request.urlopen(
            http_request,
            timeout=self.timeout_seconds,
            context=self.ssl_context,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _verified_ssl_context():
        """Use system CAs when present, otherwise certifi's verified bundle."""

        system_context = ssl.create_default_context()
        if system_context.get_ca_certs():
            return system_context
        return ssl.create_default_context(cafile=certifi.where())

    def _trusted_material_image(self, context):
        if self.stimuli_dir is None:
            return None
        material_profile = context.get("material_profile") or {}
        material_id = str(material_profile.get("id") or "").strip()
        if not material_id:
            return None
        material = get_material(material_id)
        if material is None:
            return None

        filename = str(material.get("filename") or "")
        if filename not in allowed_filenames() or not filename.endswith(".png"):
            return None

        try:
            root = self.stimuli_dir.resolve(strict=True)
            candidate = (root / filename).resolve(strict=True)
            if not candidate.is_relative_to(root) or not candidate.is_file():
                return None
            if candidate.stat().st_size > self.max_image_bytes:
                return None
            with candidate.open("rb") as image_file:
                image_bytes = image_file.read(self.max_image_bytes + 1)
        except (OSError, RuntimeError):
            return None

        if (
            len(image_bytes) > self.max_image_bytes
            or not image_bytes.startswith(self.PNG_SIGNATURE)
        ):
            return None
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _extract_output_text(response):
        for step in reversed(response.get("steps") or []):
            if step.get("type") != "model_output":
                continue
            text_parts = [
                str(block.get("text") or "")
                for block in step.get("content") or []
                if block.get("type") == "text" and block.get("text")
            ]
            if text_parts:
                return "".join(text_parts)
        raise ValueError("Gemini response did not contain model text")

    def _generate_fallback(self, context, fallback):
        if self._context_requires_safety(context):
            return self._fallback(self._safety_fallback(context, fallback))
        if (
            self.fallback_provider is not None
            and self.fallback_provider is not self
        ):
            delegated = self.fallback_provider.generate(
                context=context,
                fallback=fallback,
            )
            if self._context_requires_repair(context):
                if not isinstance(delegated, dict):
                    return self._fallback(self._repair_fallback(context, fallback))
                return self._normalize(
                    delegated,
                    self._repair_fallback(context, fallback),
                    context,
                    source=str(delegated.get("source") or "fallback_provider"),
                    model=str(delegated.get("model") or self.model),
                )
            return delegated
        return self._fallback(fallback)

    @staticmethod
    def _safe_error_name(exc):
        if isinstance(exc, error.HTTPError):
            return f"http_{exc.code}"
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, error.URLError):
            reason = getattr(exc, "reason", None)
            reason_text = str(reason or "").lower()
            if isinstance(reason, socket.gaierror):
                return "dns_unavailable"
            if isinstance(reason, ssl.SSLCertVerificationError):
                return "tls_verification_failed"
            if (
                isinstance(reason, (TimeoutError, socket.timeout))
                or "timed out" in reason_text
            ):
                return "timeout"
            if isinstance(reason, ConnectionRefusedError):
                return "connection_refused"
            return "network_unavailable"
        return exc.__class__.__name__.lower()
