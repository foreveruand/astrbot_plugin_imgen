"""
AstrBot Image Generation Plugin - Persona-based chat and image generation.
"""

import base64
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import aiohttp
import mcp.types
from google import genai
from google.genai import types

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
from astrbot.core.utils.session_waiter import (
    SessionController,
    SessionFilter,
    session_waiter,
)

# Allowed image MIME types
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB


def detect_mime_type(data: bytes) -> str:
    """Detect image MIME type from bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


async def extract_images_from_event(event) -> list[tuple[bytes, str]]:
    """Extract all images from event message components."""
    images = []
    for comp in event.get_messages():
        if hasattr(comp, "convert_to_base64"):
            try:
                b64_data = await comp.convert_to_base64()
                if b64_data:
                    image_bytes = base64.b64decode(b64_data)
                    if len(image_bytes) > MAX_IMAGE_SIZE:
                        continue
                    mime_type = detect_mime_type(image_bytes)
                    if mime_type in ALLOWED_MIME_TYPES:
                        images.append((image_bytes, mime_type))
            except Exception:
                pass
    return images


def convert_size_to_aspect_ratio(size: str) -> str:
    """Convert pixel size to aspect ratio."""
    size_map = {
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
        "1280x720": "16:9",
        "720x1280": "9:16",
    }
    return size_map.get(size, "1:1")


# Chinese error messages
ERROR_MESSAGES = {
    "no_session": "当前没有正在进行的绘图会话，请先使用 /img 开始。",
    "no_prompt": "请提供绘图描述或发送图片后再使用 /generate。",
    "no_api_key": "未配置 API 密钥，请在插件设置中填写 api_key。",
    "api_error": "图像生成失败: {error}",
    "timeout": "请求超时，请稍后重试。",
    "invalid_provider": "无效的图像提供商: {provider}",
    "image_too_large": "图片大小超过限制（最大 20MB）。",
    "unsupported_format": "不支持的图片格式，仅支持 PNG、JPEG、WEBP、GIF。",
    "session_conflict": "当前会话已有正在进行的绘图任务，请先使用 /cancel 取消。",
    "generation_success": "✅ 图像生成成功！",
    "multi_turn_cleared": "✅ 已清除多轮编辑历史，可以开始新的创作。",
    "no_multi_turn_history": "当前没有多轮编辑历史。",
}


def _get_kv_key(chat_id: str) -> str:
    """Generate KV storage key for multi-turn image session."""
    return f"img_session_{chat_id}"


def _encode_image_result(image_data: bytes | str) -> str:
    """Normalize provider image results to plain base64 strings."""
    if isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    return image_data


def _extract_base64_payload(image_data: str) -> str:
    """Extract raw base64 payload from a data URL or plain base64 string."""
    if image_data.startswith("data:") and "," in image_data:
        return image_data.split(",", 1)[1]
    return image_data


def _extract_image_results(payload: dict) -> list[str]:
    """Extract image URLs or base64 payloads from an API response."""
    results = []
    for item in payload.get("data", []) or []:
        image_value = item.get("url") or item.get("b64_json")
        if image_value:
            results.append(image_value)

    if not results:
        image_value = payload.get("url") or payload.get("b64_json")
        if image_value:
            results.append(image_value)

    # OpenRouter / OpenAI chat-style image responses
    for choice in payload.get("choices", []) or []:
        message = (choice or {}).get("message") or {}

        for image_item in message.get("images", []) or []:
            if not isinstance(image_item, dict):
                continue
            image_obj = image_item.get("image_url") or image_item.get("imageUrl") or {}
            if isinstance(image_obj, dict):
                image_value = image_obj.get("url")
                if image_value:
                    results.append(image_value)

        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type not in {"output_image", "image_url"}:
                    continue
                image_obj = part.get("image_url") or {}
                if isinstance(image_obj, dict):
                    image_value = image_obj.get("url")
                    if image_value:
                        results.append(image_value)

    return results


def _is_openrouter_api_url(api_url: str) -> bool:
    """Detect whether the configured API URL targets OpenRouter."""
    return "openrouter.ai" in (urlparse(api_url).netloc or "").lower()


def _normalize_openrouter_api_url(api_url: str) -> str:
    """Normalize OpenRouter base URL so downstream calls can append `/v1/...` safely."""
    parsed = urlparse(api_url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    if not path:
        path = "/api"
    elif not path.startswith("/api"):
        path = f"/api{path}"

    return urlunparse(parsed._replace(path=path)).rstrip("/")


def _normalize_openrouter_model(model: str) -> str:
    """Normalize common OpenRouter model aliases/mistypes."""
    normalized = (model or "").strip()
    aliases = {
        "gpt-5.4-image2": "openai/gpt-5.4-image-2",
        "gpt-5.4-image-2": "openai/gpt-5.4-image-2",
        "openai/gpt-5.4-image2": "openai/gpt-5.4-image-2",
    }
    return aliases.get(normalized, normalized)


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Convert raw image bytes to `data:<mime>;base64,...`."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def _normalize_grok_resolution(resolution: str | None) -> str | None:
    """Map legacy pixel-style values to xAI's supported `1k`/`2k` resolution options."""
    if resolution is None:
        return None

    value = str(resolution).strip().lower()
    if not value:
        return None

    resolution_map = {
        "1k": "1k",
        "2k": "2k",
        "1024": "1k",
        "1024x1024": "1k",
        "2048": "2k",
        "2048x2048": "2k",
        "1792x1024": "2k",
        "1024x1792": "2k",
    }
    return resolution_map.get(value, value)


class ChatFilter(SessionFilter):
    """Session filter keyed by chat_id (chat-level scope)."""

    def filter(self, event: AstrMessageEvent) -> str:
        # Use group_id for group chats, unified_msg_origin for private
        return (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )


class ImageAdapter:
    """Base class for image generation adapters."""

    def __init__(self, api_key: str, api_url: str, timeout: int = 120):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self, prompt: str, size: str = "1024x1024", n: int = 1, **kwargs
    ) -> list[str]:
        raise NotImplementedError

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/png",
        size: str = "1024x1024",
        images: list[tuple[bytes, str]] | None = None,
        **kwargs,
    ) -> list[str]:
        """Edit image with optional multi-image support.

        Args:
            prompt: The edit prompt
            image_bytes: Single image bytes (for backward compatibility)
            mime_type: MIME type of the single image
            size: Target image size
            images: List of (image_bytes, mime_type) tuples for multi-image editing
            **kwargs: Additional provider-specific parameters
        """
        raise NotImplementedError


class OpenAIAdapter(ImageAdapter):
    """OpenAI image generation adapter for gpt-image-1 model."""

    def __init__(self, api_key: str, api_url: str, timeout: int = 120):
        super().__init__(api_key=api_key, api_url=api_url, timeout=timeout)
        self._is_openrouter = _is_openrouter_api_url(self.api_url)
        if self._is_openrouter:
            self.api_url = _normalize_openrouter_api_url(self.api_url)

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _read_json_response(
        self, resp: aiohttp.ClientResponse, error_prefix: str
    ):
        body = await resp.text()
        if resp.status != 200:
            raise Exception(f"{error_prefix} {resp.status}: {body}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            content_type = resp.headers.get("Content-Type", "unknown")
            raise Exception(
                f"{error_prefix} returned non-JSON response "
                f"(status {resp.status}, content-type {content_type}): {body[:500]}"
            ) from exc

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "auto",
        background: str = "auto",
        output_format: str = "png",
        model: str = "gpt-image-1",
    ) -> list[str]:
        """Generate image using an OpenAI-compatible image model."""
        if self._is_openrouter:
            url = f"{self.api_url}/v1/chat/completions"
            model = _normalize_openrouter_model(model)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["image", "text"],
                "stream": False,
                "image_config": {"aspect_ratio": convert_size_to_aspect_ratio(size)},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                ) as resp:
                    data = await self._read_json_response(resp, "OpenRouter API error")
                    results = _extract_image_results(data)
                    if not results:
                        raise Exception(f"OpenRouter API returned no image: {data}")
                    return results

        url = f"{self.api_url}/v1/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "n": max(1, int(n)),
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "response_format": "url",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self._get_headers(), timeout=self.timeout
            ) as resp:
                data = await self._read_json_response(resp, "OpenAI API error")
                return _extract_image_results(data)

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/png",
        size: str = "1024x1024",
        images: list[tuple[bytes, str]] | None = None,
        quality: str = "auto",
        background: str = "auto",
        output_format: str = "png",
        model: str = "gpt-image-1",
    ) -> list[str]:
        """Edit image(s) using an OpenAI-compatible image model. Supports multiple images."""
        if self._is_openrouter:
            if images and len(images) > 0:
                source_images = images
            elif image_bytes:
                source_images = [(image_bytes, mime_type)]
            else:
                raise ValueError("No image provided for editing")

            content: list[dict] = [{"type": "text", "text": prompt}]
            for img_bytes, img_mime in source_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _to_data_url(img_bytes, img_mime)},
                    }
                )

            url = f"{self.api_url}/v1/chat/completions"
            model = _normalize_openrouter_model(model)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "modalities": ["image", "text"],
                "stream": False,
                "image_config": {"aspect_ratio": convert_size_to_aspect_ratio(size)},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                ) as resp:
                    data = await self._read_json_response(resp, "OpenRouter API error")
                    results = _extract_image_results(data)
                    if not results:
                        raise Exception(f"OpenRouter API returned no image: {data}")
                    return results

        url = f"{self.api_url}/v1/images/edits"

        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("model", model)
        form.add_field("size", size)
        form.add_field("quality", quality)
        form.add_field("background", background)
        form.add_field("output_format", output_format)
        form.add_field("response_format", "url")

        # Handle multi-image input
        if images and len(images) > 0:
            # Multiple images provided
            for idx, (img_bytes, img_mime) in enumerate(images):
                filename = (
                    f"image_{idx}.png"
                    if img_mime == "image/png"
                    else f"image_{idx}.jpg"
                )
                form.add_field(
                    "image[]", img_bytes, filename=filename, content_type=img_mime
                )
        elif image_bytes:
            # Single image for backward compatibility
            filename = "image.png" if mime_type == "image/png" else "image.jpg"
            form.add_field(
                "image", image_bytes, filename=filename, content_type=mime_type
            )

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=form, headers=headers, timeout=self.timeout
            ) as resp:
                data = await self._read_json_response(resp, "OpenAI edit API error")
                return _extract_image_results(data)


class GeminiAdapter(ImageAdapter):
    """Gemini image generation adapter using official google-genai SDK."""

    def __init__(
        self, api_key: str, timeout: int = 120, vertex_config: dict | None = None
    ):
        # Note: api_url is not used for Gemini SDK - uses Google's official endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._using_vertex = False

        # Vertex AI configuration
        if vertex_config and vertex_config.get("enabled"):
            credentials_path = vertex_config.get("credentials_path")
            project = vertex_config.get("project")
            location = vertex_config.get("location", "us-central1")

            if credentials_path and project:
                # Load service account credentials
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self.client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    credentials=credentials,
                )
                self._using_vertex = True
                logger.info(
                    f"GeminiAdapter initialized with Vertex AI (project={project})"
                )
            else:
                # Fall back to API key if Vertex AI config is incomplete
                self.client = genai.Client(api_key=api_key)
                logger.info(
                    "GeminiAdapter initialized with API key (Vertex AI config incomplete)"
                )
        else:
            # Use API key
            self.client = genai.Client(api_key=api_key)
            logger.info("GeminiAdapter initialized with API key")

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        model: str = "imagen-3.0-generate-002",
    ) -> list[str]:
        """Generate image from text prompt using google-genai SDK."""
        aspect_ratio = convert_size_to_aspect_ratio(size)

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        if aspect_ratio != "1:1":
            config.image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            images = []
            for part in response.parts:
                if part.inline_data:
                    images.append(_encode_image_result(part.inline_data.data))
            return images

        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                raise Exception(f"Gemini API 密钥无效或未授权: {e}")
            elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
                raise Exception(f"Gemini API 配额不足或请求过于频繁: {e}")
            else:
                raise Exception(f"Gemini 图像生成失败: {e}")

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/png",
        size: str = "1024x1024",
        images: list[tuple[bytes, str]] | None = None,
        model: str = "imagen-3.0-generate-002",
    ) -> list[str]:
        """Edit image(s) with text prompt using google-genai SDK. Supports multiple images."""
        aspect_ratio = convert_size_to_aspect_ratio(size)

        # Build content parts - all images followed by text prompt
        content_parts = []

        if images and len(images) > 0:
            # Multiple images
            for img_bytes, img_mime in images:
                image_part = types.Part.from_bytes(data=img_bytes, mime_type=img_mime)
                content_parts.append(image_part)
        elif image_bytes:
            # Single image for backward compatibility
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            content_parts.append(image_part)

        # Add text prompt
        text_part = types.Part.from_text(text=prompt)
        content_parts.append(text_part)

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        if aspect_ratio != "1:1":
            config.image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=content_parts,
                config=config,
            )

            images_result = []
            for part in response.parts:
                if part.inline_data:
                    images_result.append(_encode_image_result(part.inline_data.data))
            return images_result

        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                raise Exception(f"Gemini API 密钥无效或未授权: {e}")
            elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
                raise Exception(f"Gemini API 配额不足或请求过于频繁: {e}")
            else:
                raise Exception(f"Gemini 图像编辑失败: {e}")


class GrokAdapter(ImageAdapter):
    """Grok adapter using xAI's OpenAI-compatible REST endpoints."""

    def __init__(
        self, api_key: str, api_url: str = "https://api.x.ai", timeout: int = 120
    ):
        super().__init__(api_key, api_url, timeout)

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        model: str = "grok-imagine-image",
        aspect_ratio: str = "1:1",
        resolution: str = "1k",
        n: int = 1,
    ) -> list[str]:
        """Generate image using xAI's OpenAI-compatible `/v1/images/generations` API."""
        url = f"{self.api_url}/v1/images/generations"
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "n": max(1, int(n)),
        }

        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio

        normalized_resolution = _normalize_grok_resolution(resolution)
        if normalized_resolution:
            payload["resolution"] = normalized_resolution

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(
                        f"Grok generate API error {resp.status}: {error_text}"
                    )
                data = await resp.json()

        results = _extract_image_results(data)
        if results:
            return results
        raise Exception(f"Grok generate API returned unexpected response: {data}")

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/png",
        size: str = "1024x1024",
        images: list[tuple[bytes, str]] | None = None,
        model: str = "grok-imagine-image",
        aspect_ratio: str | None = None,
    ) -> list[str]:
        """Edit image(s) using xAI's JSON-based `/v1/images/edits` API."""
        if images and len(images) > 0:
            source_images = images[:5]
            if len(images) > 5:
                logger.warning(
                    "Grok adapter received %s images; xAI only supports up to 5, extra images were ignored",
                    len(images),
                )
        elif image_bytes:
            source_images = [(image_bytes, mime_type)]
        else:
            raise ValueError("No image provided for editing")

        image_payloads = [
            {
                "type": "image_url",
                "url": f"data:{img_mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}",
            }
            for img_bytes, img_mime in source_images
        ]

        payload: dict[str, object] = {
            "prompt": prompt,
            "model": model,
        }
        if len(image_payloads) == 1:
            payload["image"] = image_payloads[0]
        else:
            payload["images"] = image_payloads
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/v1/images/edits",
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Grok edit API error {resp.status}: {error_text}")
                data = await resp.json()

        results = _extract_image_results(data)
        if results:
            return results
        raise Exception(f"Grok edit API returned unexpected response: {data}")


class Main(star.Star):
    """Main class for the Image Generation plugin."""

    # Track active image generation sessions by chat_id
    ACTIVE_SESSIONS: dict[str, dict] = {}  # chat_id -> session data

    def __init__(self, context: star.Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config

    async def initialize(self) -> None:
        """Called when the plugin is activated."""
        logger.info("Image Generation plugin initialized")

    async def terminate(self) -> None:
        """Called when the plugin is disabled or reloaded."""
        logger.info("Image Generation plugin terminated")

    def _get_plugin_data_dir(self) -> Path:
        """Return the plugin's persistent data directory."""
        return Path(get_astrbot_plugin_data_path()) / "astrbot_plugin_imgen"

    def _resolve_plugin_data_file(self, file_path: str | None) -> str | None:
        """Resolve a plugin-uploaded file path against astrbot_plugin_data_dir."""
        if not file_path:
            return None

        candidate = Path(file_path)
        if candidate.is_absolute():
            return str(candidate)

        return str((self._get_plugin_data_dir() / candidate).resolve(strict=False))

    def _is_provider_configured(self, provider: str) -> bool:
        """Check whether the current provider has the required credentials."""
        if provider == "gemini" and self.config.get("gemini_vertex_enabled"):
            credentials_files = self.config.get("gemini_vertex_credentials", [])
            credentials_path = self._resolve_plugin_data_file(
                credentials_files[0] if credentials_files else None
            )
            return bool(
                credentials_path
                and os.path.isfile(credentials_path)
                and self.config.get("gemini_vertex_project", "").strip()
            )

        api_key_map = {
            "openai": ("openai_api_key", "api_key"),
            "gemini": ("gemini_api_key", "api_key"),
            "grok": ("grok_api_key", "api_key"),
        }
        primary_key, fallback_key = api_key_map.get(provider, ("api_key", "api_key"))
        return bool(self.config.get(primary_key) or self.config.get(fallback_key))

    async def _send_image_output(
        self, event: AstrMessageEvent, image_data: str, mime_type: str = "image/png"
    ) -> None:
        """Send image output to the user, handling URLs and base64 payloads."""
        if image_data.startswith("http"):
            await event.send(event.image_result(image_data))
            return

        image_b64 = _extract_base64_payload(image_data)
        await event.send(event.chain_result([Comp.Image.fromBase64(image_b64)]))

    def _get_adapter(self, provider: str) -> ImageAdapter:
        """Get the appropriate adapter for the provider."""
        timeout = self.config.get("timeout", 120)

        if provider == "openai":
            api_key = self.config.get("openai_api_key") or self.config.get(
                "api_key", ""
            )
            api_url = self.config.get("openai_api_url") or self.config.get(
                "api_url", "https://api.openai.com"
            )
            return OpenAIAdapter(api_key, api_url, timeout)
        elif provider == "gemini":
            api_key = self.config.get("gemini_api_key") or self.config.get(
                "api_key", ""
            )
            # Vertex AI configuration
            vertex_config = None
            if self.config.get("gemini_vertex_enabled"):
                credentials_files = self.config.get("gemini_vertex_credentials", [])
                credentials_path = self._resolve_plugin_data_file(
                    credentials_files[0] if credentials_files else None
                )
                vertex_config = {
                    "enabled": True,
                    "credentials_path": credentials_path,
                    "project": self.config.get("gemini_vertex_project", ""),
                    "location": self.config.get(
                        "gemini_vertex_location", "us-central1"
                    ),
                }
            return GeminiAdapter(api_key, timeout, vertex_config=vertex_config)
        elif provider == "grok":
            api_key = self.config.get("grok_api_key") or self.config.get("api_key", "")
            api_url = self.config.get("grok_api_url") or self.config.get(
                "api_url", "https://api.x.ai"
            )
            return GrokAdapter(api_key, api_url, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _store_last_image(
        self,
        chat_id: str,
        image_data: str,
        provider: str,
        model: str,
        prompt: str,
        is_url: bool = False,
        mime_type: str = "image/png",
    ) -> None:
        """Store last generated image to KV storage for multi-turn editing."""
        key = _get_kv_key(chat_id)
        kv_data: dict[str, object] = await self.get_kv_data(key, None) or {
            "prompt_history": [],
        }

        # Update stored data
        kv_data["last_image"] = image_data
        kv_data["last_image_is_url"] = is_url
        kv_data["last_image_mime"] = mime_type
        kv_data["provider"] = provider
        kv_data["model"] = model
        kv_data["prompt_history"].append(prompt)
        kv_data["timestamp"] = int(time.time())

        await self.put_kv_data(key, kv_data)
        logger.debug(f"Stored last image for chat {chat_id}")

    async def _get_last_image(self, chat_id: str) -> dict | None:
        """Retrieve last generated image from KV storage."""
        key = _get_kv_key(chat_id)
        data = await self.get_kv_data(key, None)
        return data

    async def _clear_last_image(self, chat_id: str) -> None:
        """Clear multi-turn editing history from KV storage."""
        key = _get_kv_key(chat_id)
        await self.delete_kv_data(key)
        logger.debug(f"Cleared last image for chat {chat_id}")

    async def _do_generate(
        self,
        event: AstrMessageEvent,
        chat_id: str,
        session_data: dict,
        size: str | None = None,
        n: int = 1,
    ) -> bool:
        """Execute image generation. Returns True on success."""
        prompt = session_data.get("text", "")
        images = session_data.get("images", [])

        if not prompt and not images:
            await event.send(event.plain_result(ERROR_MESSAGES["no_prompt"]))
            return False

        provider = self.config.get("default_provider", "openai")

        if not self._is_provider_configured(provider):
            await event.send(event.plain_result(ERROR_MESSAGES["no_api_key"]))
            return False

        try:
            adapter = self._get_adapter(provider)
        except ValueError:
            await event.send(
                event.plain_result(
                    ERROR_MESSAGES["invalid_provider"].format(provider=provider)
                )
            )
            return False

        # Add persona prompt if configured
        default_persona = self.config.get("default_persona", "")
        if default_persona:
            persona = await self.context.persona_manager.get_persona(default_persona)
            if persona and hasattr(persona, "system_prompt") and persona.system_prompt:
                prompt = (
                    f"{persona.system_prompt}\n\n{prompt}"
                    if prompt
                    else persona.system_prompt
                )

        try:
            image_size = size or self.config.get("default_size", "1024x1024")

            # Get provider-specific model
            model_map = {
                "openai": self.config.get("openai_model", "gpt-image-1"),
                "gemini": self.config.get("gemini_model", "imagen-3.0-generate-002"),
                "grok": self.config.get("grok_model", "grok-imagine-image"),
            }
            model = model_map.get(provider, "")

            # Get provider-specific settings
            aspect_ratio = convert_size_to_aspect_ratio(image_size)
            grok_aspect_ratio = aspect_ratio
            configured_grok_aspect_ratio = self.config.get("grok_aspect_ratio", "")
            if not size and configured_grok_aspect_ratio:
                grok_aspect_ratio = configured_grok_aspect_ratio
            grok_resolution = self.config.get("grok_resolution", "1k")

            # Multi-turn editing: check for last_image in KV storage
            enable_multi_turn = self.config.get("enable_multi_turn", True)
            last_image_data = None
            if enable_multi_turn and not images:
                last_image_data = await self._get_last_image(chat_id)

            # Determine if we should use multi-turn editing
            use_multi_turn_edit = (
                enable_multi_turn
                and last_image_data is not None
                and last_image_data.get("last_image")
                and not images  # Don't use multi-turn if user provided new images
            )

            if use_multi_turn_edit:
                # Multi-turn editing using stored last_image
                assert last_image_data is not None
                stored_image = last_image_data["last_image"]
                is_url = last_image_data.get("last_image_is_url", False)
                stored_mime = last_image_data.get("last_image_mime", "image/png")

                logger.info(f"Using multi-turn editing for chat {chat_id}")

                if is_url:
                    image_bytes = None
                    mime_type = "image/png"
                    download_failed_reason = ""
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(stored_image) as resp:
                                if resp.status == 200:
                                    image_bytes = await resp.read()
                                    mime_type = detect_mime_type(image_bytes)
                                else:
                                    download_failed_reason = f"Failed to download stored image: {resp.status}"
                    except Exception as err:
                        download_failed_reason = (
                            f"Failed to download stored image: {err}"
                        )

                    if image_bytes is not None:
                        if provider == "grok":
                            result_urls = await adapter.edit(
                                prompt,
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                                model=model,
                                aspect_ratio=grok_aspect_ratio,
                            )
                        elif provider == "gemini":
                            result_urls = await adapter.edit(
                                prompt,
                                image_bytes,
                                mime_type,
                                image_size,
                                model=model,
                            )
                        else:  # openai
                            quality = self.config.get("openai_quality", "auto")
                            background = self.config.get("openai_background", "auto")
                            output_format = self.config.get(
                                "openai_output_format", "png"
                            )
                            result_urls = await adapter.edit(
                                prompt,
                                image_bytes,
                                mime_type,
                                image_size,
                                quality=quality,
                                background=background,
                                output_format=output_format,
                                model=model,
                            )
                    elif prompt.strip():
                        logger.warning(
                            f"{download_failed_reason}; fallback to prompt-only generation."
                        )
                        await self._clear_last_image(chat_id)
                        if provider == "gemini":
                            result_urls = await adapter.generate(
                                prompt, image_size, model=model
                            )
                        elif provider == "grok":
                            result_urls = await adapter.generate(
                                prompt,
                                model=model,
                                aspect_ratio=grok_aspect_ratio,
                                resolution=grok_resolution,
                                n=n,
                            )
                        else:  # openai
                            quality = self.config.get("openai_quality", "auto")
                            background = self.config.get("openai_background", "auto")
                            output_format = self.config.get(
                                "openai_output_format", "png"
                            )
                            result_urls = await adapter.generate(
                                prompt,
                                image_size,
                                n=n,
                                quality=quality,
                                background=background,
                                output_format=output_format,
                                model=model,
                            )
                    else:
                        raise Exception(download_failed_reason)
                else:
                    # Base64-based (OpenAI, Gemini)
                    image_bytes = base64.b64decode(stored_image)
                    mime_type = stored_mime
                    if provider == "gemini":
                        result_urls = await adapter.edit(
                            prompt, image_bytes, mime_type, image_size, model=model
                        )
                    elif provider == "grok":
                        # Grok now uses bytes directly
                        image_bytes = base64.b64decode(stored_image)
                        result_urls = await adapter.edit(
                            prompt,
                            image_bytes=image_bytes,
                            mime_type=mime_type,
                            model=model,
                            aspect_ratio=grok_aspect_ratio,
                        )
                    else:  # openai
                        quality = self.config.get("openai_quality", "auto")
                        background = self.config.get("openai_background", "auto")
                        output_format = self.config.get("openai_output_format", "png")
                        result_urls = await adapter.edit(
                            prompt,
                            image_bytes,
                            mime_type,
                            image_size,
                            quality=quality,
                            background=background,
                            output_format=output_format,
                            model=model,
                        )
            elif images:
                # Image-to-image
                processed_images = []
                for img_comp in images:
                    if not hasattr(img_comp, "convert_to_base64"):
                        continue
                    b64_data = await img_comp.convert_to_base64()
                    if not b64_data:
                        continue
                    image_bytes = base64.b64decode(b64_data)
                    processed_images.append(
                        (image_bytes, detect_mime_type(image_bytes))
                    )

                if processed_images:
                    if provider == "gemini":
                        result_urls = await adapter.edit(
                            prompt,
                            images=processed_images,
                            size=image_size,
                            model=model,
                        )
                    elif provider == "grok":
                        result_urls = await adapter.edit(
                            prompt,
                            images=processed_images,
                            model=model,
                            aspect_ratio=grok_aspect_ratio,
                        )
                    else:  # openai
                        quality = self.config.get("openai_quality", "auto")
                        background = self.config.get("openai_background", "auto")
                        output_format = self.config.get("openai_output_format", "png")
                        result_urls = await adapter.edit(
                            prompt,
                            images=processed_images,
                            size=image_size,
                            quality=quality,
                            background=background,
                            output_format=output_format,
                            model=model,
                        )
                else:
                    if provider == "gemini":
                        result_urls = await adapter.generate(
                            prompt, image_size, model=model
                        )
                    elif provider == "grok":
                        result_urls = await adapter.generate(
                            prompt,
                            model=model,
                            aspect_ratio=grok_aspect_ratio,
                            resolution=grok_resolution,
                            n=n,
                        )
                    else:  # openai
                        quality = self.config.get("openai_quality", "auto")
                        background = self.config.get("openai_background", "auto")
                        output_format = self.config.get("openai_output_format", "png")
                        result_urls = await adapter.generate(
                            prompt,
                            image_size,
                            n=n,
                            quality=quality,
                            background=background,
                            output_format=output_format,
                            model=model,
                        )
            else:
                # Text-to-image
                if provider == "gemini":
                    result_urls = await adapter.generate(
                        prompt, image_size, model=model
                    )
                elif provider == "grok":
                    result_urls = await adapter.generate(
                        prompt,
                        model=model,
                        aspect_ratio=grok_aspect_ratio,
                        resolution=grok_resolution,
                        n=n,
                    )
                else:  # openai
                    quality = self.config.get("openai_quality", "auto")
                    background = self.config.get("openai_background", "auto")
                    output_format = self.config.get("openai_output_format", "png")
                    result_urls = await adapter.generate(
                        prompt,
                        image_size,
                        n=n,
                        quality=quality,
                        background=background,
                        output_format=output_format,
                        model=model,
                    )

            # Send results
            first_result = result_urls[0] if result_urls else None
            for url in result_urls:
                await self._send_image_output(event, url)

            # Store result for multi-turn editing if enabled
            if enable_multi_turn and first_result:
                is_url = first_result.startswith("http")
                await self._store_last_image(
                    chat_id=chat_id,
                    image_data=first_result if is_url else first_result,
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    is_url=is_url,
                    mime_type="image/png",
                )

            await event.send(event.plain_result(ERROR_MESSAGES["generation_success"]))
            return True

        except Exception as e:
            logger.error(f"Image generation error: {e}")
            await event.send(
                event.plain_result(ERROR_MESSAGES["api_error"].format(error=str(e)))
            )
            return False

    @filter.command("task")
    async def task_cmd(
        self, event: AstrMessageEvent, persona_name: str = "", prompt: str = ""
    ):
        """使用指定人格进行无记忆对话。

        用法: /task <人格名> <提示词>
        """
        if not persona_name or not prompt:
            yield event.plain_result(
                "用法: /task <人格名> <提示词>\n示例: /task default 你好"
            )
            return

        # Get persona
        persona_mgr = self.context.persona_manager
        persona = await persona_mgr.get_persona(persona_name)

        if not persona:
            # List available personas
            all_personas = await persona_mgr.get_all_personas()
            persona_list = (
                "\n".join([f"  - {p.id}" for p in all_personas])
                if all_personas
                else "  (无)"
            )
            yield event.plain_result(
                f"未找到人格 '{persona_name}'。\n可用人格:\n{persona_list}"
            )
            return

        # Get system prompt from persona
        system_prompt = (
            persona.system_prompt if hasattr(persona, "system_prompt") else ""
        )

        # Call LLM without history (memoryless)
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                ),
                prompt=prompt,
                system_prompt=system_prompt,
                contexts=[],  # Empty contexts = memoryless
            )
            yield event.plain_result(llm_resp.completion_text)
        except Exception as e:
            logger.error(f"Task command error: {e}")
            yield event.plain_result(f"对话失败: {e}")

    @filter.command("img")
    async def img_cmd(self, event: AstrMessageEvent, initial_prompt: str = ""):
        """开始图像生成会话，收集文本和图片直到 /generate 或超时。"""
        chat_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )

        # Check for existing session
        if chat_id in self.ACTIVE_SESSIONS:
            yield event.plain_result(
                "当前会话已有正在进行的绘图任务，请先使用 /cancel 取消或等待超时。"
            )
            return

        # Initialize session
        session_id = str(uuid.uuid4())[:8]
        self.ACTIVE_SESSIONS[chat_id] = {
            "id": session_id,
            "text": initial_prompt,
            "images": [],
        }

        # Send confirmation
        msg = "🎨 绘图会话已开始！\n"
        if initial_prompt:
            msg += f"已记录描述: {initial_prompt}\n"
        msg += "请发送图片或文字描述，完成后使用 /generate 生成，或 /cancel 取消。\n"
        msg += f"⏰ 会话将在 {self.config.get('session_timeout', 300) // 60} 分钟后自动超时。"
        yield event.plain_result(msg)

        # Session waiter
        @session_waiter(
            timeout=self.config.get("session_timeout", 300), record_history_chains=False
        )
        async def img_waiter(controller: SessionController, event: AstrMessageEvent):
            # Refresh session data
            session_data = self.ACTIVE_SESSIONS.get(chat_id, {})
            if not session_data:
                controller.stop()
                return

            # Check for generate/cancel commands
            # Note: wake_prefix (/) has been stripped by AstrBot pipeline
            msg_str = event.message_str.strip().lower()

            if msg_str.startswith("generate"):
                # Parse optional arguments: generate [n] [size]
                # e.g., "generate 2 1024x1024" or "generate 1024x1024" or "generate 2"
                parts = msg_str.split()
                gen_n = 1
                gen_size = None

                for part in parts[1:]:  # Skip "generate" itself
                    if part.isdigit():
                        gen_n = int(part)
                    elif "x" in part:  # Size format like "1024x1024"
                        gen_size = part

                await self._do_generate(
                    event, chat_id, session_data, size=gen_size, n=gen_n
                )
                if chat_id in self.ACTIVE_SESSIONS:
                    del self.ACTIVE_SESSIONS[chat_id]
                controller.stop()
                return

            if msg_str == "cancel":
                await self._clear_last_image(chat_id)
                await event.send(event.plain_result("已取消绘图会话。"))
                if chat_id in self.ACTIVE_SESSIONS:
                    del self.ACTIVE_SESSIONS[chat_id]
                controller.stop()
                return

            if msg_str == "clear":
                last_image_data = await self._get_last_image(chat_id)
                if last_image_data and last_image_data.get("last_image"):
                    await self._clear_last_image(chat_id)
                    await event.send(
                        event.plain_result(ERROR_MESSAGES["multi_turn_cleared"])
                    )
                else:
                    await event.send(
                        event.plain_result(ERROR_MESSAGES["no_multi_turn_history"])
                    )
                controller.keep(timeout=self.config.get("session_timeout", 300))
                return

            # Collect text
            if event.message_str:
                if self.ACTIVE_SESSIONS.get(chat_id, {}).get("text"):
                    self.ACTIVE_SESSIONS[chat_id]["text"] += " " + event.message_str
                else:
                    self.ACTIVE_SESSIONS[chat_id]["text"] = event.message_str

            # Collect images
            for comp in event.get_messages():
                if isinstance(comp, Comp.Image):
                    self.ACTIVE_SESSIONS[chat_id]["images"].append(comp)

            # Send confirmation
            session_data = self.ACTIVE_SESSIONS.get(chat_id, {})
            text_preview = (
                session_data.get("text", "")[:50] + "..."
                if session_data.get("text")
                else "无"
            )
            img_count = len(session_data.get("images", []))
            await event.send(
                event.plain_result(f"已记录: 文字({text_preview}) 图片({img_count}张)")
            )

            controller.keep(timeout=self.config.get("session_timeout", 300))

        try:
            await img_waiter(event, session_filter=ChatFilter())
        except TimeoutError:
            # Clear KV storage on timeout
            await self._clear_last_image(chat_id)
            yield event.plain_result("⏰ 绘图会话已超时，请重新开始。")
        finally:
            if chat_id in self.ACTIVE_SESSIONS:
                del self.ACTIVE_SESSIONS[chat_id]
            event.stop_event()

    @filter.command("cancel")
    async def cancel_cmd(self, event: AstrMessageEvent):
        """取消当前绘图会话。"""
        chat_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )
        # Clear KV storage for multi-turn editing
        await self._clear_last_image(chat_id)
        if chat_id in self.ACTIVE_SESSIONS:
            del self.ACTIVE_SESSIONS[chat_id]
            yield event.plain_result("已取消绘图会话。")
        else:
            yield event.plain_result("当前没有正在进行的绘图会话。")

    @filter.command("clear")
    async def clear_cmd(self, event: AstrMessageEvent):
        """清除多轮编辑历史，保留当前绘图会话。"""
        chat_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )
        last_image_data = await self._get_last_image(chat_id)
        if last_image_data and last_image_data.get("last_image"):
            await self._clear_last_image(chat_id)
            yield event.plain_result(ERROR_MESSAGES["multi_turn_cleared"])
        else:
            yield event.plain_result(ERROR_MESSAGES["no_multi_turn_history"])

    async def _process_image_input(
        self, image_input: str | bytes
    ) -> tuple[bytes, str] | None:
        """处理单个图片输入，支持URL、本地路径、base64和bytes。

        Args:
            image_input: 图片输入，可以是URL字符串、本地路径、base64 data URL或bytes

        Returns:
            tuple: (image_bytes, mime_type) 或 None 如果处理失败
        """
        try:
            if isinstance(image_input, bytes):
                # 直接是bytes
                image_bytes = image_input
                mime_type = detect_mime_type(image_bytes)
                return image_bytes, mime_type

            if not isinstance(image_input, str):
                logger.warning(f"Unsupported image input type: {type(image_input)}")
                return None

            image_input = image_input.strip()
            if not image_input:
                return None

            # 检查是否是 base64 data URL
            if image_input.startswith("data:"):
                import re

                match = re.match(r"data:image/(\w+);base64,(.+)", image_input)
                if match:
                    mime_type = f"image/{match.group(1)}"
                    image_bytes = base64.b64decode(match.group(2))
                    return image_bytes, mime_type
                else:
                    logger.warning("Invalid base64 data URL format")
                    return None

            # 检查是否是本地文件路径
            import os

            if os.path.isfile(image_input):
                with open(image_input, "rb") as f:
                    image_bytes = f.read()
                mime_type = detect_mime_type(image_bytes)
                return image_bytes, mime_type

            # 认为是 HTTP/HTTPS URL
            if image_input.startswith("http://") or image_input.startswith("https://"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_input) as resp:
                        if resp.status != 200:
                            logger.warning(
                                f"Failed to download image from URL: {resp.status}"
                            )
                            return None
                        image_bytes = await resp.read()
                        mime_type = detect_mime_type(image_bytes)
                        return image_bytes, mime_type

            # 尝试作为 base64 字符串处理（无 data: 前缀）
            try:
                image_bytes = base64.b64decode(image_input)
                mime_type = detect_mime_type(image_bytes)
                return image_bytes, mime_type
            except Exception:
                logger.warning(f"Unable to process image input: {image_input[:100]}...")
                return None

        except Exception as e:
            logger.warning(f"Error processing image input: {e}")
            return None

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        images: list | None = None,
        size: str = "1024x1024",
    ) -> str:
        """生成或编辑图像。当用户请求生成图片、画图、创建图像或编辑图片时使用此工具。

        Args:
            prompt(string): 图像生成的详细描述，描述要生成的图像内容。
            images(array[string]): 可选。要编辑的图片列表，支持URL、本地路径、base64。如果提供，将对图片进行编辑；否则生成新图片。
            size(string): 图像尺寸，如 1024x1024、1792x1024、1024x1792。默认为 1024x1024。
        """
        images_info = f"{len(images)} images" if images else "None"
        logger.info(
            f"generate_image_tool called: prompt={prompt}, images={images_info}, size={size}"
        )

        # Use default provider from config
        provider = self.config.get("default_provider", "openai")

        if not self._is_provider_configured(provider):
            if provider == "gemini" and self.config.get("gemini_vertex_enabled"):
                return "错误：未配置可用的 Vertex AI 凭证或项目 ID，请检查上传的 JSON 文件和 Vertex 配置。"
            return f"错误：未配置 {provider} 的 API 密钥，请在插件设置中配置。"

        try:
            adapter = self._get_adapter(provider)
        except ValueError as e:
            return f"错误：{str(e)}"

        try:
            model_map = {
                "openai": self.config.get("openai_model", "gpt-image-1"),
                "gemini": self.config.get("gemini_model", "imagen-3.0-generate-002"),
                "grok": self.config.get("grok_model", "grok-imagine-image"),
            }
            model = model_map.get(provider, "")
            aspect_ratio = convert_size_to_aspect_ratio(size)
            grok_aspect_ratio = aspect_ratio
            configured_grok_aspect_ratio = self.config.get("grok_aspect_ratio", "")
            if configured_grok_aspect_ratio and size == self.config.get(
                "default_size", "1024x1024"
            ):
                grok_aspect_ratio = configured_grok_aspect_ratio
            grok_resolution = self.config.get("grok_resolution", "1k")

            # Process images list if provided
            processed_images = []
            if images:
                if not isinstance(images, list):
                    images = [images]
                for img_input in images:
                    result = await self._process_image_input(img_input)
                    if result:
                        processed_images.append(result)

            # Check if we're doing image-to-image (editing) or text-to-image
            is_editing = len(processed_images) > 0

            if is_editing:
                # Image-to-image editing - pass all processed images
                logger.info(
                    f"Performing image-to-image editing for provider {provider} with {len(processed_images)} image(s)"
                )

                if provider == "grok":
                    result_urls = await adapter.edit(
                        prompt,
                        images=processed_images,
                        model=model,
                        aspect_ratio=grok_aspect_ratio,
                    )
                elif provider == "gemini":
                    result_urls = await adapter.edit(
                        prompt, images=processed_images, size=size, model=model
                    )
                else:  # openai
                    quality = self.config.get("openai_quality", "auto")
                    background = self.config.get("openai_background", "auto")
                    output_format = self.config.get("openai_output_format", "png")
                    result_urls = await adapter.edit(
                        prompt,
                        images=processed_images,
                        size=size,
                        quality=quality,
                        background=background,
                        output_format=output_format,
                        model=model,
                    )
            else:
                # Text-to-image generation
                logger.info(
                    f"Performing text-to-image generation for provider {provider}"
                )

                if provider == "gemini":
                    result_urls = await adapter.generate(prompt, size, model=model)
                elif provider == "grok":
                    result_urls = await adapter.generate(
                        prompt,
                        model=model,
                        aspect_ratio=grok_aspect_ratio,
                        resolution=grok_resolution,
                    )
                else:  # openai
                    quality = self.config.get("openai_quality", "auto")
                    background = self.config.get("openai_background", "auto")
                    output_format = self.config.get("openai_output_format", "png")
                    result_urls = await adapter.generate(
                        prompt,
                        size,
                        quality=quality,
                        background=background,
                        output_format=output_format,
                        model=model,
                    )

            if not result_urls:
                return "图像生成失败：未返回结果。"

            first_url = result_urls[0]
            result_count = len(result_urls)
            action = "编辑" if is_editing else "生成"

            # Determine MIME type and base64 data
            if first_url.startswith("http"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(first_url) as resp:
                        if resp.status != 200:
                            return f"图像{action}成功，但无法获取图片数据。"
                        image_bytes = await resp.read()
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        mime_type = detect_mime_type(image_bytes)
                # Send to user
                await self._send_image_output(event, first_url, mime_type)
            elif first_url.startswith("data:"):
                import re

                match = re.match(r"data:image/(\w+);base64,(.+)", first_url)
                if match:
                    mime_type = f"image/{match.group(1)}"
                    image_b64 = match.group(2)
                else:
                    mime_type = "image/png"
                    image_b64 = first_url
                # Send to user
                await self._send_image_output(event, first_url, mime_type)
            else:
                mime_type = "image/png"
                image_b64 = first_url
                # Send to user
                await self._send_image_output(event, first_url, mime_type)

            # Return ImageContent so LLM can see the generated image
            # This allows the LLM to evaluate the result
            return mcp.types.CallToolResult(
                content=[
                    mcp.types.ImageContent(
                        type="image",
                        data=image_b64,
                        mimeType=mime_type,
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Image generation tool error: {e}")
            return f"图像生成失败：{str(e)}"
