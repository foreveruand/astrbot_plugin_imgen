"""
AstrBot Image Generation Plugin - Persona-based chat and image generation.
"""

import asyncio
import base64
import json
import os
import re
import time
import uuid
from pathlib import Path

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
MAX_ERROR_TEXT_LENGTH = 220


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
        "auto": "auto",
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1536x1024": "16:9",
        "1024x1792": "9:16",
        "1024x1536": "9:16",
        "1280x720": "16:9",
        "720x1280": "9:16",
        "2048x2048": "1:1",
        "2048x1152": "16:9",
        "1152x2048": "9:16",
        "3840x2160": "16:9",
        "2160x3840": "9:16",
    }
    return size_map.get(size, "1:1")


def convert_size_to_resolution(size: str | None) -> str:
    """Map the shared size setting to provider 1k/2k resolution options."""
    normalized = (size or "").strip().lower()
    if normalized == "auto":
        return "1k"
    match = re.fullmatch(r"(\d+)x(\d+)", normalized)
    if not match:
        return "1k"
    return "2k" if max(int(match.group(1)), int(match.group(2))) > 1536 else "1k"


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
    if image_data.startswith("base64://"):
        return image_data.removeprefix("base64://")
    if image_data.startswith("data:") and "," in image_data:
        return image_data.split(",", 1)[1]
    return image_data


def _normalize_base64_payload(image_data: str) -> str:
    """Normalize provider base64 output for AstrBot message components."""
    payload = _extract_base64_payload(image_data)
    payload = re.sub(r"\s+", "", payload).strip()
    return payload + ("=" * (-len(payload) % 4))


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
                    if part_type == "text" and isinstance(part.get("text"), str):
                        results.extend(
                            _extract_image_results_from_text_content(part.get("text"))
                        )
                    continue
                image_obj = part.get("image_url") or {}
                if isinstance(image_obj, dict):
                    image_value = image_obj.get("url")
                    if image_value:
                        results.append(image_value)
        elif isinstance(content, str):
            results.extend(_extract_image_results_from_text_content(content))

    if not results:
        # Fallback for non-standard compatible responses:
        # scan the full JSON text for inline image payloads.
        payload_text = json.dumps(payload, ensure_ascii=False)
        results.extend(_extract_image_results_from_text_content(payload_text))

    if results:
        # Keep order while removing duplicates
        return list(dict.fromkeys(results))
    return results


def _payload_summary(payload: dict) -> str:
    """Build a compact payload summary for logs/errors without inline large data."""
    keys = sorted(payload.keys())
    choices = payload.get("choices", [])
    data = payload.get("data", [])
    return (
        f"keys={keys}, choices={len(choices) if isinstance(choices, list) else 0}, "
        f"data={len(data) if isinstance(data, list) else 0}"
    )


def _sanitize_error_text(value: str, max_length: int = MAX_ERROR_TEXT_LENGTH) -> str:
    """Redact inline image payloads and truncate long errors."""
    text = str(value)
    text = re.sub(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
        "data:image/...;base64,[omitted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[A-Za-z0-9+/]{512,}={0,2}\b",
        "[base64 omitted]",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        return f"{text[:max_length]}..."
    return text


def _extract_image_results_from_text_content(content: str) -> list[str]:
    """Extract image payloads from string content returned by some compatible APIs."""
    results: list[str] = []

    # Markdown image syntax: ![alt](http://... or data:image/...)
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", content, flags=re.S):
        image_value = match.group(1).strip()
        if image_value.startswith("http") or image_value.startswith("data:image/"):
            results.append(image_value)

    # Tolerate incomplete markdown image syntax without a closing ')'
    for match in re.finditer(
        r"!\[[^\]]*\]\(\s*(data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/_=\-\s]+)",
        content,
        flags=re.IGNORECASE | re.S,
    ):
        image_value = re.sub(r"\s+", "", match.group(1))
        if image_value:
            results.append(image_value)

    # Direct data URL embedded in plain text
    for match in re.finditer(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/_=\-\s]+",
        content,
        flags=re.IGNORECASE,
    ):
        image_value = re.sub(r"\s+", "", match.group(0))
        results.append(image_value)

    # Third-party format: "![image]" followed by raw base64
    marker_match = re.search(r"!\[image\](.*)", content, flags=re.IGNORECASE | re.S)
    if marker_match:
        tail = marker_match.group(1).strip().strip("`")
        candidate = re.sub(r"\s+", "", tail).lstrip("(").rstrip(")")
        if candidate and candidate.startswith("data:image/"):
            results.append(candidate)
        elif (
            re.fullmatch(r"[A-Za-z0-9+/_=\-]+", candidate or "")
            and len(candidate) >= 128
        ):
            padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
            try:
                if "-" in padded or "_" in padded:
                    base64.urlsafe_b64decode(padded)
                else:
                    base64.b64decode(padded, validate=True)
                results.append(padded)
            except Exception:
                pass

    # Compatibility fallback for some OpenAI-compatible channels that return
    # plain URL text in `message.content` instead of structured image fields.
    plain_urls: list[str] = []
    for match in re.finditer(r'https?://[^\s)\]>"]+', content):
        candidate = match.group(0).strip().rstrip(".,;!?")
        if candidate:
            plain_urls.append(candidate)
    if plain_urls:
        image_like = [
            url
            for url in plain_urls
            if re.search(r"\.(png|jpe?g|webp|gif|bmp|avif)(?:$|[?#])", url, re.I)
        ]
        if image_like:
            results.extend(image_like)
        elif len(plain_urls) == 1:
            normalized = content.strip().strip("`").strip()
            if normalized.startswith("<") and normalized.endswith(">"):
                normalized = normalized[1:-1].strip()
            if normalized == plain_urls[0]:
                results.append(plain_urls[0])

    if results:
        return list(dict.fromkeys(results))
    return results


def _join_api_path(api_url: str, api_path: str) -> str:
    """Join an API base URL with a path without duplicating `/v1`."""
    base = api_url.rstrip("/")
    path = "/" + api_path.strip("/")
    return f"{base}{path}"


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


def _is_gpt_image_model(model: str | None) -> bool:
    """Detect GPT Image model family."""
    normalized = (model or "").strip().lower()
    return normalized.startswith("gpt-image-")


def _is_gpt_image_2_model(model: str | None) -> bool:
    """Detect GPT Image 2 model family."""
    normalized = (model or "").strip().lower()
    return normalized.startswith("gpt-image-2")


def _is_valid_gpt_image_2_size(size: str) -> bool:
    """Validate custom GPT Image 2 dimensions."""
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        return False

    width, height = int(match.group(1)), int(match.group(2))
    long_edge = max(width, height)
    short_edge = min(width, height)
    total_pixels = width * height

    return (
        long_edge <= 3840
        and width % 16 == 0
        and height % 16 == 0
        and long_edge / short_edge <= 3
        and 655_360 <= total_pixels <= 8_294_400
    )


def _normalize_openai_image_size(size: str | None, model: str | None) -> str:
    """Normalize configured sizes to values accepted by OpenAI's Images API."""
    normalized = (size or "").strip().lower() or "1024x1024"

    if _is_gpt_image_2_model(model):
        if normalized == "auto" or _is_valid_gpt_image_2_size(normalized):
            return normalized
        return "1024x1024"

    if _is_gpt_image_model(model):
        size_aliases = {
            "1792x1024": "1536x1024",
            "1024x1792": "1024x1536",
        }
        normalized = size_aliases.get(normalized, normalized)
        if normalized in {"1024x1024", "1536x1024", "1024x1536", "auto"}:
            return normalized
        return "1024x1024"

    if normalized in {"256x256", "512x512", "1024x1024"}:
        return normalized
    return "1024x1024"


def _build_openai_generate_payload(
    *,
    prompt: str,
    size: str,
    n: int,
    quality: str,
    background: str,
    output_format: str,
    model: str,
) -> dict:
    """Build a standard OpenAI Images API request body."""
    normalized_size = _normalize_openai_image_size(size, model)
    payload = {
        "model": model,
        "prompt": prompt,
        "n": max(1, int(n)),
        "size": normalized_size,
    }

    if quality and quality != "auto":
        payload["quality"] = quality

    if _is_gpt_image_model(model):
        if background and background != "auto":
            payload["background"] = background
        if output_format and output_format != "png":
            payload["output_format"] = output_format
    else:
        if output_format == "png":
            payload["response_format"] = "url"

    return payload


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

    def __init__(
        self,
        api_key: str,
        api_url: str,
        timeout: int = 120,
        use_chat_completions: bool = True,
    ):
        super().__init__(api_key=api_key, api_url=api_url, timeout=timeout)
        self._use_chat_completions = use_chat_completions

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
            safe_body = _sanitize_error_text(body, max_length=500)
            raise Exception(f"{error_prefix} {resp.status}: {safe_body}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            content_type = resp.headers.get("Content-Type", "unknown")
            safe_body = _sanitize_error_text(body, max_length=500)
            raise Exception(
                f"{error_prefix} returned non-JSON response "
                f"(status {resp.status}, content-type {content_type}): {safe_body}"
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
        if self._use_chat_completions:
            url = _join_api_path(self.api_url, "/v1/chat/completions")
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
                    data = await self._read_json_response(
                        resp, "OpenAI-compatible API error"
                    )
                    results = _extract_image_results(data)
                    if not results:
                        raise Exception(
                            "OpenAI-compatible API returned no image; "
                            f"response summary: {_payload_summary(data)}"
                        )
                    return results

        url = _join_api_path(self.api_url, "/v1/images/generations")
        payload = _build_openai_generate_payload(
            prompt=prompt,
            size=size,
            n=n,
            quality=quality,
            background=background,
            output_format=output_format,
            model=model,
        )

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
        if self._use_chat_completions:
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

            url = _join_api_path(self.api_url, "/v1/chat/completions")
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
                    data = await self._read_json_response(
                        resp, "OpenAI-compatible API error"
                    )
                    results = _extract_image_results(data)
                    if not results:
                        raise Exception(
                            "OpenAI-compatible API returned no image; "
                            f"response summary: {_payload_summary(data)}"
                        )
                    return results

        url = _join_api_path(self.api_url, "/v1/images/edits")
        normalized_size = _normalize_openai_image_size(size, model)

        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("model", model)
        form.add_field("size", normalized_size)
        if quality:
            form.add_field("quality", quality)
        if _is_gpt_image_model(model):
            if background:
                form.add_field("background", background)
            if output_format:
                form.add_field("output_format", output_format)

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
        image_size = convert_size_to_resolution(size).upper()

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        image_config_kwargs = {}
        if aspect_ratio not in {"1:1", "auto"}:
            image_config_kwargs["aspect_ratio"] = aspect_ratio
        if image_size != "1K":
            image_config_kwargs["image_size"] = image_size
        if image_config_kwargs:
            config.image_config = types.ImageConfig(**image_config_kwargs)

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
        image_size = convert_size_to_resolution(size).upper()

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

        image_config_kwargs = {}
        if aspect_ratio not in {"1:1", "auto"}:
            image_config_kwargs["aspect_ratio"] = aspect_ratio
        if image_size != "1K":
            image_config_kwargs["image_size"] = image_size
        if image_config_kwargs:
            config.image_config = types.ImageConfig(**image_config_kwargs)

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
    RUNNING_GENERATIONS: dict[str, asyncio.Task[None]] = {}  # chat_id -> task

    def __init__(self, context: star.Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config

    async def initialize(self) -> None:
        """Called when the plugin is activated."""
        logger.info("Image Generation plugin initialized")

    async def terminate(self) -> None:
        """Called when the plugin is disabled or reloaded."""
        for task in list(self.RUNNING_GENERATIONS.values()):
            task.cancel()
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

    def _config_get(self, section: str, key: str, default=None):
        """Read nested plugin config."""
        section_data = self.config.get(section, {})
        if isinstance(section_data, dict) and key in section_data:
            return section_data.get(key, default)
        return default

    def _general_config(self, key: str, default=None):
        return self._config_get("general_config", key, default)

    def _openai_config(self, key: str, default=None):
        return self._config_get("openai_config", key, default)

    def _gemini_config(self, key: str, default=None):
        return self._config_get("gemini_config", key, default)

    def _grok_config(self, key: str, default=None):
        return self._config_get("grok_config", key, default)

    def _provider_model(self, provider: str) -> str:
        """Return configured model for a provider."""
        if provider == "openai":
            return self._openai_config("model", "gpt-image-1")
        if provider == "gemini":
            return self._gemini_config("model", "imagen-3.0-generate-002")
        if provider == "grok":
            return self._grok_config("model", "grok-imagine-image")
        return ""

    def _openai_output_options(self) -> tuple[str, str, str]:
        """Return OpenAI Images API output options."""
        return (
            self._openai_config("quality", "auto"),
            self._openai_config("background", "auto"),
            self._openai_config("output_format", "png"),
        )

    def _is_provider_configured(self, provider: str) -> bool:
        """Check whether the current provider has the required credentials."""
        if provider == "gemini" and self._gemini_config("vertex_enabled", False):
            credentials_files = self._gemini_config("vertex_credentials", [])
            credentials_path = self._resolve_plugin_data_file(
                credentials_files[0] if credentials_files else None
            )
            return bool(
                credentials_path
                and os.path.isfile(credentials_path)
                and self._gemini_config("vertex_project", "").strip()
            )

        if provider == "openai":
            return bool(self._openai_config("api_key", ""))
        if provider == "gemini":
            return bool(self._gemini_config("api_key", ""))
        if provider == "grok":
            return bool(self._grok_config("api_key", ""))
        return False

    def _resolve_tool_provider(self, requested_provider: str | None) -> str:
        """Resolve the provider for tool calls with fallback to the configured default."""
        default_provider = self._general_config("default_provider", "openai")
        normalized_provider = (requested_provider or "").strip().lower()

        if normalized_provider not in {"openai", "gemini", "grok"}:
            return default_provider

        if self._is_provider_configured(normalized_provider):
            return normalized_provider

        logger.info(
            "Requested provider %s is not fully configured; falling back to default provider %s.",
            normalized_provider,
            default_provider,
        )
        return default_provider

    async def _apply_default_persona_prompt(self, prompt: str) -> str:
        """Prefix the prompt with the configured default persona when available."""
        default_persona = self._general_config("default_persona", "")
        if not default_persona:
            return prompt

        persona = await self.context.persona_manager.get_persona(default_persona)
        if not persona or not getattr(persona, "system_prompt", ""):
            return prompt

        system_prompt = str(persona.system_prompt).strip()
        if not system_prompt:
            return prompt
        return f"{system_prompt}\n\n{prompt}" if prompt else system_prompt

    async def _handle_inline_image_generation(self, event: AstrMessageEvent):
        """Generate a single image for a chosen Telegram inline query."""
        prompt = str(getattr(event, "query", "") or event.message_str or "").strip()
        if not prompt:
            yield event.plain_result("请先输入绘图描述，再选择图像生成助手。")
            return

        provider = self._general_config("default_provider", "openai")
        if not self._is_provider_configured(provider):
            yield event.plain_result(ERROR_MESSAGES["no_api_key"])
            return

        try:
            adapter = self._get_adapter(provider)
        except ValueError:
            yield event.plain_result(
                ERROR_MESSAGES["invalid_provider"].format(provider=provider)
            )
            return

        prompt = await self._apply_default_persona_prompt(prompt)

        try:
            image_size = self._general_config("default_size", "1024x1024")
            model = self._provider_model(provider)
            aspect_ratio = convert_size_to_aspect_ratio(image_size)
            grok_resolution = convert_size_to_resolution(image_size)
            openai_quality, openai_background, openai_output_format = (
                self._openai_output_options()
            )

            if provider == "gemini":
                result_urls = await adapter.generate(prompt, image_size, model=model)
            elif provider == "grok":
                result_urls = await adapter.generate(
                    prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=grok_resolution,
                    n=1,
                )
            else:
                result_urls = await adapter.generate(
                    prompt,
                    image_size,
                    n=1,
                    quality=openai_quality,
                    background=openai_background,
                    output_format=openai_output_format,
                    model=model,
                )

            if not result_urls:
                yield event.plain_result("图像生成失败：未返回结果。")
                return

            await self._send_image_output(event, result_urls[0])
        except Exception as e:
            safe_error = _sanitize_error_text(str(e))
            logger.error(f"Inline image generation error: {safe_error}")
            yield event.plain_result(
                ERROR_MESSAGES["api_error"].format(error=safe_error)
            )

    async def _send_image_output(
        self, event: AstrMessageEvent, image_data: str, mime_type: str = "image/png"
    ) -> None:
        """Send image output to the user, handling URLs and base64 payloads."""
        if image_data.startswith("http"):
            await event.send(event.image_result(image_data))
            return

        image_b64 = _normalize_base64_payload(image_data)
        await event.send(event.chain_result([Comp.Image.fromBase64(image_b64)]))

    def _get_adapter(self, provider: str) -> ImageAdapter:
        """Get the appropriate adapter for the provider."""
        timeout = self._general_config("timeout", 120)

        if provider == "openai":
            api_key = self._openai_config("api_key", "")
            api_url = self._openai_config("api_url", "https://api.openai.com")
            return OpenAIAdapter(
                api_key,
                api_url,
                timeout,
                self._openai_config("use_completions", True),
            )
        elif provider == "gemini":
            api_key = self._gemini_config("api_key", "")
            # Vertex AI configuration
            vertex_config = None
            if self._gemini_config("vertex_enabled", False):
                credentials_files = self._gemini_config("vertex_credentials", [])
                credentials_path = self._resolve_plugin_data_file(
                    credentials_files[0] if credentials_files else None
                )
                vertex_config = {
                    "enabled": True,
                    "credentials_path": credentials_path,
                    "project": self._gemini_config("vertex_project", ""),
                    "location": self._gemini_config("vertex_location", "us-central1"),
                }
            return GeminiAdapter(api_key, timeout, vertex_config=vertex_config)
        elif provider == "grok":
            api_key = self._grok_config("api_key", "")
            return GrokAdapter(api_key, "https://api.x.ai", timeout)
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

    async def _run_generation_task(
        self,
        event: AstrMessageEvent,
        chat_id: str,
        session_data: dict,
        *,
        size: str | None = None,
        n: int = 1,
    ) -> None:
        """Run generation in background so the session waiter can exit immediately."""
        try:
            await self._do_generate(event, chat_id, session_data, size=size, n=n)
        except asyncio.CancelledError:
            logger.info("Cancelled image generation task for chat %s", chat_id)
            raise
        finally:
            current_task = asyncio.current_task()
            if self.RUNNING_GENERATIONS.get(chat_id) is current_task:
                self.RUNNING_GENERATIONS.pop(chat_id, None)

    def _start_generation_task(
        self,
        event: AstrMessageEvent,
        chat_id: str,
        session_data: dict,
        *,
        size: str | None = None,
        n: int = 1,
    ) -> bool:
        """Start a background generation task if the chat is currently idle."""
        running_task = self.RUNNING_GENERATIONS.get(chat_id)
        if running_task and not running_task.done():
            return False

        session_snapshot = {
            "id": session_data.get("id"),
            "text": session_data.get("text", ""),
            "images": list(session_data.get("images", [])),
        }
        self.RUNNING_GENERATIONS[chat_id] = asyncio.create_task(
            self._run_generation_task(
                event,
                chat_id,
                session_snapshot,
                size=size,
                n=n,
            ),
            name=f"imgen-generate-{chat_id}",
        )
        return True

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

        provider = self._general_config("default_provider", "openai")

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

        prompt = await self._apply_default_persona_prompt(prompt)

        try:
            image_size = size or self._general_config("default_size", "1024x1024")

            # Get provider-specific model
            model = self._provider_model(provider)

            # Get provider-specific settings
            aspect_ratio = convert_size_to_aspect_ratio(image_size)
            grok_aspect_ratio = aspect_ratio
            grok_resolution = convert_size_to_resolution(image_size)
            openai_quality, openai_background, openai_output_format = (
                self._openai_output_options()
            )

            enable_multi_turn = self._general_config("enable_multi_turn", True)

            if images:
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
                        result_urls = await adapter.edit(
                            prompt,
                            images=processed_images,
                            size=image_size,
                            quality=openai_quality,
                            background=openai_background,
                            output_format=openai_output_format,
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
                        result_urls = await adapter.generate(
                            prompt,
                            image_size,
                            n=n,
                            quality=openai_quality,
                            background=openai_background,
                            output_format=openai_output_format,
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
                    result_urls = await adapter.generate(
                        prompt,
                        image_size,
                        n=n,
                        quality=openai_quality,
                        background=openai_background,
                        output_format=openai_output_format,
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
                    image_data=first_result
                    if is_url
                    else _normalize_base64_payload(first_result),
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    is_url=is_url,
                    mime_type="image/png",
                )

            await event.send(event.plain_result(ERROR_MESSAGES["generation_success"]))
            return True

        except Exception as e:
            safe_error = _sanitize_error_text(str(e))
            logger.error(f"Image generation error: {safe_error}")
            await event.send(
                event.plain_result(ERROR_MESSAGES["api_error"].format(error=safe_error))
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

    @filter.inline_query()
    async def inline_query_entry(self, event: AstrMessageEvent):
        """Expose this plugin as a Telegram inline-query option.

        The event bus only needs the handler registration so it can surface a
        plugin-specific choice in the inline result picker. Actual image
        generation runs after the user chooses the plugin result.
        """
        if not hasattr(event, "inline_message_id") or not hasattr(event, "result_id"):
            return

        async for result in self._handle_inline_image_generation(event):
            yield result

    @filter.command("img")
    async def img_cmd(self, event: AstrMessageEvent, initial_prompt: str = ""):
        """开始图像生成会话，收集文本和图片直到 /generate 或超时。"""
        chat_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )

        running_task = self.RUNNING_GENERATIONS.get(chat_id)
        if running_task and not running_task.done():
            yield event.plain_result(ERROR_MESSAGES["session_conflict"])
            return

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
        session_timeout = self._general_config("session_timeout", 300)
        msg += f"⏰ 会话将在 {session_timeout // 60} 分钟后自动超时。"
        yield event.plain_result(msg)

        # Session waiter
        @session_waiter(timeout=session_timeout, record_history_chains=False)
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

                started = self._start_generation_task(
                    event,
                    chat_id,
                    session_data,
                    size=gen_size,
                    n=gen_n,
                )
                if not started:
                    await event.send(
                        event.plain_result(ERROR_MESSAGES["session_conflict"])
                    )
                    controller.keep(timeout=session_timeout)
                    return

                if chat_id in self.ACTIVE_SESSIONS:
                    del self.ACTIVE_SESSIONS[chat_id]
                await event.send(
                    event.plain_result(
                        "已开始生成。当前绘图会话已关闭，机器人会继续响应其他命令；如需中止本次请求，请发送 /cancel。"
                    )
                )
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
                controller.keep(timeout=session_timeout)
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

            controller.keep(timeout=session_timeout)

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
        running_task = self.RUNNING_GENERATIONS.get(chat_id)
        if running_task and not running_task.done():
            running_task.cancel()
            self.RUNNING_GENERATIONS.pop(chat_id, None)
            yield event.plain_result("已取消当前绘图任务。")
            return

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
                    image_bytes = base64.b64decode(
                        _normalize_base64_payload(match.group(2))
                    )
                    return image_bytes, mime_type
                else:
                    logger.warning("Invalid base64 data URL format")
                    return None

            # 检查是否是本地文件路径
            image_path = Path(image_input)
            if image_path.is_file():
                with image_path.open("rb") as f:
                    image_bytes = f.read()
                mime_type = detect_mime_type(image_bytes)
                return image_bytes, mime_type

            # 认为是 HTTP/HTTPS URL
            if image_input.startswith("http://") or image_input.startswith("https://"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_input) as resp:
                        if resp.status != 200:
                            logger.info(
                                "Skip image input URL because download failed (HTTP %s): %s",
                                resp.status,
                                image_input,
                            )
                            return None
                        image_bytes = await resp.read()
                        mime_type = detect_mime_type(image_bytes)
                        return image_bytes, mime_type

            # 尝试作为 base64 字符串处理（无 data: 前缀）
            try:
                image_bytes = base64.b64decode(_normalize_base64_payload(image_input))
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
        size: str | None = None,
        provider: str | None = None,
    ) -> str:
        """生成或编辑图像。当用户请求生成图片、画图、创建图像或编辑图片时使用此工具。

        Args:
            prompt(string): 图像生成的详细描述，描述要生成的图像内容。
            images(array[string]): 可选。要编辑的图片列表，支持URL、本地路径、base64。如果提供，将对图片进行编辑；否则生成新图片。
            size(string): 可选。图像尺寸，如 1024x1024、1536x1024、1024x1536；默认使用插件设置中的 default_size。
            provider(string): 可选。指定使用的图像提供商，可选 openai、gemini、grok；若该提供商未完整配置则回退到 default_provider。
        """
        images_info = f"{len(images)} images" if images else "None"
        resolved_size = size or self._general_config("default_size", "1024x1024")
        resolved_provider = self._resolve_tool_provider(provider)
        logger.info(
            "generate_image_tool called: prompt=%s, images=%s, size=%s, provider=%s, requested_provider=%s",
            prompt,
            images_info,
            resolved_size,
            resolved_provider,
            provider,
        )

        if not self._is_provider_configured(resolved_provider):
            if resolved_provider == "gemini" and self._gemini_config(
                "vertex_enabled", False
            ):
                return "错误：未配置可用的 Vertex AI 凭证或项目 ID，请检查上传的 JSON 文件和 Vertex 配置。"
            return f"错误：未配置 {resolved_provider} 的 API 密钥，请在插件设置中配置。"

        try:
            adapter = self._get_adapter(resolved_provider)
        except ValueError as e:
            return f"错误：{str(e)}"

        try:
            model = self._provider_model(resolved_provider)
            aspect_ratio = convert_size_to_aspect_ratio(resolved_size)
            grok_aspect_ratio = aspect_ratio
            grok_resolution = convert_size_to_resolution(resolved_size)
            openai_quality, openai_background, openai_output_format = (
                self._openai_output_options()
            )

            # Process images list if provided
            processed_images = []
            if images:
                if not isinstance(images, list):
                    images = [images]
                for img_input in images:
                    result = await self._process_image_input(img_input)
                    if result:
                        processed_images.append(result)
                if not processed_images:
                    logger.info(
                        "No valid image inputs after preprocessing (%d provided); fallback to text-to-image generation.",
                        len(images),
                    )

            # Check if we're doing image-to-image (editing) or text-to-image
            is_editing = len(processed_images) > 0

            if is_editing:
                # Image-to-image editing - pass all processed images
                logger.info(
                    "Performing image-to-image editing for provider %s with %d image(s)",
                    resolved_provider,
                    len(processed_images),
                )

                if resolved_provider == "grok":
                    result_urls = await adapter.edit(
                        prompt,
                        images=processed_images,
                        model=model,
                        aspect_ratio=grok_aspect_ratio,
                    )
                elif resolved_provider == "gemini":
                    result_urls = await adapter.edit(
                        prompt, images=processed_images, size=resolved_size, model=model
                    )
                else:  # openai
                    result_urls = await adapter.edit(
                        prompt,
                        images=processed_images,
                        size=resolved_size,
                        quality=openai_quality,
                        background=openai_background,
                        output_format=openai_output_format,
                        model=model,
                    )
            else:
                # Text-to-image generation
                logger.info(
                    "Performing text-to-image generation for provider %s",
                    resolved_provider,
                )

                if resolved_provider == "gemini":
                    result_urls = await adapter.generate(
                        prompt, resolved_size, model=model
                    )
                elif resolved_provider == "grok":
                    result_urls = await adapter.generate(
                        prompt,
                        model=model,
                        aspect_ratio=grok_aspect_ratio,
                        resolution=grok_resolution,
                    )
                else:  # openai
                    result_urls = await adapter.generate(
                        prompt,
                        resolved_size,
                        quality=openai_quality,
                        background=openai_background,
                        output_format=openai_output_format,
                        model=model,
                    )

            if not result_urls:
                return "图像生成失败：未返回结果。"

            first_url = result_urls[0]
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
                # await self._send_image_output(event, first_url, mime_type)
            elif first_url.startswith("data:"):
                import re

                match = re.match(r"data:image/(\w+);base64,(.+)", first_url)
                if match:
                    mime_type = f"image/{match.group(1)}"
                    image_b64 = _normalize_base64_payload(match.group(2))
                else:
                    mime_type = "image/png"
                    image_b64 = _normalize_base64_payload(first_url)
                # Send to user
                # await self._send_image_output(event, first_url, mime_type)
            else:
                mime_type = "image/png"
                image_b64 = _normalize_base64_payload(first_url)
                # Send to user
                # await self._send_image_output(event, first_url, mime_type)

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
            safe_error = _sanitize_error_text(str(e))
            logger.error(f"Image generation tool error: {safe_error}")
            return f"图像生成失败：{safe_error}"
