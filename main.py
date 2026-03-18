"""
AstrBot Image Generation Plugin - Persona-based chat and image generation.
"""

import base64
import mimetypes
import uuid
from typing import Any, List, Optional, Tuple

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.core.utils.session_waiter import SessionController, SessionFilter, session_waiter


# Allowed image MIME types
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB


def detect_mime_type(data: bytes) -> str:
    """Detect image MIME type from bytes."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    if data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    if data.startswith((b'GIF87a', b'GIF89a')):
        return "image/gif"
    if data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WEBP':
        return "image/webp"
    return "image/png"


async def extract_images_from_event(event) -> List[Tuple[bytes, str]]:
    """Extract all images from event message components."""
    images = []
    for comp in event.get_messages():
        if hasattr(comp, 'convert_to_base64'):
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


class ChatFilter(SessionFilter):
    """Session filter keyed by chat_id (chat-level scope)."""

    def filter(self, event: AstrMessageEvent) -> str:
        # Use group_id for group chats, unified_msg_origin for private
        return event.get_group_id() if event.get_group_id() else event.unified_msg_origin


class ImageAdapter:
    """Base class for image generation adapters."""

    def __init__(self, api_key: str, api_url: str, timeout: int = 120):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    async def generate(self, prompt: str, size: str = "1024x1024", n: int = 1) -> List[str]:
        raise NotImplementedError

    async def edit(self, prompt: str, image_bytes: bytes, mime_type: str, size: str = "1024x1024") -> List[str]:
        raise NotImplementedError


class OpenAIAdapter(ImageAdapter):
    """OpenAI-compatible image generation adapter."""

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, size: str = "1024x1024", n: int = 1, quality: str = "standard", style: str = "vivid") -> List[str]:
        url = f"{self.api_url}/v1/images/generations"
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": "url",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._get_headers(), timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error {resp.status}: {error_text}")
                data = await resp.json()
                return [item.get("url") or item.get("b64_json") for item in data.get("data", [])]

    async def edit(self, prompt: str, image_bytes: bytes, mime_type: str, size: str = "1024x1024") -> List[str]:
        url = f"{self.api_url}/v1/images/edits"

        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename="image.png", content_type=mime_type)
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        form.add_field("response_format", "url")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, headers=headers, timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI edit API error {resp.status}: {error_text}")
                data = await resp.json()
                return [item.get("url") or item.get("b64_json") for item in data.get("data", [])]


class GeminiAdapter(ImageAdapter):
    """Gemini image generation adapter."""

    def _get_headers(self) -> dict:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, size: str = "1024x1024", model: str = "imagen-3.0-generate-002") -> List[str]:
        aspect_ratio = convert_size_to_aspect_ratio(size)
        url = f"{self.api_url}/v1beta/models/{model}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }

        if aspect_ratio != "1:1":
            payload["generationConfig"]["aspectRatio"] = aspect_ratio

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._get_headers(), timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Gemini API error {resp.status}: {error_text}")
                data = await resp.json()

                images = []
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inlineData" in part:
                            images.append(part["inlineData"].get("data"))
                return images


class GrokAdapter(OpenAIAdapter):
    """Grok image generation adapter (OpenAI-compatible)."""

    async def generate(self, prompt: str, size: str = "1024x1024", n: int = 1, model: str = "grok-imagine-1.0") -> List[str]:
        url = f"{self.api_url}/v1/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "url",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._get_headers(), timeout=self.timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Grok API error {resp.status}: {error_text}")
                data = await resp.json()
                return [item.get("url") or item.get("b64_json") for item in data.get("data", [])]


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

    def _get_adapter(self, provider: str) -> ImageAdapter:
        """Get the appropriate adapter for the provider."""
        api_key = self.config.get("api_key", "")
        api_url = self.config.get("api_url", "https://api.openai.com")
        timeout = self.config.get("timeout", 120)

        if provider == "openai":
            return OpenAIAdapter(api_key, api_url, timeout)
        elif provider == "gemini":
            return GeminiAdapter(api_key, api_url, timeout)
        elif provider == "grok":
            return GrokAdapter(api_key, api_url, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider}")

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
        persona = persona_mgr.get_persona(persona_name)

        if not persona:
            # List available personas
            all_personas = persona_mgr.get_all_personas()
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
        chat_id = event.get_group_id() if event.get_group_id() else event.unified_msg_origin

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
        @session_waiter(timeout=self.config.get("session_timeout", 300), record_history_chains=False)
        async def img_waiter(controller: SessionController, event: AstrMessageEvent):
            # Check for generate/cancel commands
            msg_str = event.message_str.strip().lower()

            if msg_str == "/generate":
                # Trigger generation (will be handled by generate command)
                await event.send(event.plain_result("正在生成图像..."))
                controller.stop()
                return

            if msg_str == "/cancel":
                await event.send(event.plain_result("已取消绘图会话。"))
                if chat_id in self.ACTIVE_SESSIONS:
                    del self.ACTIVE_SESSIONS[chat_id]
                controller.stop()
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
            await event.send(event.plain_result(f"已记录: 文字({text_preview}) 图片({img_count}张)"))

            controller.keep(timeout=self.config.get("session_timeout", 300))

        try:
            await img_waiter(event, session_filter=ChatFilter())
        except TimeoutError:
            yield event.plain_result("⏰ 绘图会话已超时，请重新开始。")
        finally:
            if chat_id in self.ACTIVE_SESSIONS:
                del self.ACTIVE_SESSIONS[chat_id]
            event.stop_event()

    @filter.command("cancel")
    async def cancel_cmd(self, event: AstrMessageEvent):
        """取消当前绘图会话。"""
        chat_id = event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        if chat_id in self.ACTIVE_SESSIONS:
            del self.ACTIVE_SESSIONS[chat_id]
            yield event.plain_result("已取消绘图会话。")
        else:
            yield event.plain_result("当前没有正在进行的绘图会话。")