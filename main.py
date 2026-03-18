"""
AstrBot Image Generation Plugin - Persona-based chat and image generation.
"""

import base64
import uuid

import aiohttp
import xai_sdk
from google import genai
from google.genai import types

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
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
}


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
        self, prompt: str, size: str = "1024x1024", n: int = 1
    ) -> list[str]:
        raise NotImplementedError

    async def edit(
        self, prompt: str, image_bytes: bytes, mime_type: str, size: str = "1024x1024"
    ) -> list[str]:
        raise NotImplementedError


class OpenAIAdapter(ImageAdapter):
    """OpenAI image generation adapter for gpt-image-1 model."""

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "auto",
        background: str = "auto",
        output_format: str = "png",
    ) -> list[str]:
        """Generate image using gpt-image-1 model."""
        url = f"{self.api_url}/v1/images/generations"
        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": n,
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
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"OpenAI API error {resp.status}: {error_text}")
                data = await resp.json()
                return [
                    item.get("url") or item.get("b64_json")
                    for item in data.get("data", [])
                ]

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        size: str = "1024x1024",
        quality: str = "auto",
        background: str = "auto",
        output_format: str = "png",
    ) -> list[str]:
        """Edit image using gpt-image-1 model."""
        url = f"{self.api_url}/v1/images/edits"

        form = aiohttp.FormData()
        form.add_field(
            "image", image_bytes, filename="image.png", content_type=mime_type
        )
        form.add_field("prompt", prompt)
        form.add_field("model", "gpt-image-1")
        form.add_field("size", size)
        form.add_field("quality", quality)
        form.add_field("background", background)
        form.add_field("output_format", output_format)
        form.add_field("response_format", "url")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=form, headers=headers, timeout=self.timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(
                        f"OpenAI edit API error {resp.status}: {error_text}"
                    )
                data = await resp.json()
                return [
                    item.get("url") or item.get("b64_json")
                    for item in data.get("data", [])
                ]


class GeminiAdapter(ImageAdapter):
    """Gemini image generation adapter using official google-genai SDK."""

    def __init__(self, api_key: str, timeout: int = 120):
        # Note: api_url is not used for Gemini SDK - uses Google's official endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)

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
                    images.append(part.inline_data.data)
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
        image_bytes: bytes,
        mime_type: str,
        size: str = "1024x1024",
        model: str = "imagen-3.0-generate-002",
    ) -> list[str]:
        """Edit image with text prompt (multi-turn editing support) using google-genai SDK."""
        aspect_ratio = convert_size_to_aspect_ratio(size)

        # Create image part from bytes
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        text_part = types.Part.from_text(text=prompt)

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        if aspect_ratio != "1:1":
            config.image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[image_part, text_part],
                config=config,
            )

            images = []
            for part in response.parts:
                if part.inline_data:
                    images.append(part.inline_data.data)
            return images

        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                raise Exception(f"Gemini API 密钥无效或未授权: {e}")
            elif "QUOTA" in error_msg.upper() or "RATE" in error_msg.upper():
                raise Exception(f"Gemini API 配额不足或请求过于频繁: {e}")
            else:
                raise Exception(f"Gemini 图像编辑失败: {e}")


class GrokAdapter(ImageAdapter):
    """Grok adapter using xai-sdk."""

    def __init__(self, api_key: str, timeout: int = 120):
        self.client = xai_sdk.AsyncClient(api_key=api_key)
        self.timeout = timeout

    async def generate(
        self, prompt: str, model: str = "grok-imagine-1.0", aspect_ratio: str = "1:1"
    ) -> list[str]:
        """Generate image using xai-sdk."""
        response = await self.client.image.sample(
            prompt=prompt,
            model=model,
            image_format="url",
        )
        return [response.url]

    async def edit(
        self, prompt: str, image_url: str, model: str = "grok-imagine-1.0"
    ) -> list[str]:
        """Edit image using xai-sdk."""
        response = await self.client.image.sample(
            prompt=prompt,
            model=model,
            image_format="url",
            image_url=image_url,
        )
        return [response.url]


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
            return GeminiAdapter(api_key, timeout)
        elif provider == "grok":
            api_key = self.config.get("grok_api_key") or self.config.get("api_key", "")
            return GrokAdapter(api_key, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _do_generate(
        self, event: AstrMessageEvent, chat_id: str, session_data: dict
    ) -> bool:
        """Execute image generation. Returns True on success."""
        prompt = session_data.get("text", "")
        images = session_data.get("images", [])

        if not prompt and not images:
            await event.send(event.plain_result(ERROR_MESSAGES["no_prompt"]))
            return False

        provider = self.config.get("default_provider", "openai")

        # Check provider-specific API key
        api_key_map = {
            "openai": ("openai_api_key", "api_key"),
            "gemini": ("gemini_api_key", "api_key"),
            "grok": ("grok_api_key", "api_key"),
        }
        primary_key, fallback_key = api_key_map.get(provider, ("api_key", "api_key"))
        if not (self.config.get(primary_key) or self.config.get(fallback_key)):
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
            persona = self.context.persona_manager.get_persona(default_persona)
            if persona and hasattr(persona, "system_prompt") and persona.system_prompt:
                prompt = (
                    f"{persona.system_prompt}\n\n{prompt}"
                    if prompt
                    else persona.system_prompt
                )

        try:
            size = self.config.get("default_size", "1024x1024")

            # Get provider-specific model
            model_map = {
                "openai": self.config.get("openai_model", "gpt-image-1"),
                "gemini": self.config.get("gemini_model", "imagen-3.0-generate-002"),
                "grok": self.config.get("grok_model", "grok-imagine-1.0"),
            }
            model = model_map.get(provider, "")

            # Get provider-specific settings
            aspect_ratio = convert_size_to_aspect_ratio(size)

            if images:
                # Image-to-image
                img_comp = images[0]
                if hasattr(img_comp, "convert_to_base64"):
                    b64_data = await img_comp.convert_to_base64()
                    if b64_data:
                        image_bytes = base64.b64decode(b64_data)
                        mime_type = detect_mime_type(image_bytes)
                        if provider == "gemini":
                            result_urls = await adapter.edit(
                                prompt, image_bytes, mime_type, size, model=model
                            )
                        elif provider == "grok":
                            # Grok needs image URL for editing
                            # Convert bytes to base64 data URL
                            b64_str = base64.b64encode(image_bytes).decode("utf-8")
                            data_url = f"data:{mime_type};base64,{b64_str}"
                            result_urls = await adapter.edit(
                                prompt, image_url=data_url, model=model
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
                                size,
                                quality=quality,
                                background=background,
                                output_format=output_format,
                            )
                    else:
                        if provider == "gemini":
                            result_urls = await adapter.generate(
                                prompt, size, model=model
                            )
                        elif provider == "grok":
                            result_urls = await adapter.generate(
                                prompt, model=model, aspect_ratio=aspect_ratio
                            )
                        else:  # openai
                            quality = self.config.get("openai_quality", "auto")
                            background = self.config.get("openai_background", "auto")
                            output_format = self.config.get(
                                "openai_output_format", "png"
                            )
                            result_urls = await adapter.generate(
                                prompt,
                                size,
                                quality=quality,
                                background=background,
                                output_format=output_format,
                            )
                else:
                    if provider == "gemini":
                        result_urls = await adapter.generate(prompt, size, model=model)
                    elif provider == "grok":
                        result_urls = await adapter.generate(
                            prompt, model=model, aspect_ratio=aspect_ratio
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
                        )
            else:
                # Text-to-image
                if provider == "gemini":
                    result_urls = await adapter.generate(prompt, size, model=model)
                elif provider == "grok":
                    result_urls = await adapter.generate(
                        prompt, model=model, aspect_ratio=aspect_ratio
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
                    )

            # Send results
            for url in result_urls:
                if url.startswith("http"):
                    await event.send(event.image_result(url))
                else:
                    await event.send(event.image_result(f"data:image/png;base64,{url}"))

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
            msg_str = event.message_str.strip().lower()

            if msg_str == "/generate":
                await self._do_generate(event, chat_id, session_data)
                if chat_id in self.ACTIVE_SESSIONS:
                    del self.ACTIVE_SESSIONS[chat_id]
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
            await event.send(
                event.plain_result(f"已记录: 文字({text_preview}) 图片({img_count}张)")
            )

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
        chat_id = (
            event.get_group_id() if event.get_group_id() else event.unified_msg_origin
        )
        if chat_id in self.ACTIVE_SESSIONS:
            del self.ACTIVE_SESSIONS[chat_id]
            yield event.plain_result("已取消绘图会话。")
        else:
            yield event.plain_result("当前没有正在进行的绘图会话。")
