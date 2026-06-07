---
name: imgen_tool_prompt_expansion
description: Use when calling the `generate_image` tool. Expand the user's image request into a clear, detailed visual prompt before generation, avoid provider/model overrides unless the user confirms a configuration change, and send the generated image to the user after generation.
---

# Image Generation Prompt Expansion

Use this skill when the user asks to create, draw, render, generate, redesign, or edit an image with `generate_image`.

## Core Rule

Always expand the user's input before calling the tool. The `prompt` argument should be a complete image brief, not just the user's short wording, unless the user explicitly asks you to preserve their prompt exactly.

After `generate_image` succeeds, call `send_message_to_user` to send the generated image result to the user. Do not stop after generating the image internally.

## Expansion Checklist

Turn the user's request into a concise but specific prompt that includes the important visual details:

- Subject: who or what should appear.
- Composition: framing, camera angle, focal point, and spatial arrangement.
- Style: photo, illustration, anime, product render, poster, UI mockup, watercolor, pixel art, or another requested medium.
- Environment: location, background, time of day, weather, era, or scene context.
- Lighting and color: mood, palette, contrast, shadows, highlights, and material appearance.
- Quality constraints: sharp focus, clean anatomy, readable text only when requested, coherent details, no extra limbs or artifacts when relevant.
- Editing intent: if source images are provided, state exactly what should change and what must remain unchanged.

Keep the expanded prompt faithful to the user. Add helpful visual specificity, but do not silently change the subject, identity, brand, text content, or requested style.

## Tool Argument Guidance

- `prompt`: pass the expanded prompt.
- `images`: include source image URLs, local paths, or base64 payloads only when the user provided images or clearly asks to edit an existing image.
- `size`: pass only when the user requests an aspect ratio, orientation, resolution, poster/wallpaper/avatar format, or another size-sensitive output. Otherwise omit it and let the plugin use `default_size`.
- Do not pass `provider` to `generate_image`.
- Do not pass `model` to `generate_image`.
- If the user asks to use a specific provider or model, do not guess and do not pass `provider` or `model` immediately. Ask the user which provider/model they want to configure or confirm the exact intended model name first. After the user confirms, tell them the plugin configuration should be changed, then call `generate_image` without `provider` and without `model` so it uses the configured defaults.
- If the user gives an incomplete or fuzzy model name, use the reference list below only to ask a clarifying question. Do not silently map the partial name and do not pass the mapped model as a tool argument.

## Model Name Reference For Clarification Only

Use this list only to understand likely user intent and ask a precise follow-up when they request a provider/model change. Never pass these names in `provider` or `model` arguments.

- OpenAI image models:
  - `gpt-image-2`
  - `gpt-image-1.5`
  - `gpt-image-1-mini`
  - `gpt-image-1`
  - `dall-e-3`
  - `dall-e-2`
- Google Gemini native image models:
  - `gemini-3-pro-image-preview`
  - `gemini-2.5-flash-image`
  - `gemini-2.5-flash-image-preview`
- Google Imagen models:
  - `imagen-4.0-generate-001`
  - `imagen-4.0-ultra-generate-001`
  - `imagen-4.0-fast-generate-001`
  - `imagen-3.0-generate-002`
- xAI/Grok image models:
  - `grok-imagine-image-quality`
  - `grok-imagine-image`
  - `grok-imagine-image-2026-03-02`
  - `grok-imagine-image-pro` (legacy/deprecated; prefer asking the user to confirm migration to `grok-imagine-image-quality`)

## Prompt Patterns

For text-to-image:

```text
generate_image(
  prompt="Expanded visual brief with subject, composition, style, environment, lighting, color, and constraints."
)
send_message_to_user(...)
```

For image editing:

```text
generate_image(
  prompt="Edit the provided image: detailed change request. Preserve the original identity, pose, composition, and important unchanged elements unless the user says otherwise.",
  images=["..."]
)
send_message_to_user(...)
```

If the user requests a provider/model override:

```text
Ask a clarification question first. Do not pass provider or model to generate_image.
```

## Examples

User request:
`画一只猫`

Tool call shape:

```text
prompt="A charming domestic cat sitting by a sunlit window, soft natural morning light, detailed fur texture, relaxed expression, cozy indoor background with subtle plants, balanced composition, warm color palette, clean high-quality illustration."
```

User request:
`把这张头像改成赛博朋克风，但脸别变`

Tool call shape:

```text
prompt="Edit the provided avatar into a cyberpunk style portrait. Preserve the person's face, identity, expression, head shape, and main pose. Add neon city lighting, subtle futuristic clothing details, cool magenta and cyan highlights, high-contrast cinematic shadows, and a polished digital art finish."
images=["..."]
```

User request:
`用 grok-imagine-image 生成 16:9 的机械城市壁纸`

Tool call shape:

```text
Ask: "你想把插件默认模型切换/确认为 grok-imagine-image 吗？当前工具调用不会直接传入 provider 或 model。"
```

After the user confirms the configured model/provider:

```text
prompt="A vast futuristic mechanical city designed as a 16:9 desktop wallpaper, towering interlocking metal structures, glowing transit lines, atmospheric depth, dramatic dusk lighting, crisp architectural detail, cinematic wide-angle composition, no text or logos."
size="2048x1152"
```

Then call `send_message_to_user` with the generated image.
