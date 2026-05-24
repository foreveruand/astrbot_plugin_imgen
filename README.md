# AstrBot 图像生成插件 (astrbot_plugin_imgen)

本插件为 AstrBot 提供基于人格的无记忆对话功能以及会话式图像生成功能。

## 功能特性

- **/task**: 基于指定人格的无记忆对话。
- **/img**: 开启会话式图像生成，支持文本描述和多张图片输入。
- **Telegram Inline**: 在 Telegram 中输入 `@botname <描述>` 后，可选择“图像生成助手”并基于 query 直接生成一张图片返回到内联消息。
- **图像生成**: 支持 OpenAI, Gemini, Grok 模型。
- **Vertex AI**: 支持 Google Cloud Vertex AI 认证。
- **多轮编辑**: 支持在当前会话中基于用户显式提供的图片进行编辑。
- **LLM 工具**: 可作为 AI 工具调用，让 AI 在对话中生成图像。
- **会话管理**: 支持 5 分钟自动超时，支持 `/generate` 生成，`/clear` 清除记录，或 `/cancel` 取消。

## 安装

将插件放置在 AstrBot 的插件目录中，通常位于 `data/plugins/astrbot_plugin_imgen`。重启 AstrBot 或在管理面板中启用该插件。

## 配置指南

插件设置按二级菜单分区。请根据您的提供商配置以下参数：

### 通用配置
- **default_provider**: 默认生成服务，可选 `openai`, `gemini`, `grok`。
- **default_size**: 默认图片分辨率，可在设置中选择 `auto`、常用 1K/2K/4K 尺寸。OpenAI 使用像素级 `size`；Gemini/Grok 会映射为各自支持的宽高比和 1K/2K 分辨率参数。
- **session_timeout**: 会话超时时间，默认 300 秒。
- **max_images**: 单次会话支持的最大输入图片数。
- **enable_multi_turn**: 是否启用多轮编辑（默认启用）。
- **default_persona**: 默认人格。

### OpenAI 兼容接口
- **api_key**: 必填。
- **api_url**: API 基础地址，默认为 `https://api.openai.com`。
- **model**: 图像生成模型，默认为 `gpt-image-1`。使用第三方兼容服务时请填写该服务要求的完整模型名。
- **use_completions**: 是否使用 `/v1/chat/completions` 图像模态接口（默认开启）。关闭后使用 `/v1/images/generations` 和 `/v1/images/edits`。
- **OpenAI Images API 兼容说明**: 当关闭 `use_completions` 且使用 `gpt-image-*` 模型时，插件会按 OpenAI 标准请求体发送 `model`、`prompt`、`n`、`size`，并仅在非默认值时发送 `quality`、`background`、`output_format`，不会再发送 `response_format`。旧尺寸 `1792x1024` / `1024x1792` 会自动规范化为 `1536x1024` / `1024x1536`。
- **gpt-image-2 尺寸说明**: 4K 直接通过 `size` 传递，例如 `3840x2160` 或 `2160x3840`。宽高都必须是 16 的倍数，最长边不超过 3840，长短边比例不超过 3:1，总像素不超过 8,294,400。
- **quality**: 图像质量 (`low`, `medium`, `high`, `auto`)。
- **background**: 图像背景 (`transparent`, `opaque`, `auto`)。
- **output_format**: 输出格式 (`png`, `jpeg`, `webp`)。

### Gemini / Imagen
- **api_key**: 必填（未启用 Vertex AI 时）。
- **model**: 生成模型，默认为 `imagen-3.0-generate-002`。
- **尺寸映射**: Gemini API 不使用 OpenAI 的像素级 `size`。插件会把通用 `default_size` 映射为 Gemini 的 `aspectRatio`，并在 2K/4K 选项下传递 `imageSize=2K`（Gemini/Imagen 官方仅列出 1K/2K）。

#### Vertex AI 配置（可选）
- **vertex_enabled**: 启用 Vertex AI。
- **vertex_credentials**: 服务账号 JSON 凭证文件。
- **vertex_project**: Google Cloud 项目 ID。
- **vertex_location**: Vertex AI 区域（默认 `us-central1`）。

### Grok / xAI
- **api_key**: 必填。
- **model**: 生成模型，默认为 `grok-imagine-image`。Grok API 地址固定为 `https://api.x.ai`。
- **尺寸映射**: xAI 使用 `aspect_ratio` 和 `resolution`，不使用 OpenAI 的像素级 `size`。插件会从通用 `default_size` 自动映射宽高比，并将 2K/4K 选项映射为 `resolution=2k`。

## 使用说明

### 1. 开启人格对话
使用 `/task` 命令调用指定人格进行一次性对话：
```
/task <人格名> <提示词>
```
*示例: /task 动漫少女 你好，请帮我画一只猫。*

### 2. 图像生成会话
使用 `/img` 命令开始会话，机器人将进入等待状态，收集接下来的文字描述或图片。
- **/img [初始描述]**: 开始绘图会话。
- 发送图片或文字以补充描述。
- **generate [数量] [分辨率]**: 发送此命令开始生成图像（会话内不需要 `/` 前缀）。
  - `generate` - 使用默认设置
  - `generate 2` - 生成 2 张图片
  - `generate 1024x1024` - 使用 1024x1024 分辨率
  - `generate 2 1024x1024` - 生成 2 张 1024x1024 分辨率的图片
- **clear**: 清除多轮编辑历史（会话内不需要 `/` 前缀）。
- **cancel**: 取消当前会话（会话内不需要 `/` 前缀）。

> **注意**: 在 `/img` 会话内，命令不需要加 `/` 前缀，直接发送 `generate`、`clear`、`cancel` 即可。
> **补充**: 发送 `generate` 后，插件会立即结束当前 `/img` 会话并在后台执行绘图请求，因此机器人仍会继续响应同一聊天中的其他命令；如果要中止正在进行的请求，可发送 `/cancel`。

### 3. 编辑流程
1. 用户输入: `/img 给海报加上金色标题`
2. 用户发送要编辑的图片
3. 用户输入: `generate` → 按当前文字和图片执行图生图。
4. 用户输入: `clear` → 清除已缓存结果。

### 4. LLM 工具调用
本插件提供 `generate_image` 工具，AI 可以在对话中自动调用：
- **文生图**: AI 直接根据用户描述生成图片
- **图生图**: AI 可以编辑用户发送的图片，支持多图输入

工具参数：
- `prompt` (必填): 图像描述
- `images` (可选): 要编辑的图片列表（array），支持以下格式：
  - 本地路径: `/data/images/cat.png`
  - HTTP/HTTPS URL: `https://example.com/image.png`
  - Base64 Data URL: `data:image/png;base64,...`
  - 纯 Base64 字符串
- `size` (可选): 图像尺寸；未传入时默认使用插件设置中的 `default_size`
- `provider` (可选): 指定绘图提供商，可选 `openai`、`gemini`、`grok`

工具调用默认使用设置中的 `default_provider`。如果 LLM 显式传入 `provider`，插件会优先尝试该渠道；若该渠道缺少必要配置，则自动回退到 `default_provider`。

### 5. Telegram Inline 绘图
当 AstrBot 的 Telegram 平台启用了 inline 模式后，可以直接在任意聊天中使用：
```
@botname 夕阳下的机械城市
```

然后在候选项中选择“图像生成助手”。插件会读取当前 inline query，调用默认图像提供商绘制，并把生成结果直接回填为图片消息。

注意：
- 只有在用户真正点击了“图像生成助手”候选项之后才会开始生成图片；仅输入 inline query、还未点选候选项时不会触发绘图。
- 该模式当前按一次 query 返回 1 张图片。
- 仅在选择“图像生成助手”这个插件选项时才会触发本插件；如果选择“插件命令”等其他 inline 选项，会交给对应命令/插件处理。
- 返回图片依赖 Telegram inline 消息编辑能力；失败时会回退为错误文本。
- 使用的提供商、模型、默认分辨率和默认人格，均沿用本插件现有配置。

## 支持的 API 服务

本插件已支持以下服务：
- OpenAI: `gpt-image-1`
- OpenRouter（通过 OpenAI 兼容配置）: 支持 `openai/gpt-5.4-image-2` 等图像模型。请在 OpenAI 兼容接口分区将 `api_url` 配为 `https://openrouter.ai/api`，并保持 `use_completions` 开启。
- Google Gemini: `imagen-3.0-generate-002`（使用 `google-genai` SDK）
- Grok: `grok-imagine-image`（通过 xAI OpenAI 兼容 `/v1/images/generations` 与 JSON `/v1/images/edits` 接口，无需 `xai_sdk`）

## 常见问题

- **提示“未配置 API 密钥”？**
  请检查当前提供商的配置是否完整。使用 Gemini Vertex AI 时，需要启用 `vertex_enabled`、填写 `vertex_project`，并上传 `vertex_credentials`，此时不需要再填写 `api_key`。
- **绘图生成失败？**
  请检查网络连接以及 API 密钥余额，或者在日志中查看详细错误提示。
- **为什么我新开 `/img` 只写文字，却触发了编辑接口？**
  新版本已调整为：`/img` 会话中如果当前没有上传图片，文字请求只会走文生图，不会自动复用历史图片缓存。
- **第三方兼容接口返回 `![image]` + 编码，提示“未返回图片”？**
  新版本已兼容这类字符串格式，会自动从 `choices[].message.content` 中提取图片编码。同时错误日志和用户侧错误信息会自动脱敏与截断，避免输出整段图片编码导致平台流控。
- **Vertex AI 报 `invalid_scope`？**
  插件现在会为上传的服务账号凭证显式申请 `https://www.googleapis.com/auth/cloud-platform` scope。更新到最新版插件后重新加载即可。
- **Grok 现在还依赖 `xai_sdk` 吗？**
  不再依赖。插件已改为调用 xAI OpenAI 兼容的图像生成接口，并在图像编辑场景下使用官方要求的 JSON `/v1/images/edits` 请求，从而避开 `xai_sdk` 的版本冲突。
- **Grok 支持多图编辑吗？**
  支持。根据 xAI 官方文档，Grok 图像编辑最多可接收 5 张输入图片，插件会在工具调用和会话模式下按顺序传递这些图片。
- **会话超时了？**
  会话默认 5 分钟无操作会自动关闭，请重新输入 `/img` 开始新会话。
- **为什么 `generate` 之后机器人还能继续响应其他命令？**
  这是当前版本的预期行为。插件会在后台执行绘图请求，避免 AstrBot 的会话控制器持续占用当前聊天；如果本次绘图还没结束，`/cancel` 可以中止该请求。

---
*Enjoy your creative journey with AstrBot!*
