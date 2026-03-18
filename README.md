# AstrBot 图像生成插件 (astrbot_plugin_imgen)

本插件为 AstrBot 提供基于人格的无记忆对话功能以及会话式图像生成功能。

## 功能特性

- **/task**: 基于指定人格的无记忆对话。
- **/img**: 开启会话式图像生成，支持文本描述和多张图片输入。
- **图像生成**: 支持 OpenAI, Gemini, Grok 模型。
- **多轮编辑**: 支持基于已生成图像进行二次编辑。
- **会话管理**: 支持 5 分钟自动超时，支持 `/generate` 生成，`/clear` 清除记录，或 `/cancel` 取消。

## 安装

将插件放置在 AstrBot 的插件目录中，通常位于 `data/plugins/astrbot_plugin_imgen`。重启 AstrBot 或在管理面板中启用该插件。

## 配置指南

在插件设置中，请根据您的提供商配置以下参数：

### 通用配置
- **default_provider**: 默认生成服务，可选 `openai`, `gemini`, `grok`。
- **default_size**: 图片分辨率（例如 `1024x1024`）。
- **session_timeout**: 会话超时时间，默认 300 秒。
- **max_images**: 单次会话支持的最大输入图片数。
- **enable_multi_turn**: 是否启用多轮编辑（默认启用）。
- **default_persona**: 默认人格。

### OpenAI 配置
- **openai_api_key**: 必填。
- **openai_api_url**: API 基础地址，默认为 `https://api.openai.com`。
- **openai_model**: 图像生成模型，默认为 `gpt-image-1`。
- **openai_quality**: 图像质量 (`low`, `medium`, `high`, `auto`)。
- **openai_background**: 图像背景 (`transparent`, `opaque`, `auto`)。
- **openai_output_format**: 输出格式 (`png`, `jpeg`, `webp`)。

### Gemini 配置
- **gemini_api_key**: 必填。
- **gemini_model**: 生成模型，默认为 `imagen-3.0-generate-002`。
- **gemini_aspect_ratio**: 宽高比 (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`)。

### Grok 配置
- **grok_api_key**: 必填。
- **grok_model**: 生成模型，默认为 `grok-imagine-1.0`。
- **grok_aspect_ratio**: 宽高比 (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`)。
- **grok_resolution**: 图片分辨率。

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
- **/generate**: 发送此命令开始生成图像。
- **/clear**: 清除多轮编辑历史。
- **/cancel**: 取消当前会话。

### 3. 多轮编辑流程
1. 用户输入: `/img 一只在森林里的猫`
2. 用户输入: `/generate` → 生成并存储猫的图片。
3. 用户输入: `/img 给猫加上帽子`
4. 用户输入: `/generate` → 对上一张图进行编辑（加上帽子）。
5. 用户输入: `/clear` → 重置历史，下一次 `/generate` 将生成全新图片。

## 支持的 API 服务

本插件已支持以下服务：
- OpenAI: `gpt-image-1`
- Google Gemini: `imagen-3.0-generate-002` (使用 `google-genai` SDK)
- Grok: `grok-imagine-1.0` (使用 `xai-sdk`)

## 常见问题

- **提示“未配置 API 密钥”？**
  请检查插件设置中的 `api_key` 是否正确填写。
- **绘图生成失败？**
  请检查网络连接以及 API 密钥余额，或者在日志中查看详细错误提示。
- **会话超时了？**
  会话默认 5 分钟无操作会自动关闭，请重新输入 `/img` 开始新会话。

---
*Enjoy your creative journey with AstrBot!*
