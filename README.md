# AstrBot 图像生成插件 (astrbot_plugin_imgen)

本插件为 AstrBot 提供基于人格的无记忆对话功能以及会话式图像生成功能。

## 功能特性

- **/task**: 基于指定人格的无记忆对话。
- **/img**: 开启会话式图像生成，支持文本描述和多张图片输入。
- **图像生成**: 支持 OpenAI (DALL-E), Gemini (Imagen), Grok 模型。
- **会话管理**: 支持 5 分钟自动超时，支持 `/generate` 生成或 `/cancel` 取消。

## 安装

将插件放置在 AstrBot 的插件目录中，通常位于 `data/plugins/astrbot_plugin_imgen`。重启 AstrBot 或在管理面板中启用该插件。

## 配置指南

在插件设置中，请确保配置以下核心参数：

- **api_key**: 图像生成 API 的密钥（必填）。
- **api_url**: API 基础地址（选填，默认为 OpenAI API）。
- **default_provider**: 默认生成服务，可选 `openai`, `gemini`, `grok`。
- **default_model**: 生成模型名称。
- **default_size**: 图片分辨率（例如 `1024x1024`）。
- **session_timeout**: 会话超时时间，默认 300 秒。
- **max_images**: 单次会话支持的最大输入图片数。

此外，可在 `openai_settings`, `gemini_settings`, `grok_settings` 中配置特定供应商的详细参数。

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
- **/cancel**: 取消当前会话。

*示例流程:*
1. 用户输入: `/img 一只在森林里的猫`
2. 机器人回复: `已开始，请输入更多描述或发送图片。`
3. 用户发送一张猫的参考图。
4. 用户输入: `/generate`
5. 机器人回复: `✅ 图像生成成功！` 并发送生成的图片。

## 支持的 API 服务

本插件已支持以下服务：
- OpenAI (DALL-E 3)
- Google Gemini (Imagen 3)
- Grok

## 常见问题

- **提示“未配置 API 密钥”？**
  请检查插件设置中的 `api_key` 是否正确填写。
- **绘图生成失败？**
  请检查网络连接以及 API 密钥余额，或者在日志中查看详细错误提示。
- **会话超时了？**
  会话默认 5 分钟无操作会自动关闭，请重新输入 `/img` 开始新会话。

---
*Enjoy your creative journey with AstrBot!*
