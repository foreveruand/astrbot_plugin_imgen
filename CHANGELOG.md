# Changelog

All notable changes to this project will be documented in this file.

## [1.3.6] - 2026-04-22

### Changed

- **OpenAI-Compatible Routing**:
  - `OpenAIAdapter` now uses `/v1/images/generations` and `/v1/images/edits` only for the official OpenAI host (`api.openai.com`).
  - All non-official OpenAI-compatible base URLs now use `/v1/chat/completions` image modalities for both generate and edit.
  - Keep OpenRouter-specific URL normalization and model alias compatibility.

## [1.3.5] - 2026-04-22

### Fixed

- **Multi-turn Editing Fallback**:
  - When the stored previous image URL cannot be downloaded (for example HTTP 404), generation no longer fails immediately.
  - If a prompt is present, the plugin now falls back to prompt-only generation and clears the broken multi-turn cache entry.
  - Keep explicit error behavior only when there is no prompt to fall back with.

## [1.3.4] - 2026-04-22

### Fixed

- **OpenRouter OpenAI-Compatible Image Calls**:
  - Detect OpenRouter base URL automatically and normalize it to include `/api` when needed (e.g. `https://openrouter.ai` -> `https://openrouter.ai/api`)
  - Route OpenRouter image generation/editing requests through `/v1/chat/completions` with image modalities, matching current OpenRouter image-generation documentation
  - Parse image outputs from chat-style responses (`choices[].message.images`) in addition to classic `data[].url/b64_json`
  - Add model alias normalization for common GPT-5.4 Image 2 typos (e.g. `gpt-5.4-image2` -> `openai/gpt-5.4-image-2`)
  - Improve non-JSON API response handling to surface clearer upstream errors

## [1.3.3] - 2026-04-06

### Changed

- **Grok / xAI Migration**:
  - Remove the `xai-sdk` dependency to avoid package version conflicts
  - Switch Grok image generation to xAI's OpenAI-compatible `/v1/images/generations` REST API
  - Keep Grok image editing on xAI's required JSON `/v1/images/edits` endpoint
  - Add `grok_api_url` configuration and align the default model to `grok-imagine-image`

## [1.3.2] - 2026-03-30

### Fixed

- **Image Sending**:
  - Fix base64/data URL image results being sent through `event.image_result()` and misidentified as remote file IDs
  - Send non-HTTP images via `Comp.Image.fromBase64(...)` to avoid platform-side `Wrong remote file identifier specified: wrong string length`

- **Gemini / Vertex AI**:
  - Add explicit `cloud-platform` OAuth scope for Vertex service account credentials to avoid `invalid_scope`

## [1.3.1] - 2026-03-30

### Fixed

- **Gemini / Vertex AI**:
  - Fix Gemini image result normalization to avoid `startswith first arg must be bytes or a tuple of bytes, not str`
  - Vertex AI mode no longer incorrectly requires `gemini_api_key`
  - Fix uploaded Vertex service account JSON path resolution to read from `astrbot_plugin_data_dir`

- **Multi-image Editing**:
  - Fix session-based image editing to pass all uploaded images to providers that support multi-image input
  - Fix Grok image editing to support up to 5 input images via xAI `images/edits` JSON API
  - Remove misleading Grok "only uses the first one" behavior for tool calls

## [1.3.0] - 2026-03-28

### Added

- **Multi-Image Support for LLM Tool**:
  - Changed `image_url` parameter to `images` (array of strings)
  - Support for multiple image inputs in a single request
  - Backward compatible with single image editing
  - OpenAI and Gemini adapters now support multiple images
  - Grok adapter uses first image (SDK limitation)

- **Enhanced Image Input Format Support**:
  - Image bytes (direct binary input)
  - Local file paths (`/path/to/image.png`)
  - HTTP/HTTPS URLs (`https://example.com/image.png`)
  - Base64 data URLs (`data:image/png;base64,...`)
  - Pure Base64 strings (without `data:` prefix)

- **New `_process_image_input` Helper Method**:
  - Automatically detects and processes various image input formats
  - Unified handling across all adapters
  - Error handling with detailed logging

### Changed

- **ImageAdapter Base Class**:
  - Updated `edit()` method signature to support `images` parameter
  - Maintains backward compatibility with `image_bytes` parameter
  - New signature: `edit(prompt, image_bytes=None, mime_type=None, size=None, images=None, **kwargs)`

- **OpenAIAdapter**:
  - Updated `edit()` to support multiple images via form data
  - Uses `image[]` field for multiple image uploads

- **GeminiAdapter**:
  - Updated `edit()` to support multiple images
  - Builds content parts array with all images followed by text prompt

- **GrokAdapter**:
  - Updated `edit()` to accept bytes and convert to data URL internally
  - Simplified interface, no longer requires external URL conversion

### Technical Details

- LLM Tool parameter type changed from `List[string]` to `array[string]` (AstrBot compatibility)
- All adapters now accept `images: list[tuple[bytes, str]]` parameter
- Internal conversion from URLs to bytes happens in `_process_image_input`

## [1.2.0] - 2026-03-19

### Added

- **Vertex AI Support for Gemini**: 
  - Add `gemini_vertex_enabled` configuration option
  - Support service account JSON credentials file upload (`gemini_vertex_credentials`)
  - Add `gemini_vertex_project` and `gemini_vertex_location` configuration
  - Automatic fallback to API key when Vertex AI config is incomplete

- **LLM Tool Integration**:
  - Add `generate_image` tool exposed via `@filter.llm_tool` decorator
  - Allows AI to generate or edit images during conversations
  - Supports `prompt`, `image_url` (optional), and `size` parameters
  - Works with all three providers: OpenAI, Gemini, Grok
  - Uses default provider from config (no need to specify provider)
  - Supports both text-to-image and image-to-image editing

### Changed

- **generate command now accepts parameters**:
  - `generate` - use default settings
  - `generate 2` - generate 2 images
  - `generate 1024x1024` - use specific resolution
  - `generate 2 1024x1024` - generate 2 images at specific resolution

- **Session command handling fix**:
  - Commands inside `/img` session no longer need `/` prefix
  - Use `generate`, `cancel`, `clear` directly in session

## [1.1.0] - 2026-03-18

### Added

- **Three Independent Provider Configurations**:
  - OpenAI: `openai_api_key`, `openai_api_url`, `openai_model`, `openai_quality`, `openai_background`, `openai_output_format`
  - Gemini: `gemini_api_key`, `gemini_model`, `gemini_aspect_ratio`
  - Grok: `grok_api_key`, `grok_model`, `grok_aspect_ratio`, `grok_resolution`

- **Official SDK Integration**:
  - Gemini adapter uses `google-genai` SDK
  - Grok adapter uses `xai-sdk` SDK
  - OpenAI adapter uses `gpt-image-1` model

- **Multi-turn Image Editing**:
  - Chain editing with KV storage
  - Store last generated image for subsequent edits
  - `/clear` command to reset multi-turn history

- **Configuration**:
  - `enable_multi_turn` option for multi-turn editing
  - `default_persona` for persona-based prompts
  - `default_provider`, `default_size`, `session_timeout`, `max_images`

### Changed

- Updated model names:
  - OpenAI: `gpt-image-1` (previously DALL-E)
  - Gemini: `imagen-3.0-generate-002`
  - Grok: `grok-imagine-1.0`

### Dependencies

- Added `google-genai>=1.0.0`
- Added `xai-sdk>=1.0.0`
- Kept `aiohttp>=3.8.0`

## [1.0.0] - Initial Release

### Added

- `/task <persona> <prompt>` - Memoryless persona-based conversations
- `/img` - Session-based image generation
- `/generate` - Generate images in session
- `/cancel` - Cancel session
- Support for OpenAI, Gemini, and Grok providers
- Session timeout management
- Image-to-image generation support
