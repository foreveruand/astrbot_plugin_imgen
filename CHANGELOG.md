# Changelog

All notable changes to this project will be documented in this file.

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