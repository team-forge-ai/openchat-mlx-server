# 🏗️ MLX Server Architecture - Thinking/Reasoning Support

## Overview

The MLX server provides a clean, unified architecture for handling thinking/reasoning capabilities across different model types, with special support for advanced models like Qwen3.

## Core Components

### 1. **ThinkingManager** (`thinking_manager.py`)

The central hub for all thinking/reasoning operations.

```python
class ThinkingManager:
    """Unified manager for thinking/reasoning operations."""

    def detect_capability() -> ThinkingCapability
    def apply_chat_template(messages, enable_thinking) -> str
    def extract_reasoning(output) -> ThinkingResult
    def process_streaming_chunk(chunk) -> (text, event)
```

**Capabilities:**

- `NONE`: No thinking support
- `BASIC`: Pattern matching only
- `NATIVE`: Tokenizer template support
- `ADVANCED`: Full special token support (Qwen3)

### 2. **GenerationEngine** (`generation.py`)

Handles text generation with integrated thinking support.

```python
class GenerationEngine:
    """Unified generation engine with thinking support."""

    def set_thinking_manager(manager)
    def generate(model, tokenizer, messages, ...) -> result
    def generate_async(...) -> AsyncIterator
```

### 3. **ModelManager** (`model_manager.py`)

Manages model loading and capability detection.

```python
class MLXModelManager:
    """Model manager with thinking detection."""

    def load_model(path) -> (success, message)
    def get_thinking_manager() -> ThinkingManager
    def format_chat_template(messages, enable_thinking) -> str
```

### 4. **Qwen3TokenHandler** (`qwen3_tokens.py`)

Specialized handler for Qwen3's rich token set.

```python
class Qwen3TokenHandler:
    """Handler for Qwen3 special tokens."""

    def configure_generation_params(enable_thinking) -> config
    def extract_structured_content(text) -> structured_result
```

## Data Flow

```
User Request
    ↓
MLXServer
    ↓
GenerationEngine
    ├── ThinkingManager.apply_chat_template()
    ├── MLX Model Generation
    └── ThinkingManager.extract_reasoning()
    ↓
Response with Reasoning
```

## Key Design Principles

### 1. **Automatic Excellence**

- Thinking is AUTO-ENABLED for capable models
- No configuration needed for optimal results
- Smart defaults based on model capabilities

### 2. **Unified Interface**

- Single `ThinkingManager` for all operations
- Consistent API across different model types
- Clean separation of concerns

### 3. **Progressive Enhancement**

- Basic models get basic support
- Advanced models get full features
- Graceful degradation when features unavailable

### 4. **Clean Abstractions**

```
ThinkingManager (High-level API)
    ↓
Capability Detection
    ↓
Template Application
    ↓
Reasoning Extraction
```

## API Structure

### Request

```json
{
  "messages": [...],
  "enable_thinking": null,  // Auto-detect (default)
  "include_reasoning": true  // Include in response
}
```

### Response

```json
{
  "choices": [
    {
      "message": {
        "content": "Main response"
      },
      "reasoning": {
        "id": "reasoning_123",
        "content": "Thinking process..."
      }
    }
  ]
}
```

## File Organization

```
src/openchat_mlx_server/
├── thinking_manager.py    # Core thinking logic
├── generation.py          # Generation engine
├── model_manager.py       # Model management
├── qwen3_tokens.py       # Qwen3 special tokens
├── server.py             # FastAPI server
└── api_models.py         # API data models

tests/
└── test_thinking_integration.py  # Comprehensive tests

examples/
├── thinking_demo.py      # Main demo script
├── test_auto_thinking.py # Auto-enable tests
└── test_qwen3_tokens.py  # Token support tests
```

## Extension Points

### Adding New Model Support

1. **Detect capability** in `ThinkingManager._detect_capability()`
2. **Add token handler** if needed (like `Qwen3TokenHandler`)
3. **Update template logic** in `_apply_*_template()` methods

### Custom Thinking Patterns

```python
# In ThinkingManager._extract_pattern_reasoning()
patterns = [
    (re.compile(r'<think>(.*?)</think>'), 'think'),
    (re.compile(r'YOUR_PATTERN'), 'your_type'),  # Add here
]
```

## Performance Considerations

- **Lazy Loading**: Token handlers created only when needed
- **Streaming**: Efficient chunk-based processing
- **Caching**: Generation configs cached per session
- **Memory**: Automatic cleanup on model unload

## Testing Strategy

### Unit Tests

- `test_thinking_integration.py`: Core functionality
- Mock-based testing for isolation
- Coverage of all capability levels

### Integration Tests

- `thinking_demo.py`: End-to-end testing
- Real API calls
- Multiple test scenarios

### Manual Testing

```bash
# Run comprehensive tests
pytest tests/test_thinking_integration.py -v

# Run demo
python examples/thinking_demo.py

# Test specific features
python examples/test_auto_thinking.py
python examples/test_qwen3_tokens.py
```

## Best Practices

1. **Let the system handle thinking** - Don't override unless necessary
2. **Use control tags sparingly** - `/think` and `/no_think` for edge cases
3. **Trust auto-detection** - System knows model capabilities
4. **Include reasoning by default** - Better visibility into model thinking

## Future Enhancements

- [ ] Token-level streaming of thinking content
- [ ] Thinking metrics and analytics
- [ ] Multi-model ensemble thinking
- [ ] Thinking caching for similar queries
- [ ] Visual thinking representation

## Summary

The refactored architecture provides:

✅ **Clean Code**: Single responsibility, clear interfaces  
✅ **Extensible**: Easy to add new models and capabilities  
✅ **Performant**: Optimized for both simple and complex queries  
✅ **User-Friendly**: Works great out of the box  
✅ **Well-Tested**: Comprehensive test coverage

The system elegantly handles everything from basic pattern matching to advanced special token processing, all through a unified interface that "just works" for users.
