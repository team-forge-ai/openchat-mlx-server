# Thinking/Reasoning Support Implementation

## Overview

This implementation provides OpenAI-compatible thinking/reasoning support for models like Qwen3 that have native thinking capabilities. Instead of manually parsing output, we leverage the tokenizer's built-in chat template support.

## Key Insight: Use Native Tokenizer Support

Rather than writing custom parsers for `<think>...</think>` tags, we leverage:

1. **Tokenizer's Chat Template**: Models like Qwen3 have chat templates that understand `enable_thinking` parameter
2. **Special Tokens**: The tokenizer has special tokens for thinking boundaries
3. **MLX-LM Integration**: Direct integration with mlx-lm's generation capabilities

## Architecture

### 1. Thinking Handler (`thinking_handler.py`)

- Detects if a model supports thinking
- Applies chat templates with thinking parameters
- Handles special token decoding
- Provides fallback extraction for legacy support

### 2. Simplified Generation Engine (`generation_simple.py`)

- Uses tokenizer's `apply_chat_template()` with `enable_thinking`
- Leverages mlx-lm's native generation
- Minimal post-processing of output

### 3. API Integration

- OpenAI-compatible endpoints with reasoning support
- `enable_thinking` parameter in requests
- `include_reasoning` to control response format
- Returns reasoning items similar to OpenAI's format

## Usage

### Basic Request with Thinking

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "Solve: What is 15 * 23?"}
        ],
        "enable_thinking": True,
        "include_reasoning": True
    }
)
```

### Response Format

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The answer is 345."
      },
      "reasoning": {
        "id": "reasoning_abc123",
        "type": "reasoning",
        "content": "Let me calculate 15 * 23 step by step..."
      }
    }
  ]
}
```

## How It Works

### 1. Template Application

```python
# The tokenizer applies its chat template with thinking support
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # ← Key parameter
)
```

### 2. Generation

```python
# MLX-LM generates with the properly formatted prompt
output = mlx_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    # ... other parameters
)
```

### 3. Output Processing

- If model outputs thinking tags, they're extracted
- Content and reasoning are separated
- Returned in OpenAI-compatible format

## Benefits of This Approach

1. **Simpler**: No complex parsing logic needed
2. **Native**: Uses the model's trained behavior
3. **Efficient**: Leverages tokenizer's optimized handling
4. **Compatible**: Works with OpenAI API format
5. **Flexible**: Supports both Qwen3-style and OpenAI-style responses

## Model Support

### Supported Models

- **Qwen3**: Full thinking support with `<think>` tags
- **Other MLX models**: Graceful degradation (no thinking)

### Detection

The system automatically detects thinking support by:

1. Checking for think/end_think special tokens
2. Analyzing the chat template for `enable_thinking` parameter
3. Checking model architecture (e.g., Qwen3)

## Configuration

### Server Start

```bash
# Start server with a Qwen3 model
python -m openchat_mlx_server.main \
    --model-path models/Qwen3-0.6B-MLX-4bit
```

### Default Behavior (NEW!)

**Thinking is now ENABLED by default for models that support it!**

- Models like Qwen3 will automatically use thinking unless explicitly disabled
- No need to set `enable_thinking=true` in every request
- Better responses out of the box for complex queries

### Request Parameters

- `enable_thinking`:
  - `null` (default) → **Auto-enabled for capable models**
  - `true` → Force enable thinking
  - `false` → Disable thinking
- `include_reasoning`: Include reasoning in response (default: `true`)
- `/think` or `/no_think` in messages for control tag override

## Testing

Run the test scripts to verify thinking support:

```bash
# Test native tokenizer thinking support
python examples/test_thinking_simple.py

# Test auto-enable behavior (thinking on by default)
python examples/test_auto_thinking.py

# Test comprehensive Qwen3 token support
python examples/test_qwen3_tokens.py
```

## Implementation Files

- `thinking_handler.py`: Core thinking detection and handling
- `generation_simple.py`: Simplified generation with native support
- `api_models.py`: Extended with ReasoningItem models
- `model_manager.py`: Detects and configures thinking support
- `server.py`: API endpoints with reasoning support

## Future Improvements

1. **Streaming**: Better streaming of reasoning events
2. **Summary Generation**: Auto-summarize long reasoning
3. **Multi-Model**: Support for more thinking-capable models
4. **Metrics**: Token counting for reasoning content
5. **Caching**: Cache reasoning for similar queries
