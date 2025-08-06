# Qwen3 Special Token Support

## Overview

Yes, our implementation fully supports all Qwen3 special tokens, including `<|endoftext|>`, message boundaries, thinking tags, and more. Rather than manually parsing these, we leverage the tokenizer's built-in capabilities.

## Supported Special Tokens

### Core Message Tokens

- **`<|endoftext|>`** (151643) - Document end/padding token
- **`<|im_start|>`** (151644) - Message start boundary
- **`<|im_end|>`** (151645) - Message end boundary (also EOS)

### Thinking/Reasoning

- **`<think>`** (151667) - Start of thinking section
- **`</think>`** (151668) - End of thinking section

### Tool Calling

- **`<tool_call>`** (151659) - Start tool call
- **`</tool_call>`** (151660) - End tool call
- **`<tool_response>`** (151665) - Start tool response
- **`</tool_response>`** (151666) - End tool response

### Vision/Multimodal

- **`<|vision_start|>`**, **`<|vision_end|>`** - Vision content boundaries
- **`<|image_pad|>`**, **`<|video_pad|>`** - Padding tokens
- **`<|box_start|>`**, **`<|box_end|>`** - Bounding box markers
- **`<|object_ref_start|>`**, **`<|object_ref_end|>`** - Object references

## Implementation Architecture

### 1. Token Registry (`qwen3_tokens.py`)

```python
class Qwen3SpecialTokens:
    """Complete registry of all Qwen3 special tokens"""
    ENDOFTEXT = "<|endoftext|>"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINK_START = "<think>"
    THINK_END = "</think>"
    # ... and more
```

### 2. Token Handler (`qwen3_tokens.py`)

```python
class Qwen3TokenHandler:
    """Manages token operations and generation control"""

    def configure_generation_params(self, enable_thinking=True):
        # Returns proper stop tokens including <|im_end|>, <|endoftext|>

    def extract_structured_content(self, text):
        # Extracts thinking, tool calls, removes boundaries
```

### 3. Integration with Tokenizer

```python
# The tokenizer's chat template handles these automatically:
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Controls thinking behavior
)
```

## How It Works

### Generation Flow

1. **Input Processing**: Messages are formatted with proper boundaries

   ```
   <|im_start|>user
   What is 2+2?
   <|im_end|>
   <|im_start|>assistant
   ```

2. **Generation**: Model generates with special tokens

   ```
   <think>
   Let me calculate 2+2...
   </think>

   The answer is 4.
   <|im_end|>
   ```

3. **Output Processing**: System extracts and structures content
   - Thinking content → `reasoning` field
   - Main content → `content` field
   - Boundaries removed from user-facing output

### Stop Token Behavior

- Primary stops: `<|im_end|>`, `<|endoftext|>`
- Thinking stops: `</think>` (when structured output needed)
- Tool stops: `</tool_call>`, `</tool_response>`

### Streaming Support

- Detects token boundaries during streaming
- Can emit reasoning events separately
- Maintains proper chunking at token boundaries

## Key Benefits

1. **Native Support**: Uses tokenizer's built-in understanding
2. **No Manual Parsing**: Tokenizer handles complex templates
3. **Comprehensive**: All Qwen3 tokens supported
4. **Efficient**: No regex overhead for most operations
5. **Flexible**: Can configure for different use cases

## Testing

Run comprehensive token tests:

```bash
python examples/test_qwen3_tokens.py
```

This demonstrates:

- Message boundary handling
- Thinking extraction
- Stop token behavior
- Streaming with special tokens

## Example API Usage

```python
# Request with thinking enabled
response = requests.post("/v1/chat/completions", json={
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Complex question"}],
    "enable_thinking": True,
    "include_reasoning": True
})

# Response includes structured output
{
    "choices": [{
        "message": {
            "content": "Main response without special tokens"
        },
        "reasoning": {
            "id": "reasoning_123",
            "content": "Thinking content extracted"
        }
    }]
}
```

## Files Involved

- `qwen3_tokens.py` - Token registry and handler
- `thinking_handler.py` - Thinking extraction with Qwen3 support
- `model_manager.py` - Detects Qwen3 and configures handler
- `generation_simple.py` - Uses native tokenizer capabilities

The system automatically detects Qwen3 models and enables full special token support!
