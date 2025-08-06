# 🧠 Thinking is Now Enabled by Default!

## What Changed?

For models that support thinking/reasoning (like Qwen3), **thinking is now automatically enabled** without needing to specify `enable_thinking=true` in your requests.

## Why This Change?

1. **Better responses out of the box** - Complex queries get thoughtful, step-by-step reasoning
2. **Simpler API usage** - No need to remember to enable thinking
3. **Backwards compatible** - You can still explicitly control it when needed

## How It Works

### Default Behavior

```python
# Before: Had to explicitly enable thinking
response = requests.post("/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Solve: What is 15 * 23?"}],
    "enable_thinking": True  # Had to remember this!
})

# Now: Thinking is automatic for capable models!
response = requests.post("/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Solve: What is 15 * 23?"}]
    # No enable_thinking needed - it's on by default!
})
```

### Control Options

| Setting                              | Behavior                                               |
| ------------------------------------ | ------------------------------------------------------ |
| `enable_thinking: null` (or omitted) | ✅ **Auto-enabled** for Qwen3 and other capable models |
| `enable_thinking: true`              | ✅ Force thinking ON                                   |
| `enable_thinking: false`             | ❌ Disable thinking                                    |
| Message contains `/think`            | ✅ Override to enable thinking                         |
| Message contains `/no_think`         | ❌ Override to disable thinking                        |

## Examples

### Complex Math (Thinking Auto-Enabled)

```python
# Thinking will be used automatically
response = requests.post("/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "What is 125 * 8?"}]
})
# Response includes reasoning steps automatically!
```

### Simple Query (Still Works Fine)

```python
# Even with thinking enabled, simple queries stay simple
response = requests.post("/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})
# Response: "The capital of France is Paris."
```

### Explicitly Disable When Not Needed

```python
# For maximum speed on simple tasks, you can disable thinking
response = requests.post("/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "enable_thinking": false  # Skip thinking for greetings
})
```

## Model Detection

The system automatically detects if a model supports thinking:

```python
# Check model capabilities
response = requests.get("http://localhost:8000/model/info")
info = response.json()
print(f"Supports thinking: {info['supports_thinking']}")
# For Qwen3: True
# For other models: False
```

## Testing

Verify the auto-enable behavior:

```bash
# Run the auto-thinking test
python examples/test_auto_thinking.py
```

This test will show you:

- ✅ Thinking is ON by default for complex queries
- ✅ Can be disabled when needed
- ✅ Control tags still work
- ✅ Backwards compatible

## Benefits

1. **Smarter by Default**: Models that can think will think!
2. **Better User Experience**: No configuration needed for optimal results
3. **Performance**: Can still disable for simple queries when speed matters
4. **Flexibility**: Full control when you need it

## Summary

**For Qwen3 and other thinking-capable models:**

- Thinking is ON by default ✅
- Better responses automatically 🎯
- No API changes needed 🔄
- Full control still available 🎛️

**For other models:**

- No change in behavior
- `enable_thinking` is safely ignored

Enjoy smarter, more thoughtful AI responses without the extra configuration!
