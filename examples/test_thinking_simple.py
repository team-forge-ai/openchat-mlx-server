#!/usr/bin/env python3
"""
Test script demonstrating how to use the tokenizer's built-in thinking support
with Qwen3 models rather than manual parsing.
"""

import json
import requests
from typing import Dict, Any, Optional


def test_native_thinking(
    base_url: str = "http://localhost:8000",
    model: str = "qwen3"
):
    """
    Test the native thinking support using tokenizer's built-in capabilities.
    
    The key insight is that Qwen3's tokenizer already knows how to handle
    thinking/reasoning through its chat template, so we don't need to
    manually parse <think> tags - we just need to use the right parameters.
    """
    
    print("Testing Native Thinking Support with Qwen3")
    print("=" * 50)
    
    # Test 1: Complex reasoning task with thinking enabled
    print("\n1. Testing with enable_thinking=True (complex task):")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Count the number of 'r's in 'strawberry' step by step."}
            ],
            "enable_thinking": True,  # Enable thinking in the template
            "include_reasoning": True,  # Include reasoning in response
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Check if reasoning was included
        if result.get("choices", [{}])[0].get("reasoning"):
            print("\n✓ Reasoning content detected and returned!")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test 2: Simple task with thinking disabled
    print("\n2. Testing with enable_thinking=False (simple task):")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "enable_thinking": False,  # Disable thinking
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Check that no reasoning was included
        if not result.get("choices", [{}])[0].get("reasoning"):
            print("\n✓ No reasoning content (as expected for simple task)")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test 3: Using control tags in messages (auto-detection)
    print("\n3. Testing with /think control tag in message:")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "/think\nSolve this math problem: What is 15 * 23?"}
            ],
            # Don't specify enable_thinking - let it auto-detect from /think tag
            "include_reasoning": True,
            "max_tokens": 300,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("choices", [{}])[0].get("reasoning"):
            print("\n✓ Auto-detected thinking from /think tag!")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test 4: Streaming with thinking
    print("\n4. Testing streaming with thinking enabled:")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Explain how photosynthesis works."}
            ],
            "enable_thinking": True,
            "include_reasoning": True,
            "stream": True,
            "max_tokens": 200
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("Streaming response:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data != "[DONE]":
                        chunk = json.loads(data)
                        # Check for reasoning events
                        if chunk.get("choices", [{}])[0].get("reasoning_event"):
                            print(f"  [Reasoning Event]: {chunk['choices'][0]['reasoning_event']}")
                        elif chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                            print(f"  {chunk['choices'][0]['delta']['content']}", end="")
        print("\n\n✓ Streaming with thinking completed!")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def explain_native_approach():
    """
    Explain how the native tokenizer approach works vs manual parsing.
    """
    print("\n" + "=" * 60)
    print("Native Tokenizer Approach vs Manual Parsing")
    print("=" * 60)
    
    explanation = """
    ## Why Use Native Tokenizer Support?
    
    1. **Template-Based Control**: 
       - The tokenizer's chat_template already knows how to handle thinking
       - It uses the `enable_thinking` parameter to control generation
       - No need to manually parse <think> tags from output
    
    2. **Special Token Handling**:
       - Tokenizers have special tokens for <think> and </think>
       - These are handled natively during generation
       - Can be used as stop tokens or generation boundaries
    
    3. **Cleaner Implementation**:
       - Let mlx-lm and the tokenizer handle the complexity
       - Use apply_chat_template() with proper parameters
       - Model generates structured output based on its training
    
    4. **Better Performance**:
       - No regex parsing overhead
       - Native token handling is more efficient
       - Streaming can handle reasoning boundaries properly
    
    ## How It Works:
    
    ```python
    # The tokenizer applies the template with thinking support
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # ← Key parameter for Qwen3
    )
    
    # Generate with mlx-lm
    output = mlx_generate(model, tokenizer, prompt, ...)
    
    # Tokenizer can decode with special token handling
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    ```
    
    This approach leverages the model's training and the tokenizer's
    built-in understanding of thinking/reasoning patterns.
    """
    
    print(explanation)


if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Server is not responding. Please start the server first.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server at http://localhost:8000")
        print("Please start the server with: max serve --enable-thinking")
        sys.exit(1)
    
    # Run tests
    test_native_thinking()
    explain_native_approach()