#!/usr/bin/env python3
"""
Test script demonstrating comprehensive Qwen3 special token support.

This shows how the system handles all Qwen3 special tokens including:
- <|endoftext|> for document boundaries
- <|im_start|> and <|im_end|> for message boundaries  
- <think> and </think> for reasoning
- Tool calling tokens
- And more...
"""

import json
import requests
from typing import Dict, Any, List


def demonstrate_special_tokens():
    """
    Demonstrate how different Qwen3 special tokens are handled.
    """
    print("Qwen3 Special Token Support Demo")
    print("=" * 60)
    
    print("\nQwen3 uses these special tokens:")
    print("-" * 40)
    
    tokens = {
        "Message Boundaries": {
            "<|im_start|>": "Marks the start of a message",
            "<|im_end|>": "Marks the end of a message (also EOS)",
            "<|endoftext|>": "Document end/padding token"
        },
        "Thinking/Reasoning": {
            "<think>": "Start of thinking/reasoning section",
            "</think>": "End of thinking/reasoning section"
        },
        "Tool Calling": {
            "<tool_call>": "Start of a tool call",
            "</tool_call>": "End of a tool call",
            "<tool_response>": "Start of tool response",
            "</tool_response>": "End of tool response"
        },
        "Vision/Multimodal": {
            "<|vision_start|>": "Start of vision content",
            "<|vision_end|>": "End of vision content",
            "<|image_pad|>": "Image padding token",
            "<|video_pad|>": "Video padding token"
        }
    }
    
    for category, token_dict in tokens.items():
        print(f"\n{category}:")
        for token, description in token_dict.items():
            print(f"  {token:20} - {description}")


def test_thinking_with_boundaries(
    base_url: str = "http://localhost:8000",
    model: str = "qwen3"
):
    """
    Test how thinking content is handled with proper message boundaries.
    """
    print("\n\nTesting Thinking with Message Boundaries")
    print("=" * 60)
    
    # Test complex reasoning that should trigger thinking
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that thinks step by step."
                },
                {
                    "role": "user",
                    "content": "Solve this problem: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?"
                }
            ],
            "enable_thinking": True,
            "include_reasoning": True,
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nResponse Structure:")
        
        # Main content
        content = result['choices'][0]['message']['content']
        print(f"\nMain Content:\n{content}")
        
        # Check for reasoning
        if 'reasoning' in result['choices'][0]:
            reasoning = result['choices'][0]['reasoning']
            if reasoning:
                print(f"\nReasoning Item:")
                print(f"  ID: {reasoning.get('id', 'N/A')}")
                print(f"  Type: {reasoning.get('type', 'N/A')}")
                print(f"  Content: {reasoning.get('content', '')[:200]}...")
            else:
                print(f"\nReasoning: None (thinking content in main response)")
            
        print("\nHow it works internally:")
        print("-" * 40)
        print("1. User message formatted with <|im_start|>user and <|im_end|>")
        print("2. Model generates with <|im_start|>assistant")
        print("3. Thinking content in <think>...</think> tags")
        print("4. Main response content")
        print("5. Ends with <|im_end|>")
        print("6. System extracts and separates thinking from main content")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_stop_token_behavior(
    base_url: str = "http://localhost:8000",
    model: str = "qwen3"
):
    """
    Test how stop tokens affect generation.
    """
    print("\n\nTesting Stop Token Behavior")
    print("=" * 60)
    
    # Test with custom stop sequences
    test_cases = [
        {
            "name": "Normal generation (stops at <|im_end|>)",
            "message": "Write a haiku about coding.",
            "custom_stops": None
        },
        {
            "name": "Stop at thinking boundary",
            "message": "Explain quantum computing in simple terms.",
            "custom_stops": ["</think>"]
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 40)
        
        request_data = {
            "model": model,
            "messages": [
                {"role": "user", "content": test['message']}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        if test['custom_stops']:
            request_data['stop'] = test['custom_stops']
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            finish_reason = result['choices'][0].get('finish_reason', 'unknown')
            
            print(f"Content: {content[:150]}...")
            print(f"Finish Reason: {finish_reason}")
            
            if test['custom_stops']:
                print(f"Custom Stop Tokens: {test['custom_stops']}")
        else:
            print(f"Error: {response.status_code}")


def test_streaming_with_special_tokens(
    base_url: str = "http://localhost:8000",
    model: str = "qwen3"
):
    """
    Test streaming behavior with special tokens.
    """
    print("\n\nTesting Streaming with Special Tokens")
    print("=" * 60)
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Count from 1 to 5 and explain why you're doing it."}
            ],
            "enable_thinking": True,
            "stream": True,
            "max_tokens": 300
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("\nStreaming chunks:")
        print("-" * 40)
        
        thinking_chunks = []
        content_chunks = []
        in_thinking = False
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        delta = chunk['choices'][0].get('delta', {})
                        
                        # Check for reasoning events
                        if 'reasoning_event' in chunk['choices'][0]:
                            event = chunk['choices'][0]['reasoning_event']
                            print(f"[Reasoning Event]: {event}")
                            in_thinking = True
                        
                        # Collect content
                        if 'content' in delta:
                            content = delta['content']
                            if in_thinking:
                                thinking_chunks.append(content)
                            else:
                                content_chunks.append(content)
                            
                            # Check for thinking boundaries in content
                            if '</think>' in content:
                                in_thinking = False
                            elif '<think>' in content:
                                in_thinking = True
                    except json.JSONDecodeError:
                        pass
        
        print(f"\nThinking chunks collected: {len(thinking_chunks)}")
        print(f"Content chunks collected: {len(content_chunks)}")
        
        if thinking_chunks:
            print(f"\nThinking content: {''.join(thinking_chunks)[:100]}...")
        if content_chunks:
            print(f"\nMain content: {''.join(content_chunks)[:100]}...")
    else:
        print(f"Error: {response.status_code}")


def explain_token_handling():
    """
    Explain how the implementation handles all these tokens.
    """
    print("\n\n" + "=" * 60)
    print("How Our Implementation Handles Qwen3 Special Tokens")
    print("=" * 60)
    
    explanation = """
    1. **Token Detection (qwen3_tokens.py)**:
       - Qwen3SpecialTokens class defines all special tokens
       - Qwen3TokenHandler manages token operations
       - Automatically detects token IDs from tokenizer
    
    2. **Generation Control**:
       - Stop tokens: <|im_end|>, <|endoftext|>, </think>
       - Generation boundaries for structured output
       - Configurable based on use case
    
    3. **Content Extraction**:
       - Separates thinking from main content
       - Extracts tool calls and responses
       - Removes message boundary tokens from output
    
    4. **Chat Template Integration**:
       - Uses tokenizer's native apply_chat_template()
       - Passes enable_thinking parameter when supported
       - Handles message boundaries automatically
    
    5. **Streaming Support**:
       - Can detect thinking boundaries during streaming
       - Sends reasoning events separately
       - Maintains proper token boundaries
    
    6. **OpenAI API Compatibility**:
       - Thinking content → reasoning items
       - Proper finish_reason based on stop tokens
       - Clean content without special tokens
    
    Key Benefits:
    - No manual parsing needed for most cases
    - Leverages tokenizer's built-in understanding
    - Handles all Qwen3 special tokens properly
    - Maintains API compatibility
    """
    
    print(explanation)


if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Server health check failed")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server at http://localhost:8000")
        print("Please start the server with your Qwen3 model")
        sys.exit(1)
    
    # Run demonstrations
    demonstrate_special_tokens()
    test_thinking_with_boundaries()
    test_stop_token_behavior()
    test_streaming_with_special_tokens()
    explain_token_handling()