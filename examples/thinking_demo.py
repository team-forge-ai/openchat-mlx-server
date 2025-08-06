#!/usr/bin/env python3
"""
Demo script showing thinking/reasoning functionality with MLX models.

This demonstrates how models like Qwen3 can use thinking to provide
better, more thoughtful responses to complex queries.
"""

import json
import requests
from typing import Dict, Any


def test_thinking_api(base_url: str = "http://localhost:8000"):
    """
    Demonstrate the thinking/reasoning API functionality.
    """
    print("=" * 60)
    print("MLX Thinking/Reasoning Demo")
    print("=" * 60)
    
    # Check server health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server is not healthy")
            return
        print("✅ Server is running")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server at", base_url)
        print("Please start the server first:")
        print("  python -m openchat_mlx_server.main --model-path <your-model>")
        return
    
    # Check model capabilities
    response = requests.get(f"{base_url}/model/info")
    if response.status_code == 200:
        info = response.json()
        print(f"\n📦 Model: {info.get('path', 'Unknown')}")
        print(f"🧠 Supports Thinking: {info.get('supports_thinking', False)}")
        print(f"🎯 Capability Level: {info.get('thinking_capability', 'none')}")
    
    print("\n" + "-" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Complex Math Problem",
            "messages": [
                {
                    "role": "user",
                    "content": "A train travels 120 km in 2 hours, then 180 km in 3 hours. What is the average speed?"
                }
            ],
            "description": "Should trigger step-by-step reasoning"
        },
        {
            "name": "Simple Query",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            "description": "Simple factual answer, minimal thinking"
        },
        {
            "name": "Logic Puzzle",
            "messages": [
                {
                    "role": "user",
                    "content": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?"
                }
            ],
            "description": "Requires logical reasoning"
        },
        {
            "name": "Forced Thinking with /think",
            "messages": [
                {
                    "role": "user",
                    "content": "/think\nWhat is 2 + 2?"
                }
            ],
            "description": "Control tag forces thinking even for simple query"
        },
        {
            "name": "Disabled Thinking with /no_think",
            "messages": [
                {
                    "role": "user",
                    "content": "/no_think\nExplain quantum computing"
                }
            ],
            "description": "Control tag disables thinking for complex topic"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['name']}")
        print(f"   {test['description']}")
        print("-" * 40)
        
        # Make request
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": test["messages"],
                "include_reasoning": True,  # Include reasoning in response
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract response
            choice = result["choices"][0]
            content = choice["message"]["content"]
            reasoning = choice.get("reasoning")
            
            # Display results
            if reasoning and reasoning.get("content"):
                print("💭 Thinking:")
                thinking_preview = reasoning["content"][:200]
                if len(reasoning["content"]) > 200:
                    thinking_preview += "..."
                print(f"   {thinking_preview}")
                print()
            
            print("💬 Response:")
            response_preview = content[:200]
            if len(content) > 200:
                response_preview += "..."
            print(f"   {response_preview}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nKey Features:")
    print("• Thinking is AUTO-ENABLED for capable models")
    print("• Use /think to force thinking ON")
    print("• Use /no_think to force thinking OFF")
    print("• Complex queries automatically trigger deeper reasoning")
    print("• Simple queries stay fast and direct")


def test_streaming_with_thinking(base_url: str = "http://localhost:8000"):
    """
    Demonstrate streaming with thinking/reasoning events.
    """
    print("\n" + "=" * 60)
    print("Streaming with Thinking Demo")
    print("=" * 60)
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Explain how photosynthesis works in simple terms."
                }
            ],
            "stream": True,
            "include_reasoning": True,
            "max_tokens": 300
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("\nStreaming response:")
        print("-" * 40)
        
        reasoning_content = []
        response_content = []
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            choice = choices[0]
                            
                            # Check for reasoning events
                            if "reasoning_event" in choice:
                                event = choice["reasoning_event"]
                                if event and event.get("type") == "start":
                                    print("💭 [Thinking started...]")
                                elif event and event.get("type") == "complete":
                                    print("💭 [Thinking complete]")
                                    if event.get("content"):
                                        reasoning_content.append(event["content"])
                            
                            # Check for content
                            delta = choice.get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                response_content.append(content)
                                print(content, end="", flush=True)
                    
                    except json.JSONDecodeError:
                        pass
        
        print("\n" + "-" * 40)
        
        if reasoning_content:
            print("\n💭 Thinking process:")
            print("   " + "".join(reasoning_content)[:300] + "...")
        
        print("\n✅ Streaming complete!")
    else:
        print(f"❌ Error: {response.status_code}")


def explain_thinking_system():
    """
    Explain how the thinking system works.
    """
    print("\n" + "=" * 60)
    print("How Thinking/Reasoning Works")
    print("=" * 60)
    
    explanation = """
    The MLX server now includes intelligent thinking/reasoning support:
    
    1. **Automatic Detection**
       - System detects if your model supports thinking (e.g., Qwen3)
       - Thinking is ENABLED by default for capable models
    
    2. **Smart Processing**
       - Complex queries → Deep reasoning
       - Simple queries → Quick responses
       - No manual configuration needed
    
    3. **Control Options**
       - Let the system decide (default)
       - Force ON with /think in your message
       - Force OFF with /no_think
       - API parameter: enable_thinking
    
    4. **Response Format**
       - Main response: The actual answer
       - Reasoning: The thinking process (when available)
       - Clean separation for easy processing
    
    5. **Model Support**
       - Qwen3: Full thinking with <think> tags
       - Other models: Graceful degradation
       - Extensible for future models
    
    The system uses a unified ThinkingManager that:
    - Detects model capabilities
    - Applies appropriate templates
    - Extracts reasoning content
    - Handles streaming events
    - Manages special tokens
    """
    
    print(explanation)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🌐 Using server at: {base_url}\n")
    
    # Run demos
    test_thinking_api(base_url)
    print("\n" + "=" * 60 + "\n")
    test_streaming_with_thinking(base_url)
    explain_thinking_system()
    
    print("\n✨ All demos complete!")