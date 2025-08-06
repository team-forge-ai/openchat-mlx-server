#!/usr/bin/env python3
"""
Test script to verify that thinking is automatically enabled for supported models.

This demonstrates that you don't need to explicitly set enable_thinking=True
for models like Qwen3 that support thinking - it's on by default!
"""

import json
import requests
from typing import Dict, Any


def test_default_thinking_behavior(
    base_url: str = "http://localhost:8000",
    model: str = "qwen3"
):
    """
    Test that thinking is enabled by default for supported models.
    """
    print("Testing Default Thinking Behavior")
    print("=" * 60)
    print("\nFor models that support thinking (like Qwen3), thinking should")
    print("be ENABLED by default unless explicitly disabled.\n")
    
    test_cases = [
        {
            "name": "Default behavior (no enable_thinking specified)",
            "request": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is 25 * 13? Show your work."}
                ],
                # NOTE: Not specifying enable_thinking at all
                "include_reasoning": True,
                "max_tokens": 300,
                "temperature": 0.7
            },
            "expected": "Should include thinking/reasoning"
        },
        {
            "name": "Explicitly enabled (enable_thinking=True)",
            "request": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is 18 + 27? Think step by step."}
                ],
                "enable_thinking": True,  # Explicitly enabled
                "include_reasoning": True,
                "max_tokens": 300,
                "temperature": 0.7
            },
            "expected": "Should include thinking/reasoning"
        },
        {
            "name": "Explicitly disabled (enable_thinking=False)",
            "request": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is 10 - 3? Calculate this."}
                ],
                "enable_thinking": False,  # Explicitly disabled
                "include_reasoning": True,
                "max_tokens": 200,
                "temperature": 0.7
            },
            "expected": "Should NOT include thinking/reasoning"
        },
        {
            "name": "Control tag override (/no_think in message)",
            "request": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "/no_think\nWhat is 2 + 2?"}
                ],
                # Not specifying enable_thinking, but /no_think should override
                "include_reasoning": True,
                "max_tokens": 100,
                "temperature": 0.7
            },
            "expected": "Should NOT include thinking (overridden by /no_think)"
        },
        {
            "name": "Control tag force enable (/think in message)",
            "request": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "/think\nWhat is the capital of France?"}
                ],
                "enable_thinking": False,  # Explicitly disabled but /think should override
                "include_reasoning": True,
                "max_tokens": 200,
                "temperature": 0.7
            },
            "expected": "Might include thinking (control tag vs explicit setting)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        print(f"Expected: {test_case['expected']}")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_case["request"]
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for reasoning in response
            has_reasoning = False
            reasoning_content = None
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "reasoning" in choice and choice["reasoning"]:
                    has_reasoning = True
                    reasoning_content = choice["reasoning"].get("content", "")
            
            # Display results
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content[:100]}..." if len(content) > 100 else f"Response: {content}")
            
            if has_reasoning:
                print(f"✅ Reasoning found: {reasoning_content[:100]}..." if len(reasoning_content) > 100 else f"✅ Reasoning found: {reasoning_content}")
            else:
                print("❌ No reasoning found")
            
            # Check if result matches expectation
            if "NOT" in test_case["expected"]:
                if not has_reasoning:
                    print("✓ Result matches expectation (no thinking)")
                else:
                    print("⚠️ Unexpected: Found reasoning when it should be disabled")
            else:
                if has_reasoning:
                    print("✓ Result matches expectation (thinking enabled)")
                else:
                    print("⚠️ Unexpected: No reasoning found when it should be enabled")
        else:
            print(f"Error: {response.status_code} - {response.text}")


def print_summary():
    """
    Print a summary of the default thinking behavior.
    """
    print("\n" + "=" * 60)
    print("Summary: Default Thinking Behavior")
    print("=" * 60)
    
    summary = """
    For models that support thinking (like Qwen3):
    
    1. **Default (enable_thinking=None)**: 
       → Thinking is ENABLED automatically
       
    2. **Explicit True (enable_thinking=True)**:
       → Thinking is ENABLED
       
    3. **Explicit False (enable_thinking=False)**:
       → Thinking is DISABLED
       
    4. **Control Tags in Messages**:
       - `/think` → Forces thinking ON (overrides settings)
       - `/no_think` → Forces thinking OFF (overrides settings)
    
    Benefits of Auto-Enable:
    - Better responses by default for complex queries
    - No need to remember to enable thinking
    - Can still disable when not needed
    - Backwards compatible (can override)
    
    For models that DON'T support thinking:
    - enable_thinking has no effect
    - System gracefully ignores the parameter
    """
    
    print(summary)


def verify_model_info(base_url: str = "http://localhost:8000"):
    """
    Check if the loaded model supports thinking.
    """
    print("\nChecking Model Capabilities")
    print("-" * 40)
    
    response = requests.get(f"{base_url}/model/info")
    if response.status_code == 200:
        info = response.json()
        supports_thinking = info.get("supports_thinking", False)
        model_path = info.get("path", "unknown")
        
        print(f"Model: {model_path}")
        print(f"Supports Thinking: {'✅ Yes' if supports_thinking else '❌ No'}")
        
        if supports_thinking:
            print("\n→ This model will have thinking ENABLED by default!")
        else:
            print("\n→ This model does not support thinking.")
        
        return supports_thinking
    else:
        print(f"Could not get model info: {response.status_code}")
        return None


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
        print("Please start the server with a Qwen3 model")
        sys.exit(1)
    
    # Check model capabilities
    supports_thinking = verify_model_info()
    
    if supports_thinking:
        # Run tests
        test_default_thinking_behavior()
        print_summary()
    else:
        print("\n⚠️ The loaded model doesn't support thinking.")
        print("To test thinking features, please load a Qwen3 model.")
        print("\nExample:")
        print("  python -m openchat_mlx_server.main --model-path models/Qwen3-0.6B-MLX-4bit")