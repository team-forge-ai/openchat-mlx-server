#!/usr/bin/env python3
"""
Example using the official OpenAI Python client with MLX Engine Server.

This demonstrates how to use the OpenAI client library to connect to 
a local MLX Engine Server running on localhost:8000.
"""

import os
from openai import OpenAI
import argparse
import time


def test_chat_completion(client: OpenAI, model: str = "default"):
    """
    Test basic chat completion functionality.
    """
    print("🔍 Testing Chat Completion...")
    print("-" * 40)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan? Keep it brief."}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"✅ Response received:")
        print(f"   Model: {response.model}")
        print(f"   Content: {response.choices[0].message.content}")
        print(f"   Finish reason: {response.choices[0].finish_reason}")
        
        if response.usage:
            print(f"   Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False


def test_streaming(client: OpenAI, model: str = "default"):
    """
    Test streaming chat completion functionality.
    """
    print("\n🔍 Testing Streaming...")
    print("-" * 40)
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 5 and briefly explain what each number represents."}
            ],
            max_tokens=200,
            temperature=0.7,
            stream=True
        )
        
        print("✅ Streaming response:")
        print("   ", end="")
        
        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        
        print("\n")
        print(f"   Total content length: {len(full_content)} characters")
        print(f"   Stream completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        return False


def test_tool_calling_and_logprobs(client: OpenAI, model: str = "default"):
    """
    Demonstrate tool calling markers and logprobs support from mlx_lm.server.
    """
    print("\n🔍 Testing Tool Calling and Logprobs...")
    print("-" * 40)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Return a JSON object with name='add' and arguments {\"a\":2,\"b\":3}"}
            ],
            max_tokens=64,
            temperature=0.2,
            logprobs=5
        )

        print("✅ Response received")
        print(f"   Content: {response.choices[0].message.content}")
        if getattr(response.choices[0], 'logprobs', None):
            print("   ℹ️ logprobs present")
        return True

    except Exception as e:
        print(f"❌ Tool/logprobs test failed: {e}")
        return False


def list_models(client: OpenAI):
    """
    List available models on the server.
    """
    print("🔍 Listing Available Models...")
    print("-" * 40)
    
    try:
        models = client.models.list()
        
        if models.data:
            print("✅ Available models:")
            for model in models.data:
                print(f"   - {model.id}")
            return models.data[0].id  # Return first model ID
        else:
            print("⚠️ No models available")
            return "default"
            
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return "default"


def check_server_health(client: OpenAI):
    """
    Check if the server is responding.
    """
    print("🔍 Checking Server Health...")
    print("-" * 40)
    
    try:
        # Try to list models as a health check
        models = client.models.list()
        print("✅ Server is responding")
        return True
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False


def main():
    """
    Main function to run all tests.
    """
    parser = argparse.ArgumentParser(
        description="OpenAI Client Example for MLX Engine Server"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL for the MLX Engine Server OpenAI API"
    )
    parser.add_argument(
        "--api-key",
        default="dummy-key",
        help="API key (MLX server doesn't require authentication, but OpenAI client needs something)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OpenAI Client Example for MLX Engine Server")
    print("=" * 60)
    print(f"🌐 Base URL: {args.base_url}")
    print(f"🔑 API Key: {args.api_key}")
    
    # Create OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # Check server health
    if not check_server_health(client):
        print("\n❌ Cannot connect to server. Please ensure the MLX Engine Server is running.")
        print("   Start it with: python -m openchat_mlx_server.main --model-path <your-model>")
        return 1
    
    # List available models
    model_id = list_models(client)
    print(f"\n🎯 Using model: {model_id}")
    
    # Run tests
    print("\n" + "=" * 60)
    print("Running Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic chat completion
    if test_chat_completion(client, model_id):
        success_count += 1
    
    # Test 2: Streaming
    if test_streaming(client, model_id):
        success_count += 1
    
    # Test 3: Tool calling and logprobs
    if test_tool_calling_and_logprobs(client, model_id):
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 All tests passed! OpenAI client integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check server logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())