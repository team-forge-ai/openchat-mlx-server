"""Test OpenAI API compatibility using the official OpenAI client."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import json
from pathlib import Path

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from fastapi.testclient import TestClient
from openchat_mlx_server.server import MLXServer
from openchat_mlx_server.config import ServerConfig
from openchat_mlx_server.model_manager import ModelInfo
from openchat_mlx_server.api_models import ReasoningItem


@pytest.fixture
def server_config():
    """Create test server configuration."""
    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        log_level="ERROR",
        model_path="test-model"
    )


@pytest.fixture
def mock_model_info():
    """Create mock model info."""
    model_info = MagicMock(spec=ModelInfo)
    model_info.path = "test-model"
    model_info.type = "test"
    model_info.architecture = "TestModel"
    model_info.loaded_at = "2023-01-01T00:00:00"
    model_info.memory_usage = "1.0 GB"
    model_info.model = MagicMock()
    model_info.tokenizer = MagicMock()
    return model_info


@pytest.fixture
def mlx_server(server_config):
    """Create MLX server instance."""
    return MLXServer(server_config)


@pytest.fixture
def test_client(mlx_server):
    """Create test client."""
    return TestClient(mlx_server.app)


class TestOpenAICompatibility:
    """Test OpenAI API compatibility."""
    
    def test_models_endpoint_structure(self, test_client, mock_model_info):
        """Test that /v1/models returns OpenAI-compatible structure."""
        with patch('openchat_mlx_server.model_manager.MLXModelManager.get_model') as mock_get_model:
            # Return a ModelInfo object as get_model does
            mock_get_model.return_value = mock_model_info
            
            response = test_client.get("/v1/models")
            assert response.status_code == 200
            
            data = response.json()
            
            # Check OpenAI-compatible structure
            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert isinstance(data["data"], list)
            
            if data["data"]:  # If model is loaded
                model = data["data"][0]
                # Check required OpenAI model fields
                assert "id" in model
                assert "object" in model
                assert "created" in model
                assert "owned_by" in model
                assert model["object"] == "model"
    
    def test_chat_completions_request_structure(self, test_client, mock_model_info):
        """Test that chat completions accepts OpenAI-compatible requests."""
        with patch('openchat_mlx_server.model_manager.MLXModelManager.get_model') as mock_get_model:
            with patch('openchat_mlx_server.model_manager.MLXModelManager.format_chat_template') as mock_format_chat:
                with patch('openchat_mlx_server.generation.GenerationEngine.generate_async') as mock_generate:
                    with patch('openchat_mlx_server.generation.GenerationEngine.count_tokens') as mock_count:
                        mock_get_model.return_value = mock_model_info
                        mock_format_chat.return_value = "System: You are a helpful assistant.\nUser: Hello!\nAssistant:"
                        
                        # Mock async generator
                        async def mock_async_gen():
                            yield "Hello there!"
                        
                        mock_generate.return_value = mock_async_gen()
                        mock_count.side_effect = [10, 5]  # prompt_tokens, completion_tokens
                        
                        # OpenAI-compatible request
                        request_data = {
                            "model": "test-model",  # This should be ignored in single-model setup
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "Hello!"}
                            ],
                            "max_tokens": 50,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "frequency_penalty": 0.1,
                            "presence_penalty": 0.1,
                            "stop": ["<|endoftext|>"],
                            "stream": False
                        }
                        
                        response = test_client.post("/v1/chat/completions", json=request_data)
                        assert response.status_code == 200
                        
                        data = response.json()
                        
                        # Check OpenAI-compatible response structure
                        assert "id" in data
                        assert "object" in data
                        assert data["object"] == "chat.completion"
                        assert "created" in data
                        assert "model" in data
                        assert "choices" in data
                        assert "usage" in data
                        
                        # Check choices structure
                        assert len(data["choices"]) == 1
                        choice = data["choices"][0]
                        assert "index" in choice
                        assert "message" in choice
                        assert "finish_reason" in choice
                        
                        # Check message structure
                        message = choice["message"]
                        assert "role" in message
                        assert "content" in message
                        assert message["role"] == "assistant"
                        
                        # Check usage structure
                        usage = data["usage"]
                        assert "prompt_tokens" in usage
                        assert "completion_tokens" in usage
                        assert "total_tokens" in usage
    
    def test_chat_completions_streaming_structure(self, test_client, mock_model_info):
        """Test that streaming responses are OpenAI-compatible."""
        with patch('openchat_mlx_server.model_manager.MLXModelManager.get_model') as mock_get_model:
            mock_get_model.return_value = mock_model_info
            
            request_data = {
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
                "max_tokens": 50
            }
            
            with patch('openchat_mlx_server.server.MLXServer._stream_chat_completion') as mock_stream:
                # Mock streaming response chunks
                mock_chunks = [
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    'data: [DONE]\n\n'
                ]
                mock_stream.return_value = iter(mock_chunks)
                
                response = test_client.post("/v1/chat/completions", json=request_data)
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                
                # Verify response contains proper SSE format
                content = response.text
                assert "data: " in content
                assert "[DONE]" in content
    
    def test_error_responses_openai_compatible(self, test_client):
        """Test that error responses follow OpenAI format."""
        # Test with no model loaded
        request_data = {
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        
        response = test_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 503
        
        data = response.json()
        assert "error" in data
        
        error = data["error"]
        assert "message" in error
        assert "type" in error
        assert "code" in error
    
    def test_validation_errors_openai_compatible(self, test_client):
        """Test that validation errors follow OpenAI format."""
        # Test with invalid request (empty messages)
        request_data = {
            "messages": []
        }
        
        response = test_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422
        
        # Check OpenAI-compatible error structure
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]

