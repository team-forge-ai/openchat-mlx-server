"""
Comprehensive integration tests for thinking/reasoning functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.openchat_mlx_server.thinking_manager import (
    ThinkingManager,
    ThinkingCapability,
    ThinkingResult,
    ReasoningEvent
)
from src.openchat_mlx_server.generation import GenerationEngine
from src.openchat_mlx_server.api_models import ReasoningItem


class TestThinkingManager:
    """Test suite for ThinkingManager."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with thinking support."""
        tokenizer = Mock()
        tokenizer.chat_template = "{{ enable_thinking }} {{ reasoning_content }}"
        tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
        tokenizer.added_tokens_decoder = {
            151667: {"content": "<think>"},
            151668: {"content": "</think>"}
        }
        return tokenizer
    
    @pytest.fixture
    def mock_qwen3_tokenizer(self):
        """Create a mock Qwen3 tokenizer."""
        tokenizer = Mock()
        tokenizer.__class__.__name__ = "Qwen2Tokenizer"
        tokenizer.chat_template = "{{ enable_thinking }} {{ reasoning_content }}"
        tokenizer.apply_chat_template = Mock(return_value="qwen3 formatted prompt")
        tokenizer.added_tokens_decoder = {
            151644: {"content": "<|im_start|>"},
            151645: {"content": "<|im_end|>"},
            151667: {"content": "<think>"},
            151668: {"content": "</think>"}
        }
        return tokenizer
    
    def test_capability_detection_none(self):
        """Test detection when no tokenizer is provided."""
        manager = ThinkingManager(tokenizer=None)
        assert manager.capability == ThinkingCapability.NONE
        assert not manager.supports_thinking
    
    def test_capability_detection_basic(self):
        """Test detection of basic capability."""
        tokenizer = Mock()
        tokenizer.__class__.__name__ = "BasicTokenizer"
        # Properly mock added_tokens_decoder
        tokenizer.added_tokens_decoder = {}
        tokenizer.chat_template = None
        manager = ThinkingManager(tokenizer=tokenizer)
        assert manager.capability == ThinkingCapability.BASIC
        assert manager.supports_thinking
    
    def test_capability_detection_native(self, mock_tokenizer):
        """Test detection of native thinking support."""
        manager = ThinkingManager(tokenizer=mock_tokenizer)
        assert manager.capability == ThinkingCapability.NATIVE
        assert manager.supports_thinking
    
    def test_capability_detection_advanced(self, mock_qwen3_tokenizer):
        """Test detection of advanced Qwen3 support."""
        manager = ThinkingManager(tokenizer=mock_qwen3_tokenizer)
        assert manager.capability == ThinkingCapability.ADVANCED
        assert manager.supports_thinking
    
    def test_apply_chat_template_auto_enable(self, mock_tokenizer):
        """Test that thinking is auto-enabled for capable models."""
        manager = ThinkingManager(tokenizer=mock_tokenizer)
        messages = [{"role": "user", "content": "Test message"}]
        
        manager.apply_chat_template(messages, enable_thinking=None)
        
        # Should auto-enable thinking
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("enable_thinking") is True
    

    
    def test_extract_reasoning_basic(self):
        """Test basic reasoning extraction."""
        tokenizer = Mock()
        tokenizer.added_tokens_decoder = {}
        tokenizer.chat_template = None
        manager = ThinkingManager(tokenizer=tokenizer)
        manager.capability = ThinkingCapability.BASIC
        
        output = "Let me think. <think>This is my reasoning process.</think> The answer is 42."
        result = manager.extract_reasoning(output)
        
        assert result.content == "Let me think.  The answer is 42."
        assert result.reasoning_content == "This is my reasoning process."
        assert result.has_special_tokens is True
        assert result.capability_used == ThinkingCapability.BASIC
    
    def test_extract_reasoning_no_thinking(self):
        """Test extraction when no thinking is present."""
        tokenizer = Mock()
        tokenizer.added_tokens_decoder = {}
        tokenizer.chat_template = None
        manager = ThinkingManager(tokenizer=tokenizer)
        manager.capability = ThinkingCapability.BASIC
        
        output = "The answer is simply 42."
        result = manager.extract_reasoning(output)
        
        assert result.content == output
        assert result.reasoning_content is None
        assert result.has_special_tokens is False
    
    def test_streaming_processing(self):
        """Test streaming chunk processing."""
        tokenizer = Mock()
        tokenizer.added_tokens_decoder = {}
        tokenizer.chat_template = None
        manager = ThinkingManager(tokenizer=tokenizer)
        
        # Simulate streaming chunks
        chunks = [
            "Let me ",
            "think about this. ",
            "<think>",
            "First, I need to ",
            "understand the problem.",
            "</think>",
            " The answer is ",
            "42."
        ]
        
        results = []
        for chunk in chunks:
            text, event = manager.process_streaming_chunk(chunk, "test_session")
        
            if text:
                results.append(("text", text))
            if event:
                results.append(("event", event.type))
        
        # Check that we got the expected events
        event_types = [r[1] for r in results if r[0] == "event"]
        assert "start" in event_types
        assert "complete" in event_types
        
        # Check that thinking content was captured
        manager.reset_streaming_state("test_session")


class TestGenerationEngine:
    """Test suite for GenerationEngine with thinking support."""
    
    @pytest.fixture
    def engine(self):
        """Create a GenerationEngine instance."""
        return GenerationEngine()
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = Mock()
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="decoded text")
        return model, tokenizer
    
    def test_set_thinking_manager(self, engine):
        """Test setting thinking manager."""
        manager = Mock(spec=ThinkingManager)
        engine.set_thinking_manager(manager)
        assert engine.thinking_manager == manager
    
    @patch('src.openchat_mlx_server.generation.mlx_generate')
    def test_generate_with_thinking(self, mock_mlx_generate, engine, mock_model_and_tokenizer):
        """Test generation with thinking extraction."""
        model, tokenizer = mock_model_and_tokenizer
        mock_mlx_generate.return_value = "<think>My reasoning</think>The answer is 42."
        
        # Set up thinking manager
        manager = Mock(spec=ThinkingManager)
        manager.apply_chat_template = Mock(return_value="formatted prompt")
        manager.extract_reasoning = Mock(return_value=ThinkingResult(
            content="The answer is 42.",
            reasoning_content="My reasoning",
            reasoning_id="reasoning_123",
            has_special_tokens=True,
            capability_used=ThinkingCapability.NATIVE
        ))
        manager.get_generation_config = Mock(return_value={})
        engine.set_thinking_manager(manager)
        
        # Generate with thinking
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = engine.generate(
            model, tokenizer, messages,
            enable_thinking=True,
            include_reasoning=True
        )
        
        # Check result structure
        assert isinstance(result, tuple)
        content, reasoning = result
        assert content == "The answer is 42."
        assert isinstance(reasoning, ReasoningItem)
        assert reasoning.content == "My reasoning"
    
    @patch('src.openchat_mlx_server.generation.mlx_generate')
    def test_generate_without_thinking(self, mock_mlx_generate, engine, mock_model_and_tokenizer):
        """Test generation without thinking extraction."""
        model, tokenizer = mock_model_and_tokenizer
        mock_mlx_generate.return_value = "The answer is 42."
        
        # Generate without thinking manager
        result = engine.generate(
            model, tokenizer, "What is 2+2?",
            enable_thinking=False,
            include_reasoning=False
        )
        
        # Should return plain text
        assert result == "The answer is 42."
    
    def test_count_tokens(self, engine, mock_model_and_tokenizer):
        """Test token counting."""
        model, tokenizer = mock_model_and_tokenizer
        count = engine.count_tokens(tokenizer, "test text")
        assert count == 3  # Based on mock return value
    
    def test_count_tokens_fallback(self, engine):
        """Test token counting fallback when encoding fails."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=Exception("Encoding failed"))
        
        # Should use character-based estimate
        count = engine.count_tokens(tokenizer, "test text")
        assert count == len("test text") // 4
        """Test complete flow from request to response with thinking."""
        # Create components
        tokenizer = Mock()
        tokenizer.chat_template = "{{ enable_thinking }}"
        tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.added_tokens_decoder = {}
        tokenizer.get_vocab = Mock(return_value={})  # Add get_vocab for MLX tokenizer wrapper
        tokenizer.eos_token_id = 0  # Add eos_token_id
        tokenizer.bos_token = None  # Add bos_token
        
        manager = ThinkingManager(tokenizer=tokenizer)
        engine = GenerationEngine()
        engine.set_thinking_manager(manager)
        
        # Mock MLX generation
        with patch('openchat_mlx_server.generation.mlx_generate') as mock_gen:
            mock_gen.return_value = "<think>Step 1: Add 2+2</think>The answer is 4."
            
            model = Mock()
            messages = [{"role": "user", "content": "What is 2+2?"}]
            
            result = engine.generate(
                model, tokenizer, messages,
                enable_thinking=True,
                include_reasoning=True
            )
            
            # Verify the flow
            assert isinstance(result, tuple)
            content, reasoning = result
            assert "4" in content
            assert reasoning is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])