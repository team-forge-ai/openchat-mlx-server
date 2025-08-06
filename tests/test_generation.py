"""Tests for the refactored Generation Engine."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict

from openchat_mlx_server.generation import GenerationEngine, ThinkingExtractor


@pytest.fixture
def generation_engine():
    """Create generation engine instance."""
    return GenerationEngine()


@pytest.fixture
def mock_model():
    """Create mock model."""
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.get_vocab.return_value = {}
    return tokenizer


class TestThinkingExtractor:
    """Test ThinkingExtractor class."""
    
    def test_extract_with_complete_thinking_tags(self):
        """Test extraction with complete thinking tags."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        extractor = ThinkingExtractor(tokenizer)
        
        text = "Before <think>This is thinking content</think> After"
        content, thinking = extractor.extract(text)
        
        assert content == "Before After"
        assert thinking == "This is thinking content"
    
    def test_extract_with_incomplete_thinking_tags(self):
        """Test extraction with incomplete thinking tags."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        extractor = ThinkingExtractor(tokenizer)
        
        text = "Before <think>This is incomplete thinking content"
        content, thinking = extractor.extract(text)
        
        assert content == "Before"
        assert thinking == "This is incomplete thinking content"
    
    def test_extract_with_no_thinking_tags(self):
        """Test extraction with no thinking tags."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        extractor = ThinkingExtractor(tokenizer)
        
        text = "Just regular text without thinking"
        content, thinking = extractor.extract(text)
        
        assert content == "Just regular text without thinking"
        assert thinking is None
    
    def test_extract_with_multiple_thinking_blocks(self):
        """Test extraction with multiple thinking blocks."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        extractor = ThinkingExtractor(tokenizer)
        
        text = "Start <think>First block</think> Middle <think>Second block</think> End"
        content, thinking = extractor.extract(text)
        
        assert content == "Start Middle End"
        assert thinking == "First block\n\nSecond block"


class TestGenerationEngine:
    """Test Generation Engine."""
    
    def test_initialization(self, generation_engine):
        """Test generation engine initialization."""
        assert generation_engine is not None
        assert generation_engine.thinking_extractor is None
    
    def test_set_thinking_extractor(self, generation_engine, mock_tokenizer):
        """Test setting thinking extractor."""
        generation_engine.set_thinking_extractor(mock_tokenizer)
        assert generation_engine.thinking_extractor is not None
        assert isinstance(generation_engine.thinking_extractor, ThinkingExtractor)
    
    @patch('openchat_mlx_server.generation.generate')
    def test_generate_complete_success(self, mock_generate, generation_engine, mock_model, mock_tokenizer):
        """Test complete generation success."""
        mock_generate.return_value = "This is a test response"
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = generation_engine.generate(
            mock_model, mock_tokenizer, messages,
            max_tokens=50, temperature=0.7, stream=False,
            include_reasoning=False  # Disable reasoning extraction for this test
        )
        
        assert result == "This is a test response"
        mock_generate.assert_called_once()
    
    @patch('openchat_mlx_server.generation.generate')
    def test_generate_with_thinking_extraction(self, mock_generate, generation_engine, mock_model, mock_tokenizer):
        """Test generation with thinking extraction."""
        mock_generate.return_value = "<think>Internal thoughts</think>Actual response"
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = generation_engine.generate(
            mock_model, mock_tokenizer, messages,
            max_tokens=50, temperature=0.7, stream=False,
            include_reasoning=True
        )
        
        # Should return tuple with content and reasoning item
        if isinstance(result, tuple):
            content, reasoning = result
            assert content == "Actual response"
            assert reasoning.content == "Internal thoughts"
        else:
            # If thinking wasn't extracted, just check the result
            assert "<think>" not in result or "</think>" not in result
    
    @patch('openchat_mlx_server.generation.stream_generate')
    def test_generate_streaming(self, mock_stream_generate, generation_engine, mock_model, mock_tokenizer):
        """Test streaming generation."""
        # Mock stream_generate response
        mock_response = MagicMock()
        mock_response.text = "chunk"
        mock_stream_generate.return_value = [mock_response]
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = generation_engine.generate(
            mock_model, mock_tokenizer, messages,
            max_tokens=50, stream=True
        )
        
        # Streaming should return an iterator
        assert hasattr(result, '__iter__')
        
        # Consume the iterator
        chunks = list(result)
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_generate_async_non_streaming(self, generation_engine, mock_model, mock_tokenizer):
        """Test async generation non-streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(generation_engine, 'generate') as mock_generate:
            mock_generate.return_value = "Test response"
            
            # Collect results from async generator
            results = []
            async for chunk in generation_engine.generate_async(
                mock_model, mock_tokenizer, messages,
                max_tokens=50, stream=False
            ):
                results.append(chunk)
            
            assert len(results) == 1
            assert results[0] == "Test response"
    
    @pytest.mark.asyncio
    async def test_generate_async_streaming(self, generation_engine, mock_model, mock_tokenizer):
        """Test async generation streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(generation_engine, 'generate') as mock_generate:
            # Mock generator for streaming
            def mock_gen():
                yield ("chunk1", None)
                yield ("chunk2", None)
                yield ("chunk3", None)
            
            mock_generate.return_value = mock_gen()
            
            # Collect results from async generator
            results = []
            async for chunk in generation_engine.generate_async(
                mock_model, mock_tokenizer, messages,
                max_tokens=50, stream=True
            ):
                results.append(chunk)
            
            assert len(results) == 3
    
    def test_count_tokens_success(self, generation_engine, mock_tokenizer):
        """Test token counting success."""
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        count = generation_engine.count_tokens(mock_tokenizer, "test text")
        
        assert count == 5
        mock_tokenizer.encode.assert_called_once_with("test text")
    
    def test_count_tokens_fallback(self, generation_engine):
        """Test token counting fallback when encode fails."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = Exception("Encode failed")
        
        count = generation_engine.count_tokens(tokenizer, "test text")
        
        # Should fallback to character count / 4
        assert count == len("test text") // 4
    



class TestErrorHandling:
    """Test error handling in generation."""
    
    @patch('openchat_mlx_server.generation.generate')
    def test_generate_complete_exception(self, mock_generate, generation_engine, mock_model, mock_tokenizer):
        """Test exception handling in complete generation."""
        mock_generate.side_effect = Exception("Generation failed")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="Generation failed"):
            generation_engine.generate(
                mock_model, mock_tokenizer, messages,
                max_tokens=50, stream=False
            )


if __name__ == "__main__":
    pytest.main([__file__])