"""
Simplified generation engine for MLX inference.

This module provides a thin wrapper around mlx_lm's generation functions,
with support for thinking/reasoning extraction when available.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Tuple
import re

import mlx.core as mx
from mlx_lm import generate, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .api_models import ReasoningItem
from .utils import generate_id

logger = logging.getLogger(__name__)


class ThinkingExtractor:
    """Simple extractor for thinking content from generated text."""
    
    def __init__(self, tokenizer: Any):
        """
        Initialize the thinking extractor.
        
        Args:
            tokenizer: The tokenizer (already wrapped by mlx_lm if needed)
        """
        self.tokenizer = tokenizer
        
        # Check if tokenizer has thinking support
        self.has_thinking = False
        self.think_start = None
        self.think_end = None
        
        if isinstance(tokenizer, TokenizerWrapper):
            self.has_thinking = tokenizer.has_thinking
            self.think_start = tokenizer.think_start
            self.think_end = tokenizer.think_end
        else:
            # Fallback: check for common thinking tokens
            self._detect_thinking_tokens()
    
    def _detect_thinking_tokens(self):
        """Detect thinking tokens in the tokenizer vocabulary."""
        if not hasattr(self.tokenizer, 'get_vocab'):
            return
            
        vocab = self.tokenizer.get_vocab()
        
        # Common thinking token patterns
        thinking_patterns = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<|thinking|>", "<|/thinking|>"),
        ]
        
        for start, end in thinking_patterns:
            if start in vocab and end in vocab:
                self.has_thinking = True
                self.think_start = start
                self.think_end = end
                logger.debug(f"Detected thinking tokens: {start}, {end}")
                break
    
    def extract(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking content from generated text.
        
        Args:
            text: The generated text
            
        Returns:
            Tuple of (content_without_thinking, thinking_content)
        """
        # Always check for common thinking patterns
        common_patterns = [
            (r"<think>(.*?)</think>", "<think>", "</think>"),
            (r"<thinking>(.*?)</thinking>", "<thinking>", "</thinking>"),
            (r"<\|thinking\|>(.*?)<\|/thinking\|>", "<|thinking|>", "<|/thinking|>"),
        ]
        
        # Use detected patterns if available, otherwise check all common patterns
        if self.has_thinking and self.think_start and self.think_end:
            # Add the detected pattern to the list
            pattern = re.escape(self.think_start) + r"(.*?)" + re.escape(self.think_end)
            common_patterns.insert(0, (pattern, self.think_start, self.think_end))
        
        for pattern, start, end in common_patterns:
            if start in text:
                # Check for complete thinking blocks
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    thinking_content = "\n\n".join(match.strip() for match in matches)
                    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                    return cleaned_text, thinking_content
                
                # Check for incomplete thinking block (no closing tag)
                if end not in text:
                    # Extract everything after the opening tag as thinking
                    parts = text.split(start, 1)
                    if len(parts) == 2:
                        cleaned_text = parts[0].strip()
                        thinking_content = parts[1].strip()
                        return cleaned_text, thinking_content
        
        return text, None


class GenerationEngine:
    """
    Simplified generation engine using MLX-LM directly.
    
    This engine provides:
    - Direct integration with mlx_lm's generate and stream_generate
    - Automatic thinking extraction when supported
    - Async wrapper for compatibility
    """
    
    def __init__(self):
        """Initialize the generation engine."""
        self.thinking_extractor = None
    
    def set_thinking_extractor(self, tokenizer: Any):
        """
        Set up thinking extraction for the current tokenizer.
        
        Args:
            tokenizer: The tokenizer to use for extraction
        """
        self.thinking_extractor = ThinkingExtractor(tokenizer)
    
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        include_reasoning: bool = True,
        # Additional sampling parameters from mlx_lm
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_context_size: int = 20,
        **kwargs
    ) -> Union[str, Iterator[str], Tuple[str, Optional[ReasoningItem]]]:
        """
        Generate text using MLX model.
        
        Args:
            model: MLX model object
            tokenizer: Tokenizer object (will be wrapped by mlx_lm if needed)
            messages: Input messages or prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            stop_sequences: List of stop sequences
            stream: Whether to stream the response
            seed: Random seed for generation
            enable_thinking: Whether to enable thinking (currently handled by tokenizer)
            include_reasoning: Whether to extract and include reasoning
            top_k: Top-k sampling parameter
            min_p: Min-p sampling parameter
            min_tokens_to_keep: Minimum tokens to keep for min-p
            repetition_context_size: Context size for repetition penalty
            **kwargs: Additional parameters passed to mlx_lm
        
        Returns:
            Generated text, iterator of text chunks, or tuple with reasoning
        """
        # Ensure we have a thinking extractor
        if not self.thinking_extractor:
            self.set_thinking_extractor(tokenizer)
        
        # Format messages to prompt
        if isinstance(messages, list):
            # Use tokenizer's apply_chat_template (always available)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = messages
        
        logger.debug(f"Generating with prompt length: {len(prompt)} chars")
        
        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)
        
        # Prepare stop tokens
        stop_tokens = []
        if stop_sequences:
            if isinstance(tokenizer, TokenizerWrapper):
                # Add stop sequences as EOS tokens
                for seq in stop_sequences:
                    tokenizer.add_eos_token(seq)
            else:
                # Will be handled by mlx_lm internally
                stop_tokens = stop_sequences
        
        try:
            if stream:
                return self._generate_stream(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_p,
                    repetition_penalty, stop_tokens,
                    include_reasoning,
                    top_k, min_p, min_tokens_to_keep,
                    repetition_context_size,
                    **kwargs
                )
            else:
                # Use mlx_lm's generate directly
                from mlx_lm.sample_utils import make_sampler, make_logits_processors
                
                # Create sampler with all parameters
                sampler = make_sampler(
                    temp=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    min_tokens_to_keep=min_tokens_to_keep,
                    top_k=top_k
                )
                
                # Create logits processors for repetition penalty
                logits_processors = None
                if repetition_penalty and repetition_penalty != 1.0:
                    logits_processors = make_logits_processors(
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=repetition_context_size
                    )
                
                # Generate text
                generated_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    **kwargs
                )
                
                # Extract reasoning if needed
                if include_reasoning and self.thinking_extractor:
                    content, thinking = self.thinking_extractor.extract(generated_text)
                    if thinking:
                        reasoning_item = ReasoningItem(
                            id=generate_id("reasoning"),
                            content=thinking
                        )
                        return content, reasoning_item
                    return generated_text, None
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    def _generate_stream(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_tokens: List[str],
        include_reasoning: bool,
        top_k: int,
        min_p: float,
        min_tokens_to_keep: int,
        repetition_context_size: int,
        **kwargs
    ) -> Iterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """
        Stream generation using mlx_lm's stream_generate.
        
        Yields tuples of (text_chunk, reasoning_event).
        """
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        
        # Create sampler
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k
        )
        
        # Create logits processors
        logits_processors = None
        if repetition_penalty and repetition_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size
            )
        
        # Track thinking state for streaming
        in_thinking = False
        thinking_buffer = []
        session_id = generate_id("stream")
        
        # Stream using mlx_lm
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            **kwargs
        ):
            text_chunk = response.text
            
            # Simple thinking detection for streaming
            if include_reasoning and self.thinking_extractor and self.thinking_extractor.has_thinking:
                # Check for thinking markers
                if self.thinking_extractor.think_start in text_chunk:
                    in_thinking = True
                    # Send thinking start event
                    yield None, {
                        "type": "start",
                        "id": session_id,
                        "content": None,
                        "partial": False
                    }
                    # Remove thinking start marker from output
                    text_chunk = text_chunk.replace(self.thinking_extractor.think_start, "")
                
                if in_thinking:
                    if self.thinking_extractor.think_end in text_chunk:
                        # End of thinking
                        parts = text_chunk.split(self.thinking_extractor.think_end)
                        thinking_buffer.append(parts[0])
                        
                        # Send thinking complete event
                        yield None, {
                            "type": "complete",
                            "id": session_id,
                            "content": "".join(thinking_buffer),
                            "partial": False
                        }
                        
                        # Reset state
                        in_thinking = False
                        thinking_buffer = []
                        
                        # Continue with remaining text
                        text_chunk = parts[1] if len(parts) > 1 else ""
                    else:
                        # Still in thinking, buffer the content
                        thinking_buffer.append(text_chunk)
                        continue  # Don't yield thinking content as regular text
            
            # Yield the text chunk
            if text_chunk:
                yield text_chunk, None
    
    async def generate_async(
        self,
        model: Any,
        tokenizer: Any,
        messages: Union[str, List[Dict[str, str]]],
        **kwargs
    ) -> AsyncIterator:
        """
        Async wrapper for generation.
        
        Yields results asynchronously for both streaming and non-streaming modes.
        """
        stream = kwargs.get('stream', False)
        
        if stream:
            # Run sync generator in thread for streaming
            def gen():
                return self.generate(
                    model, tokenizer, messages, **kwargs
                )
            
            generator = await asyncio.to_thread(gen)
            for chunk in generator:
                yield chunk
        else:
            # For non-streaming, run in thread and yield once
            result = await asyncio.to_thread(
                self.generate,
                model, tokenizer, messages, **kwargs
            )
            yield result
    
    def count_tokens(self, tokenizer: Any, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            tokenizer: Tokenizer object
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
    
