"""
Simplified generation engine for MLX inference.

This module provides a thin wrapper around mlx_lm's generation functions,
with support for thinking/reasoning extraction when available.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Tuple

import mlx.core as mx
from mlx_lm import generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from .api_models import ReasoningItem
from .thinking import ThinkingExtractor, StreamingThinkingProcessor
from .utils import generate_id

logger = logging.getLogger(__name__)


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
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.8,  # Updated default for non-thinking mode
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        include_reasoning: bool = True,
        # Additional sampling parameters from mlx_lm
        top_k: int = 20,  # Updated default
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
        
        # Apply Qwen3 recommended parameters based on thinking mode
        # Default to thinking mode for Qwen3 models (enable_thinking=None -> True)
        actual_enable_thinking = enable_thinking
        if actual_enable_thinking is None:
            # Check for explicit control tags that override the default
            if isinstance(messages, list):
                message_text = " ".join([msg.get("content", "") for msg in messages])
                if "/no_think" in message_text:
                    actual_enable_thinking = False
                else:
                    actual_enable_thinking = True  # Default to thinking mode for capable models
            else:
                actual_enable_thinking = True  # Default to thinking mode
        
        # Apply mode-specific parameter defaults if not explicitly overridden
        final_temperature = temperature
        final_top_p = top_p
        final_top_k = top_k
        final_min_p = min_p
        
        # Only apply defaults if values appear to be defaults (not explicitly set by user)
        if actual_enable_thinking:
            # Thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
            if temperature == 0.7:  # Default temperature, apply thinking recommendation
                final_temperature = 0.6
            if top_p in [0.8, 1.0]:  # Default top_p values, apply thinking recommendation  
                final_top_p = 0.95
            if top_k in [0, 20]:  # Default top_k values, apply thinking recommendation
                final_top_k = 20
            # min_p=0.0 is already correct for thinking mode
        else:
            # Non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
            # For non-thinking mode, the current defaults are already correct:
            # temperature=0.7, top_p=0.8, top_k=20, min_p=0.0
            pass  # No overrides needed for non-thinking mode with new defaults
        
        logger.debug(f"Generation parameters - Thinking: {actual_enable_thinking}, "
                    f"Temperature: {final_temperature}, TopP: {final_top_p}, "
                    f"TopK: {final_top_k}, MinP: {final_min_p}, MaxTokens: {max_tokens}")
        
        # Create sampler with all parameters
        sampler = make_sampler(
            temp=final_temperature,
            top_p=final_top_p,
            min_p=final_min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=final_top_k
        )
        
        # Create logits processors for repetition penalty
        logits_processors = None
        if repetition_penalty and repetition_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size
            )
        
        try:
            if stream:
                return self._generate_stream(
                    model, tokenizer, prompt,
                    max_tokens, sampler, logits_processors,
                    include_reasoning, **kwargs
                )
            else:
                # Generate complete text
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
        sampler: Any,
        logits_processors: Any,
        include_reasoning: bool,
        **kwargs
    ) -> Iterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """
        Stream generation using mlx_lm's stream_generate.
        
        Yields tuples of (text_chunk, reasoning_event).
        """
        # Create streaming processor if reasoning is enabled
        processor = None
        if include_reasoning and self.thinking_extractor:
            processor = StreamingThinkingProcessor(self.thinking_extractor)
            processor.session_id = generate_id("stream")
        
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

            logger.debug(f"[STREAM] {repr(text_chunk)}")
            
            # Process through thinking processor if available
            if processor:
                text_chunk, reasoning_event = processor.process_chunk(text_chunk)
                if text_chunk or reasoning_event:
                    yield text_chunk, reasoning_event
            else:
                # Direct streaming without thinking processing
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