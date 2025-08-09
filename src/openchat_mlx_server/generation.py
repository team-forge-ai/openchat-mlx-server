"""
Simplified generation engine for MLX inference.

This module provides a thin wrapper around mlx_lm's generation functions,
with support for thinking/reasoning extraction when available.
"""

import asyncio
import logging
import threading
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
    - Thread-safe GPU access to prevent Metal command buffer conflicts
    """
    
    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize the generation engine.
        
        Args:
            model: MLX model object
            tokenizer: Tokenizer object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.thinking_extractor = ThinkingExtractor(tokenizer)
        # Use an asyncio lock for asynchronous, task-safe GPU access
        self._async_gpu_lock = asyncio.Lock()

    def _prepare_prompt(
        self,
        messages: Union[str, List[Dict[str, str]]]
    ) -> str:
        """Formats the input messages into a single prompt string."""
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = messages
        logger.debug(f"Generating with prompt length: {len(prompt)} chars")
        return prompt

    def _prepare_sampler_and_processors(
        self,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        min_tokens_to_keep: int,
        repetition_penalty: float,
        repetition_context_size: int,
        max_tokens: int
    ) -> Tuple[Any, Optional[Any]]:
        """Creates sampler and logits processors for generation."""
        logger.debug(f"Generation parameters - Temperature: {temperature}, TopP: {top_p}, "
                    f"TopK: {top_k}, MinP: {min_p}, MaxTokens: {max_tokens}")
        
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k
        )
        
        logits_processors = None
        if repetition_penalty and repetition_penalty != 1.0:
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size
            )
        
        return sampler, logits_processors

    def _generate_non_stream(
        self,
        prompt: str,
        max_tokens: int,
        sampler: Any,
        logits_processors: Any,
        include_reasoning: bool,
        **kwargs
    ) -> Union[str, Tuple[str, Optional[ReasoningItem]]]:
        """Handles non-streaming generation."""
        # The GPU lock is managed by the calling method (e.g., generate_async)
        generated_text = generate(
            model=self.model,
            tokenizer=self.tokenizer,
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
    
    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_tokens: int = 2000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        include_reasoning: bool = True,
        # Additional sampling parameters from mlx_lm
        top_k: int = 20,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_context_size: int = 20,
        **kwargs
    ) -> Union[str, Iterator[str], Tuple[str, Optional[ReasoningItem]]]:
        """
        Generate text using MLX model.
        
        Args:
            messages: Input messages or prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            stop_sequences: List of stop sequences
            stream: Whether to stream the response
            seed: Random seed for generation
            enable_thinking: Whether to enable thinking (currently unused)
            include_reasoning: Whether to extract and include reasoning
            top_k: Top-k sampling parameter
            min_p: Min-p sampling parameter
            min_tokens_to_keep: Minimum tokens to keep for min-p
            repetition_context_size: Context size for repetition penalty
            **kwargs: Additional parameters passed to mlx_lm
        
        Returns:
            Generated text, iterator of text chunks, or tuple with reasoning
        """
        prompt = self._prepare_prompt(messages)
        
        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)
        
        sampler, logits_processors = self._prepare_sampler_and_processors(
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            min_tokens_to_keep=min_tokens_to_keep,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            max_tokens=max_tokens
        )
        
        try:
            if stream:
                return self._generate_stream(
                    prompt, max_tokens, sampler, logits_processors,
                    include_reasoning, **kwargs
                )
            else:
                return self._generate_non_stream(
                    prompt, max_tokens, sampler, logits_processors,
                    include_reasoning, **kwargs
                )
                
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    def _generate_stream(
        self,
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
        
        try:
            for response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                **kwargs
            ):
                text_chunk = response.text
                
                # Process through thinking processor if available
                if processor:
                    text_chunk, reasoning_event = processor.process_chunk(text_chunk)
                    if text_chunk or reasoning_event:
                        yield text_chunk, reasoning_event
                else:
                    # Direct streaming without thinking processing
                    if text_chunk:
                        yield text_chunk, None
        finally:
            # The lock is released when generate_async completes
            pass
    
    async def generate_async(
        self,
        messages: Union[str, List[Dict[str, str]]],
        **kwargs
    ) -> AsyncIterator:
        """
        Async wrapper for generation, ensuring thread-safe GPU access.
        
        Yields results asynchronously for both streaming and non-streaming modes.
        """
        async with self._async_gpu_lock:
            stream = kwargs.get('stream', False)
            
            if stream:
                # Get the synchronous generator
                sync_generator = self.generate(
                    messages, **kwargs
                )

                # Helper to safely fetch next item without propagating StopIteration
                def _next_item(gen):
                    try:
                        return True, next(gen)
                    except StopIteration:
                        return False, None

                # Iterate over the generator without blocking the event loop
                while True:
                    has_next, chunk = await asyncio.to_thread(_next_item, sync_generator)
                    if not has_next:
                        break
                    yield chunk
            else:
                # For non-streaming, run in thread and yield once
                result = await asyncio.to_thread(
                    self.generate,
                    messages, **kwargs
                )
                yield result
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4