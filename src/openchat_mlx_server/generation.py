"""
Unified generation engine for MLX inference with thinking support.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Tuple
import time

import mlx.core as mx
from mlx_lm import generate as mlx_generate
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .api_models import ChatMessage, ReasoningItem
from .utils import generate_id
from .thinking_manager import ThinkingManager, ThinkingResult, ReasoningEvent

logger = logging.getLogger(__name__)


class GenerationEngine:
    """
    Unified generation engine using MLX-LM with native thinking support.
    
    This engine handles:
    - Text generation with MLX models
    - Thinking/reasoning extraction
    - Streaming responses
    - Token counting
    """
    
    def __init__(self):
        """Initialize the generation engine."""
        self.active_generations = {}
        self._generation_lock = asyncio.Lock()
        self.thinking_manager = None  # Set per-model
    
    def set_thinking_manager(self, thinking_manager: Optional[ThinkingManager]):
        """Set the thinking manager for the current model."""
        self.thinking_manager = thinking_manager
    
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
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_context_size: int = 20,
        **kwargs
    ) -> Union[str, Iterator[str], Tuple[str, Optional[ReasoningItem]]]:
        """
        Generate text using MLX model with thinking support.
        
        Args:
            model: MLX model object
            tokenizer: Tokenizer object
            messages: Input messages or prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            stop_sequences: List of stop sequences
            stream: Whether to stream the response
            seed: Random seed for generation
            enable_thinking: Whether to enable thinking (None = auto)
            include_reasoning: Whether to include reasoning in response
            top_k: Top-k sampling parameter (0 = disabled)
            min_p: Min-p sampling parameter (0.0 = disabled)
            min_tokens_to_keep: Minimum tokens to keep for min-p sampling
            repetition_context_size: Context size for repetition penalty
            **kwargs: Additional MLX-LM parameters
        
        Returns:
            Generated text, iterator of text chunks, or tuple with reasoning item
        """
        # Format messages using thinking manager if available
        if isinstance(messages, list):
            if self.thinking_manager:
                prompt = self.thinking_manager.apply_chat_template(
                    messages, 
                    enable_thinking=enable_thinking,
                    add_generation_prompt=True
                )
            else:
                prompt = self._format_messages_to_prompt(messages, tokenizer)
        else:
            prompt = messages
        
        logger.debug(f"Generating with prompt length: {len(prompt)} chars")
        
        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)
        
        # Get generation config from thinking manager
        if self.thinking_manager and enable_thinking:
            gen_config = self.thinking_manager.get_generation_config(enable_thinking)
            # Add stop tokens from config
            if 'stop_tokens' in gen_config:
                stop_sequences = (stop_sequences or []) + gen_config['stop_tokens']
        
        try:
            if stream:
                return self._generate_stream(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_p,
                    repetition_penalty, stop_sequences,
                    include_reasoning, 
                    top_k=top_k,
                    min_p=min_p,
                    min_tokens_to_keep=min_tokens_to_keep,
                    repetition_context_size=repetition_context_size,
                    **kwargs
                )
            else:
                # Generate complete response
                result = self._generate_complete(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_p,
                    repetition_penalty, stop_sequences,
                    top_k=top_k,
                    min_p=min_p,
                    min_tokens_to_keep=min_tokens_to_keep,
                    repetition_context_size=repetition_context_size,
                    **kwargs
                )
                
                # Extract reasoning if thinking manager is available
                if self.thinking_manager and include_reasoning:
                    thinking_result = self.thinking_manager.extract_reasoning(
                        result, include_reasoning
                    )
                    
                    if thinking_result.reasoning_content:
                        reasoning_item = ReasoningItem(
                            id=thinking_result.reasoning_id,
                            content=thinking_result.reasoning_content
                        )
                        return thinking_result.content, reasoning_item
                    else:
                        return result, None
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    async def generate_async(
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
        **kwargs
    ) -> AsyncIterator:
        """
        Async wrapper for generation.
        
        Yields results asynchronously for both streaming and non-streaming modes.
        """
        if stream:
            # Run sync generator in thread for streaming
            def gen():
                return self.generate(
                    model, tokenizer, messages,
                    max_tokens, temperature, top_p,
                    repetition_penalty, stop_sequences,
                    stream=True, seed=seed,
                    enable_thinking=enable_thinking,
                    include_reasoning=include_reasoning,
                    **kwargs
                )
            
            generator = await asyncio.to_thread(gen)
            for chunk in generator:
                yield chunk
        else:
            # For non-streaming, run in thread and yield once
            result = await asyncio.to_thread(
                self.generate,
                model, tokenizer, messages,
                max_tokens, temperature, top_p,
                repetition_penalty, stop_sequences,
                stream=False, seed=seed,
                enable_thinking=enable_thinking,
                include_reasoning=include_reasoning,
                **kwargs
            )
            yield result
    
    def _format_messages_to_prompt(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Any
    ) -> str:
        """
        Fallback message formatting when no thinking manager available.
        """
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.debug("Applied tokenizer chat template")
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
        
        # Manual formatting fallback
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role}: {content}")
        
        prompt = "\n\n".join(formatted)
        prompt += "\n\nAssistant:"
        
        logger.debug("Used fallback message formatting")
        return prompt
    
    def _generate_complete(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]],
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_context_size: int = 20,
        **kwargs
    ) -> str:
        """
        Generate complete response using MLX-LM with proper sampling.
        """
        try:
            # Wrap tokenizer if needed
            if not isinstance(tokenizer, TokenizerWrapper):
                tokenizer = TokenizerWrapper(tokenizer)
            
            # Add extra stop tokens to tokenizer
            if stop_sequences:
                for stop_seq in stop_sequences:
                    tokenizer.add_eos_token(stop_seq)
            
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
            
            # Use the advanced generate function with proper sampling
            generated_text = mlx_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                sampler=sampler,
                logits_processors=logits_processors
            )
            
            # Remove prompt from generated text if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            return generated_text.strip()
            
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
        stop_sequences: Optional[List[str]],
        include_reasoning: bool = True,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_context_size: int = 20,
        **kwargs
    ) -> Iterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """
        Stream generation with thinking support using MLX-LM's stream_generate.
        
        Yields tuples of (text_chunk, reasoning_event).
        """
        try:
            # Wrap tokenizer if needed
            if not isinstance(tokenizer, TokenizerWrapper):
                tokenizer = TokenizerWrapper(tokenizer)
            
            # Add extra stop tokens to tokenizer
            if stop_sequences:
                for stop_seq in stop_sequences:
                    tokenizer.add_eos_token(stop_seq)
            
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
            
            # Use MLX-LM's stream_generate for proper streaming
            session_id = generate_id("stream") if self.thinking_manager else None
            accumulated_text = ""
            
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                text_chunk = response.text
                
                # Process through thinking manager if available
                if self.thinking_manager and include_reasoning:
                    accumulated_text += text_chunk
                    processed_text, event = self.thinking_manager.process_streaming_chunk(
                        text_chunk, session_id
                    )
                    
                    if processed_text or event:
                        yield processed_text, event
                else:
                    # Direct streaming without thinking processing
                    if text_chunk:
                        yield text_chunk, None
            
            # Reset streaming state if using thinking manager
            if self.thinking_manager and session_id:
                self.thinking_manager.reset_streaming_state(session_id)
                
        except Exception as e:
            logger.error(f"Stream generation failed: {e}", exc_info=True)
            raise
    
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
    
    def cancel_generation(self, generation_id: str) -> bool:
        """
        Cancel an active generation.
        
        Args:
            generation_id: ID of generation to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        if generation_id in self.active_generations:
            # In a real implementation, this would signal the generation to stop
            del self.active_generations[generation_id]
            logger.info(f"Cancelled generation: {generation_id}")
            return True
        return False