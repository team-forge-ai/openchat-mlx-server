"""
Unified generation engine for MLX inference with thinking support.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Tuple
import time

import mlx.core as mx
from mlx_lm import generate as mlx_generate

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
        **kwargs
    ) -> Union[str, Iterator[str], Tuple[str, Optional[ReasoningItem]]]:
        """
        Generate text using MLX model with thinking support.
        
        Args:
            model: MLX model object
            tokenizer: Tokenizer object
            messages: Input messages or prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            stop_sequences: List of stop sequences
            stream: Whether to stream the response
            seed: Random seed for generation
            enable_thinking: Whether to enable thinking (None = auto)
            include_reasoning: Whether to include reasoning in response
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
                    include_reasoning, **kwargs
                )
            else:
                # Generate complete response
                result = self._generate_complete(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_p,
                    repetition_penalty, stop_sequences,
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
        **kwargs
    ) -> str:
        """
        Generate complete response using MLX-LM.
        """
        try:
            # Filter out unsupported parameters from kwargs
            # MLX currently supports only specific parameters
            supported_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['top_p', 'repetition_penalty', 'temperature']:
                    supported_kwargs[key] = value
            
            # Use MLX-LM's generate function
            generated_text = mlx_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                **supported_kwargs
            )
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break
            
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
        **kwargs
    ) -> Iterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """
        Stream generation with thinking support.
        
        Yields tuples of (text_chunk, reasoning_event).
        """
        # Generate complete text first (MLX-LM doesn't have native streaming)
        # In a real implementation, this would do token-by-token generation
        result = self._generate_complete(
            model, tokenizer, prompt,
            max_tokens, temperature, top_p,
            repetition_penalty, stop_sequences,
            **kwargs
        )
        
        # Simulate streaming with thinking manager processing
        if self.thinking_manager and include_reasoning:
            # Process in chunks for streaming effect
            chunk_size = 10  # characters at a time
            session_id = generate_id("stream")
            
            for i in range(0, len(result), chunk_size):
                chunk = result[i:i+chunk_size]
                text, event = self.thinking_manager.process_streaming_chunk(
                    chunk, session_id
                )
                
                if text or event:
                    yield text, event
            
            # Reset streaming state
            self.thinking_manager.reset_streaming_state(session_id)
        else:
            # Simple streaming without thinking processing
            chunk_size = 5  # words at a time
            words = result.split()
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= chunk_size:
                    yield ' '.join(current_chunk) + ' ', None
                    current_chunk = []
            
            if current_chunk:
                yield ' '.join(current_chunk), None
    
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