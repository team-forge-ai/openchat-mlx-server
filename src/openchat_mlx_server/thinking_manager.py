"""
Unified thinking/reasoning manager for MLX models.

This module provides comprehensive support for thinking/reasoning capabilities
across different model architectures, with special support for Qwen3 models.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .utils import generate_id
from .qwen3_tokens import Qwen3TokenHandler, Qwen3SpecialTokens

logger = logging.getLogger(__name__)


class ThinkingCapability(Enum):
    """Enum for thinking capability levels."""
    NONE = "none"
    BASIC = "basic"  # Simple pattern matching
    NATIVE = "native"  # Native tokenizer support
    ADVANCED = "advanced"  # Full special token support (Qwen3)


@dataclass
class ThinkingResult:
    """Unified result from thinking/reasoning extraction."""
    content: str  # Main content without thinking
    reasoning_content: Optional[str] = None  # Extracted reasoning
    reasoning_id: Optional[str] = None  # Unique ID for reasoning
    tool_calls: List[str] = None  # Any tool calls found
    has_special_tokens: bool = False  # Whether special tokens were found
    capability_used: ThinkingCapability = ThinkingCapability.NONE


@dataclass
class ReasoningEvent:
    """Event for streaming reasoning updates."""
    type: str  # "start", "progress", "complete"
    id: str
    content: Optional[str] = None
    partial: bool = False


class ThinkingManager:
    """
    Unified manager for all thinking/reasoning operations.
    
    This class consolidates all thinking-related functionality:
    - Detection of thinking capabilities
    - Chat template application with thinking support
    - Reasoning extraction from model outputs
    - Special token handling
    - Streaming support
    """
    
    def __init__(self, tokenizer: Any = None, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the thinking manager.
        
        Args:
            tokenizer: The tokenizer object
            model_config: Optional model configuration dictionary
        """
        self.tokenizer = tokenizer
        self.model_config = model_config or {}
        self.capability = self._detect_capability()
        self.token_handler = self._setup_token_handler()
        
        # For streaming
        self.streaming_buffer = ""
        self.streaming_state = {}
        
        logger.info(f"ThinkingManager initialized with capability: {self.capability.value}")
    
    def _detect_capability(self) -> ThinkingCapability:
        """
        Detect the thinking capability level of the model.
        
        Returns:
            ThinkingCapability enum value
        """
        if not self.tokenizer:
            return ThinkingCapability.NONE
        
        # Check for Qwen3 (advanced)
        if self._is_qwen3_model():
            return ThinkingCapability.ADVANCED
        
        # Check for native tokenizer support
        if self._has_native_thinking_support():
            return ThinkingCapability.NATIVE
        
        # Check for basic pattern support
        if self._has_basic_thinking_patterns():
            return ThinkingCapability.BASIC
        
        return ThinkingCapability.NONE
    
    def _is_qwen3_model(self) -> bool:
        """Check if this is a Qwen3 model."""
        # Check tokenizer class
        if hasattr(self.tokenizer, '__class__'):
            class_name = self.tokenizer.__class__.__name__.lower()
            if 'qwen' in class_name:
                return True
        
        # Check for Qwen3 special tokens
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            decoder = getattr(self.tokenizer, 'added_tokens_decoder', {})
            for token_info in decoder.values():
                content = token_info.get('content', '') if isinstance(token_info, dict) else str(token_info)
                if '<|im_start|>' in content or '<|im_end|>' in content:
                    return True
        
        # Check model config
        architecture = self.model_config.get('architecture', '') or ''
        if 'qwen' in architecture.lower():
            return True
        
        return False
    
    def _has_native_thinking_support(self) -> bool:
        """Check if tokenizer has native thinking support."""
        if hasattr(self.tokenizer, 'chat_template'):
            template = str(getattr(self.tokenizer, 'chat_template', ''))
            if 'enable_thinking' in template or 'reasoning_content' in template:
                return True
        
        # Check for thinking tokens
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            decoder = getattr(self.tokenizer, 'added_tokens_decoder', {})
            for token_info in decoder.values():
                content = token_info.get('content', '') if isinstance(token_info, dict) else str(token_info)
                if '<think>' in content or '</think>' in content:
                    return True
        
        return False
    
    def _has_basic_thinking_patterns(self) -> bool:
        """Check if model outputs basic thinking patterns."""
        # This would be determined by model training, but we can check for common patterns
        return True  # Most models can potentially output thinking patterns
    
    def _setup_token_handler(self) -> Optional[Any]:
        """Set up appropriate token handler based on capability."""
        if self.capability == ThinkingCapability.ADVANCED:
            return Qwen3TokenHandler(self.tokenizer)
        return None
    
    @property
    def supports_thinking(self) -> bool:
        """Check if model supports any form of thinking."""
        return self.capability != ThinkingCapability.NONE
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: Optional[bool] = None,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Apply chat template with thinking support.
        
        Args:
            messages: List of message dictionaries
            enable_thinking: Whether to enable thinking (None = auto-detect)
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt string
        """
        if not self.tokenizer:
            raise ValueError("No tokenizer available")
        
        # Auto-enable thinking for capable models
        if enable_thinking is None and self.supports_thinking:
            enable_thinking = True
            logger.debug("Auto-enabled thinking for capable model")
        
        # Apply template based on capability
        if self.capability == ThinkingCapability.ADVANCED and self.token_handler:
            return self._apply_advanced_template(messages, enable_thinking, add_generation_prompt)
        elif self.capability == ThinkingCapability.NATIVE:
            return self._apply_native_template(messages, enable_thinking, add_generation_prompt)
        else:
            return self._apply_basic_template(messages, add_generation_prompt)
    
    def _apply_advanced_template(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool,
        add_generation_prompt: bool
    ) -> str:
        """Apply advanced template with full token support."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt
        }
        
        # Add enable_thinking for Qwen3
        if hasattr(self.tokenizer, 'chat_template'):
            template = str(self.tokenizer.chat_template or '')
            if 'enable_thinking' in template:
                kwargs["enable_thinking"] = enable_thinking
        
        return self.tokenizer.apply_chat_template(messages, **kwargs)
    
    def _apply_native_template(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool,
        add_generation_prompt: bool
    ) -> str:
        """Apply native template with basic thinking support."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt
        }
        
        # Try to pass enable_thinking if supported
        if enable_thinking and hasattr(self.tokenizer, 'chat_template'):
            template = str(self.tokenizer.chat_template or '')
            if 'enable_thinking' in template:
                kwargs["enable_thinking"] = enable_thinking
        
        return self.tokenizer.apply_chat_template(messages, **kwargs)
    
    def _apply_basic_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool
    ) -> str:
        """Apply basic template without special thinking support."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        
        # Fallback to manual formatting
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        
        prompt = "\n\n".join(formatted)
        if add_generation_prompt:
            prompt += "\n\nAssistant:"
        
        return prompt
    
    def extract_reasoning(
        self,
        output: str,
        include_reasoning: bool = True
    ) -> ThinkingResult:
        """
        Extract reasoning from model output.
        
        Args:
            output: The model's output text
            include_reasoning: Whether to include reasoning in result
            
        Returns:
            ThinkingResult with extracted content
        """
        if self.capability == ThinkingCapability.ADVANCED and self.token_handler:
            return self._extract_advanced_reasoning(output, include_reasoning)
        elif self.capability in [ThinkingCapability.NATIVE, ThinkingCapability.BASIC]:
            return self._extract_pattern_reasoning(output, include_reasoning)
        else:
            return ThinkingResult(
                content=output,
                capability_used=ThinkingCapability.NONE
            )
    
    def _extract_advanced_reasoning(
        self,
        output: str,
        include_reasoning: bool
    ) -> ThinkingResult:
        """Extract reasoning using advanced token handler."""
        extracted = self.token_handler.extract_structured_content(output)
        
        reasoning_id = None
        if extracted['thinking'] and include_reasoning:
            reasoning_id = generate_id("reasoning")
        
        return ThinkingResult(
            content=extracted['main_content'],
            reasoning_content=extracted['thinking'] if include_reasoning else None,
            reasoning_id=reasoning_id,
            tool_calls=extracted.get('tool_calls', []),
            has_special_tokens=extracted.get('has_special_tokens', False),
            capability_used=ThinkingCapability.ADVANCED
        )
    
    def _extract_pattern_reasoning(
        self,
        output: str,
        include_reasoning: bool
    ) -> ThinkingResult:
        """Extract reasoning using pattern matching."""
        # Common thinking patterns
        patterns = [
            (re.compile(r'<think>(.*?)</think>', re.DOTALL), 'think'),
            (re.compile(r'<\|thinking\|>(.*?)<\|/thinking\|>', re.DOTALL), 'thinking'),
            (re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL), 'reasoning'),
        ]
        
        for pattern, name in patterns:
            matches = pattern.findall(output)
            if matches:
                # Remove thinking blocks from output
                cleaned = pattern.sub('', output).strip()
                reasoning = '\n'.join(matches).strip()
                
                reasoning_id = None
                if reasoning and include_reasoning:
                    reasoning_id = generate_id("reasoning")
                
                return ThinkingResult(
                    content=cleaned,
                    reasoning_content=reasoning if include_reasoning else None,
                    reasoning_id=reasoning_id,
                    has_special_tokens=True,
                    capability_used=ThinkingCapability.BASIC
                )
        
        # No thinking found
        return ThinkingResult(
            content=output,
            capability_used=ThinkingCapability.BASIC
        )
    

    
    def get_generation_config(self, enable_thinking: bool = True) -> Dict[str, Any]:
        """
        Get generation configuration with appropriate stop tokens.
        
        Args:
            enable_thinking: Whether thinking is enabled
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        if self.capability == ThinkingCapability.ADVANCED and self.token_handler:
            params = self.token_handler.configure_generation_params(
                enable_thinking=enable_thinking,
                structured_output=False
            )
            config.update(params)
        elif self.capability == ThinkingCapability.NATIVE:
            config['stop_tokens'] = ['</think>'] if enable_thinking else []
        
        return config
    
    def process_streaming_chunk(
        self,
        chunk: str,
        session_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[ReasoningEvent]]:
        """
        Process a streaming chunk and extract reasoning events.
        
        Args:
            chunk: The streaming chunk
            session_id: Optional session identifier
            
        Returns:
            Tuple of (display_text, reasoning_event)
        """
        session_id = session_id or "default"
        
        if session_id not in self.streaming_state:
            self.streaming_state[session_id] = {
                'buffer': '',
                'in_thinking': False,
                'thinking_content': [],
                'reasoning_id': None
            }
        
        state = self.streaming_state[session_id]
        state['buffer'] += chunk
        
        display_text = None
        reasoning_event = None
        
        # Check for thinking boundaries
        if '<think>' in state['buffer'] and not state['in_thinking']:
            parts = state['buffer'].split('<think>', 1)
            if parts[0]:
                display_text = parts[0]
            state['buffer'] = parts[1] if len(parts) > 1 else ""
            state['in_thinking'] = True
            state['reasoning_id'] = generate_id("reasoning")
            
            reasoning_event = ReasoningEvent(
                type="start",
                id=state['reasoning_id']
            )
        
        elif '</think>' in state['buffer'] and state['in_thinking']:
            parts = state['buffer'].split('</think>', 1)
            if parts[0]:
                state['thinking_content'].append(parts[0])
            state['buffer'] = parts[1] if len(parts) > 1 else ""
            state['in_thinking'] = False
            
            reasoning_event = ReasoningEvent(
                type="complete",
                id=state['reasoning_id'],
                content=''.join(state['thinking_content'])
            )
            state['thinking_content'] = []
        
        elif state['in_thinking']:
            state['thinking_content'].append(chunk)
            reasoning_event = ReasoningEvent(
                type="progress",
                id=state['reasoning_id'],
                content=chunk,
                partial=True
            )
        
        else:
            display_text = chunk
        
        return display_text, reasoning_event
    
    def reset_streaming_state(self, session_id: Optional[str] = None):
        """Reset streaming state for a session."""
        session_id = session_id or "default"
        if session_id in self.streaming_state:
            del self.streaming_state[session_id]