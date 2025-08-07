"""
Thinking/reasoning extraction utilities for MLX models.

This module provides utilities for detecting and extracting thinking/reasoning
content from generated text, supporting various thinking tag formats.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm.tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)

# Common thinking tag patterns used across different models
THINKING_PATTERNS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|thinking|>", "<|/thinking|>"),
]


def detect_thinking_support(tokenizer: Any) -> Tuple[bool, str]:
    """
    Detect if a tokenizer/model supports thinking.
    
    Args:
        tokenizer: The tokenizer to check
        
    Returns:
        Tuple of (supports_thinking, capability_level)
    """
    # Check if tokenizer has thinking support (mlx_lm TokenizerWrapper)
    if isinstance(tokenizer, TokenizerWrapper):
        if tokenizer.has_thinking:
            return True, "native"
    
    # Check for thinking tokens in vocabulary
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
        
        for start, end in THINKING_PATTERNS:
            if start in vocab and end in vocab:
                return True, "native"
    
    return False, "none"


class ThinkingExtractor:
    """Extractor for thinking/reasoning content from generated text."""
    
    def __init__(self, tokenizer: Any):
        """
        Initialize the thinking extractor.
        
        Args:
            tokenizer: The tokenizer (may be wrapped by mlx_lm)
        """
        self.tokenizer = tokenizer
        
        # Detect thinking tokens
        self.has_thinking = False
        self.think_start = None
        self.think_end = None
        
        if isinstance(tokenizer, TokenizerWrapper):
            self.has_thinking = tokenizer.has_thinking
            self.think_start = tokenizer.think_start
            self.think_end = tokenizer.think_end
        else:
            # Check for thinking tokens in vocabulary
            self._detect_thinking_tokens()
    
    def _detect_thinking_tokens(self):
        """Detect thinking tokens in the tokenizer vocabulary."""
        if not hasattr(self.tokenizer, 'get_vocab'):
            return
            
        vocab = self.tokenizer.get_vocab()
        
        for start, end in THINKING_PATTERNS:
            if start in vocab and end in vocab:
                self.has_thinking = True
                self.think_start = start
                self.think_end = end
                logger.debug(f"Detected thinking tokens: {start}, {end}")
                break
    
    def extract(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking content from generated text.
        
        This method handles both complete and incomplete thinking blocks,
        removing them from the main content and returning them separately.
        
        Args:
            text: The generated text containing potential thinking tags
            
        Returns:
            Tuple of (content_without_thinking, thinking_content)
        """
        if not text:
            return text, None
        
        # Build patterns to check
        patterns_to_check = []
        
        # Add detected pattern first if available
        if self.has_thinking and self.think_start and self.think_end:
            pattern = re.escape(self.think_start) + r"(.*?)" + re.escape(self.think_end)
            patterns_to_check.append((pattern, self.think_start, self.think_end))
        
        # Add common patterns
        for start, end in THINKING_PATTERNS:
            pattern = re.escape(start) + r"(.*?)" + re.escape(end)
            patterns_to_check.append((pattern, start, end))
        
        # Try each pattern
        for pattern, start, end in patterns_to_check:
            if start in text:
                # Check for complete thinking blocks
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    # Extract all thinking content
                    thinking_content = "\n\n".join(match.strip() for match in matches)
                    # Remove thinking blocks from text
                    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
                    # Clean up whitespace
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
        
        # No thinking content found
        return text, None
    
    def process_streaming_chunk(
        self, 
        chunk: str, 
        in_thinking: bool = False
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        """
        Process a streaming chunk for thinking content.
        
        Args:
            chunk: The text chunk from streaming
            in_thinking: Whether we're currently inside a thinking block
            
        Returns:
            Tuple of (output_text, still_in_thinking, thinking_content)
        """
        if not chunk:
            return chunk, in_thinking, None
        
        output_text = chunk
        thinking_content = None
        
        # Check for thinking start marker
        if self.think_start in chunk and not in_thinking:
            # Split at thinking start
            parts = chunk.split(self.think_start, 1)
            output_text = parts[0] if parts[0] else None
            
            # Check if thinking ends in same chunk
            if self.think_end in parts[1]:
                # Complete thinking block in one chunk
                thinking_parts = parts[1].split(self.think_end, 1)
                thinking_content = thinking_parts[0]
                # Add any remaining text after thinking
                if thinking_parts[1]:
                    output_text = (output_text or "") + thinking_parts[1]
                return output_text, False, thinking_content
            else:
                # Thinking continues to next chunk
                thinking_content = parts[1]
                return output_text, True, thinking_content
        
        # Check for thinking end marker when in thinking
        if in_thinking and self.think_end in chunk:
            parts = chunk.split(self.think_end, 1)
            thinking_content = parts[0]
            output_text = parts[1] if len(parts) > 1 else None
            return output_text, False, thinking_content
        
        # If in thinking but no end marker, entire chunk is thinking
        if in_thinking:
            return None, True, chunk
        
        return output_text, False, None


class StreamingThinkingProcessor:
    """Helper class to manage thinking state during streaming."""
    
    def __init__(self, extractor: ThinkingExtractor):
        """
        Initialize the streaming processor.
        
        Args:
            extractor: The thinking extractor to use
        """
        self.extractor = extractor
        self.in_thinking = False
        self.thinking_buffer = []
        self.session_id = None
    
    def process_chunk(self, chunk: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process a streaming chunk and return output text and reasoning events.
        
        Args:
            chunk: The text chunk from streaming
            
        Returns:
            Tuple of (output_text, reasoning_event)
        """
        if not self.extractor.has_thinking:
            return chunk, None
        
        output_text, still_thinking, thinking_content = \
            self.extractor.process_streaming_chunk(chunk, self.in_thinking)
        
        reasoning_event = None
        
        # Handle state transitions
        if not self.in_thinking and still_thinking:
            # Started thinking
            self.in_thinking = True
            self.thinking_buffer = [thinking_content] if thinking_content else []
            reasoning_event = {
                "type": "start",
                "id": self.session_id,
                "content": None,
                "partial": False
            }
        elif self.in_thinking and not still_thinking:
            # Finished thinking
            if thinking_content:
                self.thinking_buffer.append(thinking_content)
            
            reasoning_event = {
                "type": "complete",
                "id": self.session_id,
                "content": "".join(self.thinking_buffer),
                "partial": False
            }
            
            # Reset state
            self.in_thinking = False
            self.thinking_buffer = []
        elif self.in_thinking:
            # Still in thinking, accumulate
            if thinking_content:
                self.thinking_buffer.append(thinking_content)
        
        return output_text, reasoning_event
    
    def reset(self):
        """Reset the processor state."""
        self.in_thinking = False
        self.thinking_buffer = []