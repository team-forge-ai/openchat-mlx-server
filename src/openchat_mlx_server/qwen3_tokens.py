"""
Qwen3 special token handling and utilities.

This module provides comprehensive support for all Qwen3 special tokens,
including thinking, message boundaries, tool calling, and more.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Qwen3SpecialTokens:
    """Registry of Qwen3 special tokens and their IDs."""
    
    # Core message tokens
    ENDOFTEXT = "<|endoftext|>"  # 151643 - Document end/padding
    IM_START = "<|im_start|>"    # 151644 - Message start
    IM_END = "<|im_end|>"        # 151645 - Message end (also EOS token)
    
    # Object and vision tokens
    OBJECT_REF_START = "<|object_ref_start|>"  # 151646
    OBJECT_REF_END = "<|object_ref_end|>"      # 151647
    BOX_START = "<|box_start|>"                # 151648
    BOX_END = "<|box_end|>"                    # 151649
    QUAD_START = "<|quad_start|>"              # 151650
    QUAD_END = "<|quad_end|>"                  # 151651
    VISION_START = "<|vision_start|>"          # 151652
    VISION_END = "<|vision_end|>"              # 151653
    VISION_PAD = "<|vision_pad|>"              # 151654
    IMAGE_PAD = "<|image_pad|>"                # 151655
    VIDEO_PAD = "<|video_pad|>"                # 151656
    
    # Tool calling tokens (non-special but important)
    TOOL_CALL_START = "<tool_call>"            # 151659
    TOOL_CALL_END = "</tool_call>"             # 151660
    TOOL_RESPONSE_START = "<tool_response>"    # 151665
    TOOL_RESPONSE_END = "</tool_response>"     # 151666
    
    # Thinking tokens (non-special but important)
    THINK_START = "<think>"                    # 151667
    THINK_END = "</think>"                     # 151668
    
    # Token ID mapping (when available)
    token_ids: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        """Initialize token ID mapping."""
        self.token_ids = {
            self.ENDOFTEXT: 151643,
            self.IM_START: 151644,
            self.IM_END: 151645,
            self.OBJECT_REF_START: 151646,
            self.OBJECT_REF_END: 151647,
            self.BOX_START: 151648,
            self.BOX_END: 151649,
            self.QUAD_START: 151650,
            self.QUAD_END: 151651,
            self.VISION_START: 151652,
            self.VISION_END: 151653,
            self.VISION_PAD: 151654,
            self.IMAGE_PAD: 151655,
            self.VIDEO_PAD: 151656,
            self.TOOL_CALL_START: 151659,
            self.TOOL_CALL_END: 151660,
            self.TOOL_RESPONSE_START: 151665,
            self.TOOL_RESPONSE_END: 151666,
            self.THINK_START: 151667,
            self.THINK_END: 151668,
        }
    
    def get_stop_tokens(self, include_thinking: bool = True) -> List[str]:
        """
        Get appropriate stop tokens for generation.
        
        Args:
            include_thinking: Whether to include thinking boundaries
            
        Returns:
            List of stop token strings
        """
        stop_tokens = [
            self.IM_END,  # Primary message boundary
            self.ENDOFTEXT,  # Document end
        ]
        
        if include_thinking:
            # Can stop at thinking boundaries for structured output
            stop_tokens.extend([
                self.THINK_END,
                self.THINK_START,  # Sometimes useful to stop before thinking
            ])
        
        return stop_tokens
    
    def get_generation_boundaries(self) -> Set[str]:
        """
        Get tokens that mark generation boundaries.
        
        These are tokens where we might want to pause generation
        for structured output or streaming.
        """
        return {
            self.IM_END,
            self.ENDOFTEXT,
            self.THINK_END,
            self.TOOL_CALL_END,
            self.TOOL_RESPONSE_END,
        }
    
    def is_special_token(self, token: str) -> bool:
        """Check if a token is a special token."""
        return token in self.token_ids
    
    def get_token_id(self, token: str) -> Optional[int]:
        """Get the ID for a special token."""
        return self.token_ids.get(token)


class Qwen3TokenHandler:
    """
    Handler for Qwen3 special tokens in generation and processing.
    
    This class provides utilities for working with Qwen3's special tokens
    during text generation, including proper boundary handling, structured
    output support, and thinking content extraction.
    """
    
    def __init__(self, tokenizer: Any = None):
        """
        Initialize the token handler.
        
        Args:
            tokenizer: The tokenizer object (optional)
        """
        self.tokenizer = tokenizer
        self.special_tokens = Qwen3SpecialTokens()
        self._update_token_ids_from_tokenizer()
    
    def _update_token_ids_from_tokenizer(self):
        """Update token IDs from the tokenizer if available."""
        if not self.tokenizer:
            return
        
        # Try to get actual token IDs from tokenizer
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            decoder = self.tokenizer.added_tokens_decoder
            for token_id, token_info in decoder.items():
                if isinstance(token_info, dict):
                    content = token_info.get('content', '')
                else:
                    content = str(token_info)
                
                # Update our mapping with actual IDs
                if content in self.special_tokens.token_ids:
                    self.special_tokens.token_ids[content] = token_id
                    logger.debug(f"Updated token ID for {content}: {token_id}")
    
    def configure_generation_params(
        self,
        enable_thinking: bool = True,
        structured_output: bool = False
    ) -> Dict[str, Any]:
        """
        Configure generation parameters for Qwen3.
        
        Args:
            enable_thinking: Whether to enable thinking mode
            structured_output: Whether to enforce structured output
            
        Returns:
            Dictionary of generation parameters
        """
        params = {}
        
        # Set appropriate stop tokens
        params['stop_tokens'] = self.special_tokens.get_stop_tokens(
            include_thinking=enable_thinking
        )
        
        # Add stop token IDs if tokenizer is available
        if self.tokenizer:
            stop_ids = []
            for token in params['stop_tokens']:
                token_id = self.special_tokens.get_token_id(token)
                if token_id:
                    stop_ids.append(token_id)
            if stop_ids:
                params['stop_token_ids'] = stop_ids
        
        # Configure for structured output if needed
        if structured_output:
            # For structured output, we might want to stop at specific boundaries
            params['stop_at_boundaries'] = list(
                self.special_tokens.get_generation_boundaries()
            )
        
        return params
    
    def extract_structured_content(self, text: str) -> Dict[str, Any]:
        """
        Extract structured content from Qwen3 output.
        
        This handles all special tokens and returns a structured
        representation of the output.
        
        Args:
            text: The generated text
            
        Returns:
            Dictionary with extracted content sections
        """
        import re
        
        result = {
            'main_content': text,
            'thinking': None,
            'tool_calls': [],
            'tool_responses': [],
            'has_special_tokens': False
        }
        
        # Extract thinking content
        think_pattern = re.compile(
            re.escape(self.special_tokens.THINK_START) + 
            r'(.*?)' + 
            re.escape(self.special_tokens.THINK_END),
            re.DOTALL
        )
        thinking_matches = think_pattern.findall(text)
        if thinking_matches:
            result['thinking'] = '\n'.join(thinking_matches).strip()
            result['has_special_tokens'] = True
            # Remove thinking from main content
            result['main_content'] = think_pattern.sub('', text).strip()
        
        # Extract tool calls
        tool_call_pattern = re.compile(
            re.escape(self.special_tokens.TOOL_CALL_START) + 
            r'(.*?)' + 
            re.escape(self.special_tokens.TOOL_CALL_END),
            re.DOTALL
        )
        tool_calls = tool_call_pattern.findall(result['main_content'])
        if tool_calls:
            result['tool_calls'] = tool_calls
            result['has_special_tokens'] = True
        
        # Extract tool responses
        tool_response_pattern = re.compile(
            re.escape(self.special_tokens.TOOL_RESPONSE_START) + 
            r'(.*?)' + 
            re.escape(self.special_tokens.TOOL_RESPONSE_END),
            re.DOTALL
        )
        tool_responses = tool_response_pattern.findall(result['main_content'])
        if tool_responses:
            result['tool_responses'] = tool_responses
            result['has_special_tokens'] = True
        
        # Clean up message boundaries from main content
        for token in [self.special_tokens.IM_START, self.special_tokens.IM_END]:
            result['main_content'] = result['main_content'].replace(token, '')
        
        # Remove any role markers (e.g., "assistant\n" after <|im_start|>)
        result['main_content'] = re.sub(
            r'^(assistant|user|system)\s*\n', 
            '', 
            result['main_content'].strip()
        )
        
        return result
    
    def format_for_qwen3(
        self,
        content: str,
        role: str = "assistant",
        include_thinking: Optional[str] = None,
        include_boundaries: bool = True
    ) -> str:
        """
        Format content with proper Qwen3 special tokens.
        
        Args:
            content: The main content
            role: The role (assistant, user, system)
            include_thinking: Optional thinking content to include
            include_boundaries: Whether to include message boundaries
            
        Returns:
            Properly formatted text with special tokens
        """
        result = ""
        
        if include_boundaries:
            result += f"{self.special_tokens.IM_START}{role}\n"
        
        if include_thinking:
            result += f"{self.special_tokens.THINK_START}\n"
            result += include_thinking.strip()
            result += f"\n{self.special_tokens.THINK_END}\n\n"
        
        result += content
        
        if include_boundaries:
            result += self.special_tokens.IM_END
        
        return result
    
    def should_stop_generation(
        self,
        generated_tokens: List[int],
        check_boundaries: bool = True
    ) -> bool:
        """
        Check if generation should stop based on special tokens.
        
        Args:
            generated_tokens: List of generated token IDs
            check_boundaries: Whether to check for boundary tokens
            
        Returns:
            True if generation should stop
        """
        if not self.tokenizer or not generated_tokens:
            return False
        
        # Decode the last few tokens to check for stop sequences
        recent_text = self.tokenizer.decode(generated_tokens[-10:])
        
        # Check for primary stop tokens
        stop_tokens = self.special_tokens.get_stop_tokens()
        for token in stop_tokens:
            if token in recent_text:
                logger.debug(f"Found stop token: {token}")
                return True
        
        # Check for generation boundaries if requested
        if check_boundaries:
            boundaries = self.special_tokens.get_generation_boundaries()
            for boundary in boundaries:
                if boundary in recent_text:
                    logger.debug(f"Found boundary token: {boundary}")
                    return True
        
        return False