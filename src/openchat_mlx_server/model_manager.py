"""
Unified model manager for MLX Engine Server with thinking support.
"""

import json
import time
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

import mlx.core as mx
from mlx_lm import load

from .utils import (
    detect_model_type,
    validate_model_path,
    format_bytes,
    SystemMonitor
)
from .thinking_manager import ThinkingManager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about the loaded model."""
    
    model_path: Path
    model: Any  # MLX model object
    tokenizer: Any  # Tokenizer object
    model_type: str
    architecture: Optional[str]
    tokenizer_config: Dict[str, Any]
    chat_template: Optional[str]
    loaded_at: datetime
    memory_usage: Optional[int] = None  # in bytes
    supports_thinking: bool = False  # Whether model supports thinking/reasoning
    thinking_capability: Optional[str] = None  # Capability level (none/basic/native/advanced)


class MLXModelManager:
    """
    Manages MLX model loading and configuration with thinking support.
    
    Features:
    - Auto-detection of model type and capabilities
    - Thinking/reasoning support detection
    - Proper tokenizer configuration
    - Memory usage tracking
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.model_info: Optional[ModelInfo] = None
        self.thinking_manager: Optional[ThinkingManager] = None
        self.system_monitor = SystemMonitor()
        logger.info("MLXModelManager initialized")
    
    def load_model(
        self,
        model_path: str,
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Load an MLX model with thinking support detection.
        
        Args:
            model_path: Path to the model directory
            tokenizer_config: Optional tokenizer configuration
        
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Validate and prepare path
            model_path = Path(model_path).resolve()
            if not validate_model_path(model_path):
                return False, f"Invalid model path: {model_path}"
            
            # Detect model metadata
            model_metadata = self._detect_model_metadata(model_path)
            logger.info(f"Detected model type: {model_metadata['type']}")
            
            # Load the model and tokenizer
            start_time = time.time()
            model, tokenizer = load(str(model_path))
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Apply tokenizer configuration if provided
            if tokenizer_config:
                self._apply_tokenizer_config(tokenizer, tokenizer_config)
            
            # Get memory usage
            memory_usage = self._estimate_model_memory(model)
            
            # Initialize thinking manager
            self.thinking_manager = ThinkingManager(tokenizer, model_metadata)
            supports_thinking = self.thinking_manager.supports_thinking
            thinking_capability = self.thinking_manager.capability.value
            
            # Create model info
            self.model_info = ModelInfo(
                model_path=model_path,
                model=model,
                tokenizer=tokenizer,
                model_type=model_metadata["type"],
                architecture=model_metadata.get("architecture"),
                tokenizer_config=tokenizer_config or {},
                chat_template=model_metadata.get("chat_template"),
                loaded_at=datetime.now(),
                memory_usage=memory_usage,
                supports_thinking=supports_thinking,
                thinking_capability=thinking_capability
            )
            
            # Log model information
            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Type: {self.model_info.model_type}")
            logger.info(f"  - Architecture: {self.model_info.architecture}")
            logger.info(f"  - Memory usage: {format_bytes(memory_usage or 0)}")
            logger.info(f"  - Supports thinking: {supports_thinking}")
            logger.info(f"  - Thinking capability: {thinking_capability}")
            
            return True, f"Model loaded successfully from {model_path}"
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False, f"Failed to load model: {str(e)}"
    
    def _detect_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """
        Detect model metadata from configuration files.
        
        Args:
            model_path: Path to model directory
        
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            "type": "unknown",
            "architecture": None,
            "chat_template": None
        }
        
        # Check for config.json
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    metadata["architecture"] = config.get("architectures", [None])[0]
                    metadata["type"] = detect_model_type(model_path)
            except Exception as e:
                logger.warning(f"Failed to read config.json: {e}")
        
        # Check for tokenizer_config.json
        tokenizer_config_path = model_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            try:
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                    metadata["chat_template"] = tokenizer_config.get("chat_template")
            except Exception as e:
                logger.warning(f"Failed to read tokenizer_config.json: {e}")
        
        return metadata
    
    def _apply_tokenizer_config(
        self,
        tokenizer: Any,
        config: Dict[str, Any]
    ):
        """
        Apply configuration to tokenizer.
        
        Args:
            tokenizer: Tokenizer object
            config: Configuration dictionary
        """
        for key, value in config.items():
            if hasattr(tokenizer, key):
                try:
                    setattr(tokenizer, key, value)
                    logger.debug(f"Set tokenizer.{key} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to set tokenizer.{key}: {e}")
    
    def get_model(self) -> Optional[ModelInfo]:
        """Get the loaded model info."""
        return self.model_info
    
    def get_thinking_manager(self) -> Optional[ThinkingManager]:
        """Get the thinking manager for the loaded model."""
        return self.thinking_manager
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status.
        
        Returns:
            Dictionary with model status information
        """
        if not self.model_info:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "path": str(self.model_info.model_path),
            "type": self.model_info.model_type,
            "architecture": self.model_info.architecture,
            "loaded_at": self.model_info.loaded_at.isoformat(),
            "memory_usage": format_bytes(self.model_info.memory_usage or 0),
            "supports_thinking": self.model_info.supports_thinking,
            "thinking_capability": self.model_info.thinking_capability
        }
    
    def format_chat_template(
        self,
        messages: list[Dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None
    ) -> str:
        """
        Apply chat template formatting with thinking support.
        
        Args:
            messages: List of message dictionaries
            add_generation_prompt: Whether to add generation prompt
            enable_thinking: Whether to enable thinking mode (None = auto)
        
        Returns:
            Formatted prompt string
        """
        if not self.model_info:
            raise ValueError("No model loaded")
        
        # Use thinking manager if available
        if self.thinking_manager:
            return self.thinking_manager.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=add_generation_prompt
            )
        
        # Fallback to standard tokenizer chat template
        tokenizer = self.model_info.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
        
        # Manual formatting fallback
        formatted_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            else:
                formatted_messages.append(f"{role}: {content}")
        
        prompt = "\n\n".join(formatted_messages)
        if add_generation_prompt:
            prompt += "\n\nAssistant:"
        
        return prompt
    
    def _estimate_model_memory(self, model: Any) -> int:
        """
        Estimate model memory usage.
        
        Args:
            model: MLX model object
        
        Returns:
            Estimated memory usage in bytes
        """
        try:
            # Get memory info from MLX
            memory_info = mx.metal.get_memory_info()
            return memory_info.get("used_bytes", 0)
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            
            # Fallback: rough estimate based on parameter count
            try:
                param_count = sum(p.size for p in model.parameters().values())
                # Assume 2 bytes per parameter (16-bit precision)
                return param_count * 2
            except:
                return 0
    
    def unload_model(self) -> Tuple[bool, str]:
        """
        Unload the current model.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.model_info:
            return False, "No model loaded"
        
        try:
            # Clear model and tokenizer
            self.model_info = None
            self.thinking_manager = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear MLX cache
            mx.clear_cache()
            
            logger.info("Model unloaded successfully")
            return True, "Model unloaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}", exc_info=True)
            return False, f"Failed to unload model: {str(e)}"