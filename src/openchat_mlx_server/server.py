"""Main FastAPI server for MLX Engine Server."""

import asyncio
import json
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import uvicorn

from . import __version__
from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    ModelListResponse,
    ModelInfo as APIModelInfo,
    ErrorResponse,
    create_chat_completion_response,
    create_stream_response_chunk,
    create_error_response,
)
from .config import ServerConfig
from .generation import GenerationEngine
from .model_manager import MLXModelManager
from .utils import (
    setup_logging,
    SystemMonitor,
    ProcessManager,
    generate_id,
    format_timestamp,
)
# Thinking is now handled by ThinkingManager

logger = logging.getLogger(__name__)


class MLXServer:
    """Main MLX Engine Server implementation."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the MLX server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.model_manager = MLXModelManager()
        self.generation_engine = GenerationEngine()
        self.system_monitor = SystemMonitor()
        self.process_manager = ProcessManager()
        
        # Track active requests
        self.active_requests = 0
        self.total_requests = 0
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage application lifecycle."""
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="MLX Engine Server",
            description="OpenAI-compatible inference server for Apple Silicon",
            version=__version__,
            lifespan=lifespan
        )
        
        # Add CORS middleware if enabled
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add exception handlers
        app.add_exception_handler(RequestValidationError, self._validation_error_handler)
        app.add_exception_handler(HTTPException, self._http_error_handler)
        app.add_exception_handler(Exception, self._general_error_handler)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register all API routes."""
        
        # Health check
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": format_timestamp(),
                "version": __version__
            }
        
        # OpenAI-compatible endpoints
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint."""
            return await self._handle_chat_completion(request)
        
        @app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """OpenAI-compatible completions endpoint."""
            return await self._handle_completion(request)
        
        @app.get("/v1/models")
        async def list_models():
            """Get information about the loaded model."""
            return self._get_model_info()
        
        # MLX-specific endpoints
        @app.get("/v1/mlx/status")
        async def server_status():
            """Get server status."""
            return self._get_server_status()
        
        @app.get("/v1/mlx/model-info")
        async def model_info():
            """Get detailed model information."""
            return self._get_detailed_model_info()
    
    async def _startup(self):
        """Server startup tasks."""
        logger.info(f"Starting MLX Engine Server v{__version__}")
        
        # Write PID file
        self.process_manager.write_pid()
        
        # Model is already loaded in main.py before server starts
        if self.model_manager.model_info:
            logger.info("Model is ready for inference")
            # Set up thinking extraction with the tokenizer
            self.generation_engine.set_thinking_extractor(self.model_manager.model_info.tokenizer)
            logger.info(f"Thinking support: {self.model_manager.model_info.thinking_capability}")
    
    async def _shutdown(self):
        """Server shutdown tasks."""
        logger.info("Shutting down MLX Engine Server")
        
        # Wait for active requests to complete
        max_wait = 30  # seconds
        start_time = time.time()
        while self.active_requests > 0 and (time.time() - start_time) < max_wait:
            logger.info(f"Waiting for {self.active_requests} active requests to complete...")
            await asyncio.sleep(1)
        
        # Clean up models
        self.model_manager.unload_model()
        
        # Clean up PID file
        self.process_manager.cleanup()
        
        logger.info("Shutdown complete")
    
    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Handle chat completion request.
        
        Args:
            request: Chat completion request
        
        Returns:
            Chat completion response or streaming response
        """
        self.active_requests += 1
        self.total_requests += 1
        request_id = generate_id("chatcmpl")
        
        try:
            # Get the loaded model
            model_info = self.model_manager.get_model()
            if not model_info:
                raise HTTPException(
                    status_code=503,
                    detail="No model loaded"
                )
            
            # Convert messages to format expected by generation engine
            messages = [msg.model_dump() for msg in request.messages]
            
            # Thinking is handled automatically by ThinkingManager
            # which auto-enables for capable models and checks control tags
            enable_thinking = request.enable_thinking
            
            # Generate response
            if request.stream:
                # Return streaming response
                return StreamingResponse(
                    self._stream_chat_completion(
                        request_id,
                        model_info,
                        messages,
                        request,
                        enable_thinking
                    ),
                    media_type="text/event-stream"
                )
            else:
                # Generate complete response
                # generate_async always returns an async generator, even for non-streaming
                async_gen = self.generation_engine.generate_async(
                    model_info.model,
                    model_info.tokenizer,
                    messages,
                    max_tokens=request.max_tokens or self.config.default_max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    min_p=request.min_p,
                    repetition_penalty=1.0,  # Map from frequency_penalty if needed
                    stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                    stream=False,
                    seed=request.seed,
                    enable_thinking=enable_thinking,
                    include_reasoning=request.include_reasoning
                )
                # For non-streaming, collect the single yielded result
                result = None
                async for res in async_gen:
                    result = res
                    break  # Only one result for non-streaming
                
                # Handle result based on whether it includes reasoning
                reasoning_item = None
                reasoning_tokens = 0
                if isinstance(result, tuple) and len(result) == 2:
                    generated_text, reasoning_item = result
                    # Count reasoning tokens if present
                    if reasoning_item and reasoning_item.content:
                        reasoning_tokens = self.generation_engine.count_tokens(
                            model_info.tokenizer,
                            reasoning_item.content
                        )
                else:
                    generated_text = result
                
                # Count tokens
                prompt_text = self.model_manager.format_chat_template(
                    messages,
                    enable_thinking=enable_thinking
                )
                prompt_tokens = self.generation_engine.count_tokens(
                    model_info.tokenizer,
                    prompt_text
                )
                completion_tokens = self.generation_engine.count_tokens(
                    model_info.tokenizer,
                    generated_text
                )
                
                # Create response
                return create_chat_completion_response(
                    request_id=request_id,
                    model=request.model,
                    content=generated_text,
                    finish_reason="stop",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_item=reasoning_item,
                    reasoning_tokens=reasoning_tokens
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat completion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
        finally:
            self.active_requests -= 1
    
    async def _stream_chat_completion(
        self,
        request_id: str,
        model_info: Any,
        messages: list,
        request: ChatCompletionRequest,
        enable_thinking: Optional[bool] = None
    ) -> AsyncIterator[str]:
        """
        Stream chat completion response.
        
        Yields:
            SSE formatted response chunks
        """
        try:
            # Send initial chunk with role
            initial_chunk = create_stream_response_chunk(
                request_id=request_id,
                model=request.model,
                role="assistant"
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"
            
            # Generate and stream response
            async for result in self.generation_engine.generate_async(
                model_info.model,
                model_info.tokenizer,
                messages,
                max_tokens=request.max_tokens or self.config.default_max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                repetition_penalty=1.0,
                stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
                stream=True,
                seed=request.seed,
                enable_thinking=enable_thinking,
                include_reasoning=request.include_reasoning
            ):
                # Handle streaming result with potential reasoning events
                if isinstance(result, tuple) and len(result) == 2:
                    chunk, reasoning_event = result
                else:
                    chunk = result
                    reasoning_event = None
                
                # Create response chunk with content and/or reasoning event
                # Convert ReasoningEvent to dict if needed
                reasoning_event_dict = None
                if reasoning_event:
                    if hasattr(reasoning_event, 'model_dump'):
                        reasoning_event_dict = reasoning_event.model_dump()
                    elif hasattr(reasoning_event, '__dict__'):
                        reasoning_event_dict = reasoning_event.__dict__
                    else:
                        reasoning_event_dict = reasoning_event
                
                if chunk:
                    response_chunk = create_stream_response_chunk(
                        request_id=request_id,
                        model=request.model,
                        content=chunk,
                        reasoning_event=reasoning_event_dict
                    )
                    yield f"data: {response_chunk.model_dump_json()}\n\n"
                elif reasoning_event_dict:
                    # Send reasoning-only event
                    response_chunk = create_stream_response_chunk(
                        request_id=request_id,
                        model=request.model,
                        reasoning_event=reasoning_event_dict
                    )
                    yield f"data: {response_chunk.model_dump_json()}\n\n"
            
            # Send final chunk
            final_chunk = create_stream_response_chunk(
                request_id=request_id,
                model=request.model,
                finish_reason="stop"
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _handle_completion(
        self,
        request: CompletionRequest
    ) -> CompletionResponse:
        """Handle completion request (for backwards compatibility)."""
        # Convert to chat completion format
        messages = [{"role": "user", "content": request.prompt if isinstance(request.prompt, str) else request.prompt[0]}]
        
        # Get the loaded model
        model_info = self.model_manager.get_model()
        if not model_info:
            raise HTTPException(
                status_code=503,
                detail="No model loaded"
            )
        
        # Generate response
        generated_text = await self.generation_engine.generate_async(
            model_info.model,
            model_info.tokenizer,
            request.prompt if isinstance(request.prompt, str) else request.prompt[0],
            max_tokens=request.max_tokens or self.config.default_max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            stream=False,
            seed=request.seed
        )
        
        # Create response
        return CompletionResponse(
            id=generate_id("cmpl"),
            created=format_timestamp(),
            model=request.model,
            choices=[{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(request.prompt) // 4,  # Rough estimate
                "completion_tokens": len(generated_text) // 4,
                "total_tokens": (len(request.prompt) + len(generated_text)) // 4
            }
        )
    
    def _get_model_info(self) -> ModelListResponse:
        """Get information about the loaded model."""
        model_status = self.model_manager.get_model_status()
        
        if model_status.get("loaded", False):
            # Use the model path name as the ID
            model_id = Path(model_status["path"]).name
            return ModelListResponse(
                data=[
                    APIModelInfo(
                        id=model_id,
                        created=format_timestamp(),
                        owned_by="local"
                    )
                ]
            )
        else:
            return ModelListResponse(data=[])

    
    def _get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        model_status = self.model_manager.get_model_status()
        system_info = self.system_monitor.get_system_info()
        
        return {
            "status": "healthy" if model_status.get("loaded", False) else "no_model",
            "model_loaded": model_status.get("loaded", False),
            "model_info": model_status if model_status.get("loaded", False) else None,
            "memory_usage": system_info.get("memory", {}),
            "cpu_usage": system_info.get("cpu", {}),
            "gpu_usage": system_info.get("gpu", {}),
            "uptime": system_info.get("uptime", 0),
            "mlx_version": system_info.get("gpu", {}).get("mlx_version", "unknown"),
            "server_version": __version__,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests
        }
    
    def _get_detailed_model_info(self) -> Dict[str, Any]:
        """Get detailed model information for /v1/mlx/model-info endpoint."""
        model_status = self.model_manager.get_model_status()
        
        if model_status.get("loaded", False):
            return {
                "path": model_status["path"],
                "type": model_status["type"],
                "architecture": model_status.get("architecture"),
                "supports_thinking": model_status.get("supports_thinking", False),
                "thinking_capability": model_status.get("thinking_capability", "none"),
                "loaded_at": model_status.get("loaded_at"),
                "memory_usage": model_status.get("memory_usage")
            }
        else:
            return {
                "loaded": False,
                "supports_thinking": False,
                "thinking_capability": "none"
            }
    
    async def _validation_error_handler(self, request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        return JSONResponse(
            status_code=422,
            content=create_error_response(
                message=str(exc),
                error_type="validation_error"
            ).model_dump()
        )
    
    async def _http_error_handler(self, request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(
                message=exc.detail,
                error_type="http_error"
            ).model_dump()
        )
    
    async def _general_error_handler(self, request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                message="Internal server error",
                error_type="internal_error"
            ).model_dump()
        )
    
    def run(self):
        """Run the server."""
        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower(),
            reload=self.config.reload,
            workers=self.config.workers if not self.config.reload else 1
        )