#!/usr/bin/env python3
"""
OpenAI-compatible API server for Shepherd
Communicates with parent Shepherd process via Unix socket
"""

import os
import sys
import json
import time
import uuid
import argparse
import signal
import asyncio
from typing import Optional, List, Dict
import socket
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# ============================================================================
# OpenAI API Models
# ============================================================================

class OpenAIError(BaseModel):
    """OpenAI-compatible error response"""
    message: str
    type: str
    code: Optional[str] = None

class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error response wrapper"""
    error: OpenAIError

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: bool = False
    tools: Optional[List[Dict]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "shepherd"
    max_model_len: Optional[int] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Shepherd Communication
# ============================================================================

class ShepherdClient:
    """Communicates with parent Shepherd process via socket"""

    def __init__(self, sock_fd: int):
        self.sock = socket.socket(fileno=sock_fd)
        self.sock_file_read = self.sock.makefile('r')
        self.sock_file_write = self.sock.makefile('w')

    def send_request(self, request: Dict) -> Dict:
        """Send JSON request and receive JSON response"""
        request_line = json.dumps(request) + '\n'
        self.sock_file_write.write(request_line)
        self.sock_file_write.flush()

        response_line = self.sock_file_read.readline()
        if not response_line:
            raise RuntimeError("Shepherd process closed connection")

        response = json.loads(response_line)

        if response.get("status") != "success":
            raise RuntimeError(f"{response.get('error')}")

        return response

    def generate(self, messages: List[Dict], parameters: Dict, tools: List[Dict] = None) -> Dict:
        """Request text generation"""
        request = {
            "action": "generate",
            "messages": messages,
            "parameters": parameters
        }
        if tools:
            request["tools"] = tools
        return self.send_request(request)

    def list_models(self) -> List[Dict]:
        """List available models"""
        request = {"action": "list_models"}
        response = self.send_request(request)
        return response.get("models", [])

    def get_model_info(self) -> Dict:
        """Get model information including context size"""
        request = {"action": "get_model_info"}
        response = self.send_request(request)
        return response.get("model_info", {})


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Shepherd OpenAI-Compatible API",
    description="OpenAI-compatible API for Shepherd LLM inference engine",
    version="1.0.0"
)

# Custom exception handler to return OpenAI-compatible error format
@app.exception_handler(HTTPException)
async def openai_exception_handler(request: Request, exc: HTTPException):
    """Convert FastAPI HTTPException to OpenAI-compatible error format"""

    # Determine error type based on status code
    if exc.status_code == 400:
        error_type = "invalid_request_error"
    elif exc.status_code == 401:
        error_type = "authentication_error"
    elif exc.status_code == 429:
        error_type = "rate_limit_error"
    elif exc.status_code == 500:
        error_type = "server_error"
    else:
        error_type = "api_error"

    error_response = {
        "error": {
            "message": exc.detail,
            "type": error_type,
            "code": str(exc.status_code)
        }
    }

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

shepherd: Optional[ShepherdClient] = None
parent_pid: int = 0  # C++ parent process PID for sending cancellation signals


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: Request, request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""

    if request.stream:
        raise HTTPException(400, "Streaming not yet supported")

    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

    parameters = {
        "temperature": request.temperature,
        "top_p": request.top_p
    }

    # Only include max_tokens if client explicitly provided it
    if request.max_tokens is not None:
        parameters["max_tokens"] = request.max_tokens

    # Forward tools if provided
    tools = None
    if request.tools:
        tools = [tool for tool in request.tools]

    # Run shepherd.generate() in a thread pool so we can check for disconnects
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    # Start generation in background thread
    generation_task = loop.run_in_executor(executor, shepherd.generate, messages, parameters, tools)

    # Check for disconnects while generation is running
    disconnect_detected = False
    while not generation_task.done():
        if await req.is_disconnected():
            disconnect_detected = True
            # Client disconnected - send SIGUSR1 to parent C++ process
            try:
                os.kill(parent_pid, signal.SIGUSR1)
            except Exception as e:
                print(f"Failed to send cancellation signal: {e}", file=sys.stderr)
            break
        # Check every 100ms
        await asyncio.sleep(0.1)

    # If disconnect detected, wait a bit for cancellation to take effect, then raise error
    if disconnect_detected:
        await asyncio.sleep(0.5)  # Give C++ time to cancel
        raise HTTPException(499, "Client disconnected")

    try:
        shepherd_response = await generation_task
    except Exception as e:
        error_msg = str(e)
        # Check if this is a context limit error (client error, not server error)
        # Return 400 with OpenAI-compatible error format
        if any(keyword in error_msg.lower() for keyword in ["context limit", "context window", "maximum context", "context length", "exceeded"]):
            error_response = OpenAIErrorResponse(
                error=OpenAIError(
                    message=error_msg,
                    type="invalid_request_error",
                    code="context_length_exceeded"
                )
            )
            return JSONResponse(status_code=400, content=error_response.model_dump())
        # All other errors are 500
        raise HTTPException(500, f"{error_msg}")

    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": shepherd_response.get("content", "")
        },
        "finish_reason": shepherd_response.get("finish_reason", "stop")
    }

    if shepherd_response.get("tool_calls"):
        choice["message"]["tool_calls"] = shepherd_response["tool_calls"]

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage=shepherd_response.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    )

    return response


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    try:
        models = shepherd.list_models()
    except Exception as e:
        error_msg = str(e)
        # Check if this is a context limit error (client error, not server error)
        if any(keyword in error_msg.lower() for keyword in ["context limit", "context window", "maximum context", "context length", "exceeded"]):
            raise HTTPException(400, error_msg)
        raise HTTPException(500, f"{error_msg}")

    model_data = [
        ModelInfo(
            id=model.get("id", "unknown"),
            created=int(time.time()),
            max_model_len=model.get("max_model_len")
        )
        for model in models
    ]

    return ModelsResponse(data=model_data)


@app.get("/v1/models/{model_name}")
async def get_model(model_name: str):
    """Get specific model information"""
    try:
        model_info = shepherd.get_model_info()
    except Exception as e:
        error_msg = str(e)
        # Check if this is a context limit error (client error, not server error)
        if any(keyword in error_msg.lower() for keyword in ["context limit", "context window", "maximum context", "context length", "exceeded"]):
            raise HTTPException(400, error_msg)
        raise HTTPException(500, f"{error_msg}")

    # Return model info regardless of what name was requested
    # (since we only have one "shepherd" model)
    return {
        "id": model_info.get("id", "shepherd"),
        "object": model_info.get("object", "model"),
        "created": model_info.get("created", int(time.time())),
        "owned_by": model_info.get("owned_by", "shepherd"),
        "context_window": model_info.get("context_window", 128000),
        "backend": model_info.get("backend", "unknown"),
        "model_name": model_info.get("model_name", "unknown")
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "shepherd_connected": shepherd is not None
    }


# ============================================================================
# Main
# ============================================================================

def main():
    global shepherd, parent_pid

    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True,
                       help="Socket file descriptor from parent process")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--parent-pid", type=int, required=True,
                       help="Parent C++ process PID for cancellation signaling")
    args = parser.parse_args()

    # Store parent PID for cancellation signaling
    parent_pid = args.parent_pid

    try:
        shepherd = ShepherdClient(args.fd)
    except Exception as e:
        print(f"Error: Failed to connect to Shepherd: {e}", file=sys.stderr)
        sys.exit(1)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
