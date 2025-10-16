# Shepherd OpenAI-Compatible API Server

FastAPI-based HTTP server providing OpenAI-compatible REST API for Shepherd.

## Architecture

- Parent process: Shepherd (C++) handles inference and tool execution
- Child process: FastAPI server handles HTTP/REST endpoints
- Communication: Unix domain socket (bidirectional pipe)
- Protocol: JSON lines over socket

## Usage

Start server mode from Shepherd:

```bash
shepherd --server --port 8080 --backend llamacpp --model llama-3-8b.gguf
```

## API Endpoints

- `POST /v1/chat/completions` - Chat completions (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- This server is spawned automatically by Shepherd in server mode
- Do not run api_server.py directly
- Tools are NOT executed by Shepherd in server mode (client-side execution)
