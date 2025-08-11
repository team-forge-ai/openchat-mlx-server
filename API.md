## MLX‑LM Server REST API

This project is a thin wrapper around upstream `mlx_lm.server`; all HTTP behavior is implemented by MLX‑LM. The API is largely OpenAI‑compatible for chat completions.

- Base URL: `http://HOST:PORT`
- Default: `http://127.0.0.1:8080`
- Content type: `application/json`

### Endpoints

- GET `/health` — Basic health check
- GET `/v1/models` — List locally available models
- POST `/v1/chat/completions` — OpenAI‑style Chat Completions (supports streaming)
- POST `/v1/completions` — Legacy text completions (non‑chat)

Notes:

- If the server is started with a specific `--model`, you may omit `model` in requests. Otherwise include it in the payload.
- Authentication is not enforced by upstream; do not expose this server publicly.

---

### GET /health

Simple readiness probe.

Request:

```bash
curl -i http://127.0.0.1:8080/health
```

Response:

- 200 OK on success (plain text or minimal JSON; exact body not guaranteed)

---

### GET /v1/models

List models the server can see/use.

Request:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
      "object": "model",
      "created": 1721634359
    }
  ]
}
```

Fields:

- `data[].id`: HF repo ID or local identifier
- `data[].created`: timestamp (seconds)
- `data[].object`: always `model`

---

### POST /v1/chat/completions

OpenAI‑compatible chat completions. Supports non‑streaming and streaming.

Common request fields:

- `model` (string, optional): model id/path
- `messages` (array): chat history, each `{role: "system"|"user"|"assistant", content: string}`
- `max_tokens` (int, optional)
- Sampling: `temperature`, `top_p`, `top_k`, `min_p` (optional)
- Repetition control: `repetition_penalty`, `repetition_context_size` (optional)
- `stop` (string | string[], optional)
- `logprobs` (int, optional): returns token logprobs metadata
- `logit_bias` (object<int,string|number>, optional)
- Streaming: `stream` (bool, default false), `stream_options.include_usage` (bool)
- Speculative decoding: `draft_model` (string), `num_draft_tokens` (int)
- Adapters: `adapters` (string) for LoRA
- Advanced: `role_mapping`, `chat_template`, `chat_template_args`, `use_default_chat_template` (support depends on upstream version)

Non‑streaming example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen3-4b-4bit-DWQ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a haiku about apples."}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

Example response (truncated):

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1721634359,
  "model": "mlx-community/qwen3-4b-4bit-DWQ",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Crisp autumn apples\n..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36 }
}
```

Streaming example (Server‑Sent Events compatible, OpenAI style):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "stream": true
  }'
```

Stream format:

- Lines prefixed with `data: {json}` chunks, ending with `data: [DONE]`
- Each chunk has `object: "chat.completion.chunk"` and `choices[].delta.content`

Chunk example:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1721634359,
  "model": "default",
  "choices": [
    { "index": 0, "delta": { "content": "1 " }, "finish_reason": null }
  ]
}
```

---

### POST /v1/completions

Legacy, non‑chat text completion.

Request fields (subset):

- `model` (string, optional)
- `prompt` (string | string[])
- `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`
- `stop`, `logprobs`, `logit_bias`, `repetition_penalty`, `repetition_context_size`
- `stream` (bool)

Example:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "prompt": "Write a haiku about apples.",
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

---

### Compatibility and caveats

- The API is designed for compatibility with OpenAI SDKs. See `examples/openai_client_example.py` for usage.
- Upstream warns this server is not production‑grade and performs basic security checks only.
- Feature availability (e.g., `logprobs`, speculative decoding, custom chat templates) may vary by MLX‑LM version.

### Starting the server

Use either the wrapper or upstream directly:

```bash
# Wrapper (this repo)
python -m openchat_mlx_server.main --host 127.0.0.1 --port 8080 --model mlx-community/qwen3-4b-4bit-DWQ

# Upstream
python -m mlx_lm server --host 127.0.0.1 --port 8080 --model mlx-community/qwen3-4b-4bit-DWQ
```

For full CLI options, run:

```bash
python -m mlx_lm server --help
```
