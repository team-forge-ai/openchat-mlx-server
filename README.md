# openchat-mlx-server (thin wrapper around mlx_lm.server)

This project is a small wrapper around upstream `mlx_lm.server`. It provides:

- A convenient Python entry point: `python -m openchat_mlx_server.main`
- A Makefile target to run locally
- A PyInstaller build script for a standalone binary

All serving functionality (endpoints, streaming, speculative decoding, prompt cache reuse, tool-calling, logprobs, model listing) is provided by `mlx_lm.server`.

## Quick start

- Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Start the server (wrapper delegates to upstream):

```bash
# Without preloading a model
make run HOST=127.0.0.1 PORT=8080

# With a specific model (either a local relative path or a cached HF repo ID)
make run HOST=127.0.0.1 PORT=8080 MODEL=mlx-community/qwen3-4b-4bit-DWQ
```

Equivalent upstream command:

```bash
python -m mlx_lm server --host 127.0.0.1 --port 8080 --model mlx-community/qwen3-4b-4bit-DWQ
```

## API endpoints (provided by mlx_lm.server)

- GET /health
- GET /v1/models — lists downloaded HF models detectable by MLX
- POST /v1/chat/completions — OpenAI-compatible chat completions (supports streaming)
- POST /v1/completions — legacy text completions

Selected request options:

- model: model id or path (optional if started with `--model`)
- draft_model, num_draft_tokens: speculative decoding
- adapters: LoRA adapter path
- stream, stream_options.include_usage
- max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, repetition_context_size
- stop, logprobs, logit_bias

## Examples

See `examples/openai_client_example.py` for:

- Chat completions (streaming and non-streaming)
- Logprobs and basic tool-calling markers
- Model listing

Run:

```bash
python examples/openai_client_example.py --base-url http://localhost:8080/v1 --api-key dummy
```

## Build a standalone binary

```bash
python build.py            # builds ./dist/openchat-mlx-server
python build.py --dist     # also creates a tar.gz in the project root
```

## Notes

- All behavior and flags are inherited from upstream `mlx_lm.server`.
- For advanced usage (e.g., custom chat templates), refer to the MLX-LM documentation.

## License

MIT
