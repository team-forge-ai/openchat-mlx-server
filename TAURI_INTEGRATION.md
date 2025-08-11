### Integrate openchat-mlx-server as a Tauri sidecar binary

This guide shows how to bundle and run the `openchat-mlx-server` binary as a Tauri sidecar and call its HTTP API.

## 1) Build the sidecar binary

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python build.py
# binary at: dist/openchat-mlx-server
chmod +x dist/openchat-mlx-server
```

## 2) Place the sidecar under src-tauri/binaries with platform suffix

Tauri looks for sidecars in `src-tauri/binaries` named as `<name>-<target_triple>[.exe]`.

- macOS Apple Silicon: `openchat-mlx-server-aarch64-apple-darwin`
- macOS Intel: `openchat-mlx-server-x86_64-apple-darwin`
- Linux x86_64 (GNU): `openchat-mlx-server-x86_64-unknown-linux-gnu`
- Windows x86_64 (MSVC): `openchat-mlx-server-x86_64-pc-windows-msvc.exe`

Find your host triple:

```bash
rustc -Vv | awk '/host/ {print $2}'
# e.g. aarch64-apple-darwin
```

Copy/rename:

```bash
cp dist/openchat-mlx-server src-tauri/binaries/openchat-mlx-server-"$(rustc -Vv | awk '/host/ {print $2}')"
```

## 3) Tauri configuration (v1 style)

`src-tauri/tauri.conf.json`:

```json
{
  "tauri": {
    "bundle": {
      "externalBin": ["binaries/openchat-mlx-server"]
    },
    "allowlist": {
      "shell": {
        "all": false,
        "execute": true,
        "sidecar": true,
        "scope": [
          {
            "name": "binaries/openchat-mlx-server",
            "cmd": "binaries/openchat-mlx-server"
          }
        ]
      }
    }
  }
}
```

Tauri v2 uses `@tauri-apps/plugin-shell`; the sidecar base name remains `binaries/openchat-mlx-server`.

## 4) Start/stop the sidecar from the frontend

```ts
// start_server.ts
import { Command } from "@tauri-apps/api/shell";

export async function startMlxServer(model?: string, port = 18080) {
  const args = ["--host", "127.0.0.1", "--port", String(port)];
  if (model) args.push("--model", model);

  const cmd = Command.sidecar("binaries/openchat-mlx-server", args);
  cmd.stdout.on("data", (l) => console.log("[mlx-server]", l));
  cmd.stderr.on("data", (l) => console.warn("[mlx-server:err]", l));

  const child = await cmd.spawn();
  return { child, port } as const;
}

export async function stopMlxServer(handle: {
  child: { kill: () => Promise<void> };
}) {
  try {
    await handle.child.kill();
  } catch {}
}
```

## 5) Call the API

```ts
export async function chatComplete(baseUrl: string, model: string) {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Say hi in one short sentence." },
      ],
      max_tokens: 64,
      temperature: 0.7,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json();
  return json.choices[0]?.message?.content ?? "";
}
```

Usage:

```ts
const { child, port } = await startMlxServer(
  "mlx-community/qwen3-4b-4bit-DWQ",
  18080
);
const baseUrl = `http://127.0.0.1:${port}`;
const answer = await chatComplete(baseUrl, "mlx-community/qwen3-4b-4bit-DWQ");
await stopMlxServer({ child });
```

## 6) Notes

- Choose a free port (e.g., 18080) or manage dynamically.
- Provide a `--model` at startup, or pass `model` in each request body.
- Streaming is available via SSE with `stream: true`.
- CORS headers are permissive by default.

## 7) Advanced (from mlx_lm.server)

- Speculative decoding: `--draft-model`, `--num-draft-tokens` or per-request.
- Prompt cache reuse/trim across requests.
- Tool-calling markers, `logprobs`, `logit_bias`.
- `/v1/models` to list cached models; `/health` for health.

## 8) Troubleshooting

- 404 on `/v1/chat/completions`: include a valid `model`.
- Slow first call: model may be downloading.
- Ensure the sidecar is executable and included via `externalBin`.
