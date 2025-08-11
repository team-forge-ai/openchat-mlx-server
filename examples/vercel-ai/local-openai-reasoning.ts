// Example: Extracting <think> reasoning via middleware with Vercel AI SDK
//
// Prereqs:
//   make run HOST=127.0.0.1 PORT=8000 MODEL=mlx-community/qwen3-4b-4bit-DWQ
//   cd examples/vercel-ai && pnpm install
// Run:
//   MLX_OPENAI_BASE_URL=http://localhost:8000/v1 \
//   MLX_OPENAI_API_KEY=dummy \
//   MLX_MODEL_ID=mlx-community/qwen3-4b-4bit-DWQ \
//   pnpm dlx tsx local-openai-reasoning.ts

import { createOpenAI } from "@ai-sdk/openai";
import { extractReasoningMiddleware, streamText, wrapLanguageModel } from "ai";
import process from "node:process";

const BASE_URL = process.env.MLX_OPENAI_BASE_URL ?? "http://localhost:8080/v1";
const API_KEY = process.env.MLX_OPENAI_API_KEY ?? "dummy";
const MODEL_ID = process.env.MLX_MODEL_ID ?? "mlx-community/qwen3-4b-4bit-DWQ";

async function main() {
  console.log("Connecting to:", BASE_URL);
  console.log("Model:", MODEL_ID);

  const openai = createOpenAI({ baseURL: BASE_URL, apiKey: API_KEY });

  // Wrap the model with middleware that extracts <think> ... </think>
  const model = wrapLanguageModel({
    model: openai.chat(MODEL_ID),
    middleware: extractReasoningMiddleware({ tagName: "think" }),
  });

  const result = await streamText({
    model,
    messages: [
      { role: "system", content: "Be concise." },
      {
        role: "user",
        content:
          "Explain why MLX is fast on Apple Silicon in one short paragraph.",
      },
    ],
  });

  let visible = "";
  let reasoning = "";

  console.log("\n--- Streaming (visible + reasoning separately) ---\n");
  for await (const part of result.fullStream) {
    // The middleware emits dedicated reasoning events
    if (part.type === "reasoning-delta") {
      const chunk = (part as any).textDelta ?? (part as any).text ?? "";
      if (chunk) {
        process.stdout.write(chunk);
        reasoning += chunk;
      }
    } else if (part.type === "reasoning-start") {
      process.stdout.write("[reasoning]\n");
    } else if (part.type === "reasoning-end") {
      process.stdout.write("\n[/reasoning]\n\n");
    } else if (part.type === "text-delta") {
      const chunk = (part as any).textDelta ?? (part as any).text ?? "";
      if (chunk) {
        process.stdout.write(chunk);
        visible += chunk;
      }
    }
  }

  const finalText = await result.text;
  console.log("\n\n--- Final visible text ---\n", finalText);
  console.log("\n--- Captured reasoning (<think>) ---\n", reasoning.trim());
}

main().catch((err) => {
  console.error("Error:", err);
  process.exitCode = 1;
});
