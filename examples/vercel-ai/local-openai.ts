// Example: Using Vercel AI SDK against local OpenAI-compatible MLX server
//
// Prereqs:
// - Start the server (from project root):
//   make run HOST=127.0.0.1 PORT=8080 MODEL=mlx-community/qwen3-4b-4bit-DWQ
// - Install deps in this subfolder (from examples/vercel-ai):
//   pnpm install
// - Run with tsx or ts-node, e.g.:
//   pnpm dlx tsx local-openai.ts
//
// Notes:
// - The server does not require an API key, but the OpenAI provider expects one; use any string.
// - If your server expects the full repo id, change MODEL_ID below to
//   "mlx-community/qwen3-4b-4bit-DWQ".

import { createOpenAI } from "@ai-sdk/openai";
import { generateText, streamText } from "ai";
import process from "node:process";

const BASE_URL = process.env.MLX_OPENAI_BASE_URL ?? "http://localhost:8080/v1";
const API_KEY = process.env.MLX_OPENAI_API_KEY ?? "dummy";
const MODEL_ID = process.env.MLX_MODEL_ID ?? "mlx-community/qwen3-4b-4bit-DWQ";

const openai = createOpenAI({
  baseURL: BASE_URL,
  apiKey: API_KEY,
  // name: "openai" // optional; defaults to 'openai'
});

async function runBasicGeneration(): Promise<void> {
  console.log("\n=== generateText (non-streaming) ===\n");
  const { text, usage } = await generateText({
    // Use chat API to hit /v1/chat/completions on the server
    model: openai.chat(MODEL_ID),
    prompt: "Give me two bullet points on why MLX is great for Apple Silicon.",
    temperature: 0.7,
    maxRetries: 0,
  });

  console.log(text);
  if (usage) {
    console.log("\nUsage:", usage);
  }
}

async function runStreamingGeneration(): Promise<void> {
  console.log("\n=== streamText (streaming) ===\n");
  const result = await streamText({
    // Use chat API to hit /v1/chat/completions on the server
    model: openai.chat(MODEL_ID),
    messages: [
      { role: "system", content: "You are a concise assistant." },
      { role: "user", content: "Count from 1 to 5, brief words only." },
    ],
    temperature: 0.7,
  });

  process.stdout.write("Streaming: ");
  for await (const delta of result.textStream) {
    process.stdout.write(delta);
  }
  process.stdout.write("\n");

  const finalText = await result.text;
  console.log("\nFinal text:\n", finalText);
}

async function main(): Promise<void> {
  console.log("Connecting to:", BASE_URL);
  console.log("Model:", MODEL_ID);

  await runBasicGeneration();
  await runStreamingGeneration();
}

main().catch((err) => {
  console.error("Error:", err);
  process.exitCode = 1;
});
