// Example: Tool (Function) Calling with Vercel AI SDK against local OpenAI-compatible MLX server
//
// Prereqs:
// - Server running locally (from project root):
//   make run HOST=127.0.0.1 PORT=8000 MODEL=mlx-community/qwen3-4b-4bit-DWQ
// - Install deps (from examples/vercel-ai):
//   pnpm install
// - Run:
//   MLX_OPENAI_BASE_URL=http://localhost:8000/v1 \
//   MLX_OPENAI_API_KEY=dummy \
//   MLX_MODEL_ID=mlx-community/qwen3-4b-4bit-DWQ \
//   pnpm dlx tsx local-openai-tools.ts

import { createOpenAI } from "@ai-sdk/openai";
import { generateText, stepCountIs, tool } from "ai";
import process from "node:process";
import { z } from "zod";

const BASE_URL = process.env.MLX_OPENAI_BASE_URL ?? "http://localhost:8080/v1";
const API_KEY = process.env.MLX_OPENAI_API_KEY ?? "dummy";
const MODEL_ID = process.env.MLX_MODEL_ID ?? "mlx-community/qwen3-4b-4bit-DWQ";

const openai = createOpenAI({ baseURL: BASE_URL, apiKey: API_KEY });

// Define a simple calculator tool using JSON Schema (no extra deps needed)
const tools = {
  add: tool({
    description: "Add two numbers and return the sum",
    inputSchema: z.object({
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
    async execute({ a, b }) {
      return { sum: a + b };
    },
  }),
};

async function runToolCalling(): Promise<void> {
  console.log("Connecting to:", BASE_URL);
  console.log("Model:", MODEL_ID);

  const a = 7;
  const b = 13;

  const result = await generateText({
    // Use the chat API so we hit /v1/chat/completions
    model: openai.chat(MODEL_ID),
    tools,
    // Force the model to call a tool
    toolChoice: "required",
    // Allow multi-step loop: tool call -> tool result -> final answer
    stopWhen: stepCountIs(5),
    messages: [
      {
        role: "system",
        content:
          "You are a helpful assistant that uses tools when appropriate.",
      },
      {
        role: "user",
        content: `Please add ${a} and ${b}. Return only the final numeric result.`,
      },
    ],
  });

  // Show intermediate steps (tool calls/results)
  if (result.steps?.length) {
    console.log("\nSteps:");
    for (const [idx, step] of result.steps.entries()) {
      const calls = step.toolCalls?.map((c) => ({
        toolName: c.toolName,
        input: c.input,
      }));
      const outputs = step.toolResults?.map((r) => ({
        toolName: r.toolName,
        output: r.output,
      }));
      console.log(`- Step ${idx + 1}:`, {
        calls,
        outputs,
        finishReason: step.finishReason,
      });
    }
  }

  console.log("\nFinal text:\n", result.text);
}

runToolCalling().catch((err) => {
  console.error("Error:", err);
  process.exitCode = 1;
});
