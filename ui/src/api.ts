import type { CompareResponse, Run } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export async function createRun(payload: {
  backend: "sglang-jax" | "sglang-pytorch" | "mock";
  prompt: string;
  parameters?: Record<string, unknown>;
}): Promise<Run> {
  const res = await fetch(`${API_BASE_URL}/api/v1/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseJson<Run>(res);
}

export async function listRuns(limit = 50): Promise<Run[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/runs?limit=${limit}`);
  return parseJson<Run[]>(res);
}

export async function compareRuns(leftRunId: string, rightRunId: string): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/compares`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ left_run_id: leftRunId, right_run_id: rightRunId })
  });
  return parseJson<CompareResponse>(res);
}
