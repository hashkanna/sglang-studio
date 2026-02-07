export type Run = {
  id: string;
  backend: string;
  prompt: string;
  parameters: Record<string, unknown>;
  status: string;
  result_json: Record<string, unknown> | null;
  artifact_key: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

export type CompareResponse = {
  left_run_id: string;
  right_run_id: string;
  score_abs_diff: number;
  latency_ms_diff: number;
  latency_pct_diff: number;
  throughput_items_per_s_diff: number;
};
