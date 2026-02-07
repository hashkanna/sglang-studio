export type Run = {
  id: string;
  backend: string;
  mode: string;
  prompt: string;
  parameters: Record<string, unknown>;
  score_input: Record<string, unknown> | null;
  mask_config: Record<string, unknown> | null;
  tolerance: Record<string, unknown> | null;
  repro_metadata: Record<string, unknown> | null;
  score_input_hash: string | null;
  mask_hash: string | null;
  status: string;
  result_json: Record<string, unknown> | null;
  artifact_key: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

export type TokenDiffRow = {
  index: number;
  token: string;
  left_logprob: number;
  right_logprob: number;
  abs_diff: number;
  rel_diff: number;
  is_match: boolean;
};

export type CompareResponse = {
  left_run_id: string;
  right_run_id: string;
  score_abs_diff: number;
  latency_ms_diff: number;
  latency_pct_diff: number;
  throughput_items_per_s_diff: number;
  token_abs_epsilon: number;
  token_rel_epsilon: number;
  token_parity_pass: boolean;
  score_parity_pass: boolean;
  score_parity_threshold: number;
  latency_regression_pass: boolean;
  latency_regression_pct_threshold: number;
  overall_pass: boolean;
  token_data_available: boolean;
  token_pair_count: number;
  token_mismatch_count: number;
  first_divergence_index: number | null;
  max_token_abs_diff: number;
  mean_token_abs_diff: number;
  token_diffs: TokenDiffRow[];
  token_loss_diff_summary: {
    pair_count: number;
    max_abs_nll_diff: number;
    mean_abs_nll_diff: number;
  };
  rank_delta_summary: {
    pair_count: number;
    worst_rank_drop: number;
    mean_abs_rank_delta: number;
    mrr_left: number;
    mrr_right: number;
    mrr_delta: number;
  };
};
