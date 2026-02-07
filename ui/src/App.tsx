import { useEffect, useMemo, useState } from "react";
import { compareRuns, createRun, listRuns, type RunCreatePayload } from "./api";
import type { CompareResponse, Run } from "./types";

function shortId(id: string): string {
  return id.slice(0, 8);
}

export function App() {
  const [prompt, setPrompt] = useState("Rate this answer quality from 0-1: Paris is the capital of France.");
  const [runMode, setRunMode] = useState<"benchmark" | "score">("benchmark");
  const [backend, setBackend] = useState<"sglang-jax" | "sglang-pytorch" | "mock">("sglang-jax");
  const [multiItemCount, setMultiItemCount] = useState(1);
  const [scoreQuery, setScoreQuery] = useState("Is this statement true or false?");
  const [scoreItemsText, setScoreItemsText] = useState(" true\n false");
  const [labelTokenIds, setLabelTokenIds] = useState("");
  const [applySoftmax, setApplySoftmax] = useState(true);
  const [itemFirst, setItemFirst] = useState(false);
  const [maskPreset, setMaskPreset] = useState<"none" | "causal" | "bidirectional-prefix" | "doc-isolation" | "custom">("none");
  const [customMaskJson, setCustomMaskJson] = useState("[[1,1],[0,1]]");
  const [absEpsilon, setAbsEpsilon] = useState("0.000001");
  const [relEpsilon, setRelEpsilon] = useState("0.0");
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [leftRunId, setLeftRunId] = useState("");
  const [rightRunId, setRightRunId] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  async function refreshRuns() {
    try {
      const data = await listRuns(100);
      setRuns(data);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  useEffect(() => {
    void refreshRuns();
    const timer = window.setInterval(() => {
      void refreshRuns();
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  const scoreItems = useMemo(() => {
    return scoreItemsText
      .split("\n")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  }, [scoreItemsText]);

  const latestByBackend = useMemo(() => {
    const latestJax = runs.find((r) => r.backend === "sglang-jax" && r.status === "succeeded");
    const latestTorch = runs.find((r) => r.backend === "sglang-pytorch" && r.status === "succeeded");
    return { latestJax, latestTorch };
  }, [runs]);

  async function onSubmitRun() {
    setLoading(true);
    setError(null);
    try {
      let payload: RunCreatePayload;
      if (runMode === "benchmark") {
        payload = {
          backend,
          mode: "benchmark",
          prompt,
          parameters: {
            multi_item_count: multiItemCount
          }
        };
      } else {
        const absVal = Number(absEpsilon);
        const relVal = Number(relEpsilon);
        if (!Number.isFinite(absVal) || absVal < 0 || !Number.isFinite(relVal) || relVal < 0) {
          throw new Error("Tolerance must be numeric and non-negative.");
        }
        if (scoreQuery.trim().length === 0 || scoreItems.length === 0) {
          throw new Error("Score mode requires a query and at least one non-empty item.");
        }
        const parsedTokenIds = labelTokenIds
          .split(",")
          .map((part) => part.trim())
          .filter((part) => part.length > 0)
          .map((part) => Number(part))
          .filter((num) => Number.isInteger(num));

        let customMask: unknown | undefined;
        if (maskPreset === "custom") {
          if (customMaskJson.trim().length === 0) {
            throw new Error("Custom mask preset requires JSON content.");
          }
          try {
            customMask = JSON.parse(customMaskJson);
          } catch {
            throw new Error("Custom mask JSON is invalid.");
          }
        }

        payload = {
          backend,
          mode: "score",
          prompt: scoreQuery,
          score_input: {
            query: scoreQuery,
            items: scoreItems,
            label_token_ids: parsedTokenIds,
            apply_softmax: applySoftmax,
            item_first: itemFirst
          },
          mask_config: {
            preset: maskPreset,
            custom_mask: customMask
          },
          tolerance: {
            abs_epsilon: absVal,
            rel_epsilon: relVal
          }
        };
      }

      const run = await createRun(payload);
      await refreshRuns();
      if (run.backend === "sglang-jax") {
        setLeftRunId(run.id);
      }
      if (run.backend === "sglang-pytorch") {
        setRightRunId(run.id);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function onCompare() {
    setLoading(true);
    setError(null);
    try {
      const result = await compareRuns(leftRunId, rightRunId);
      setCompareResult(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function useLatestSucceededPair() {
    if (!latestByBackend.latestJax || !latestByBackend.latestTorch) {
      return;
    }
    setLeftRunId(latestByBackend.latestJax.id);
    setRightRunId(latestByBackend.latestTorch.id);
  }

  const canSubmitBenchmark = prompt.trim().length > 0;
  const canSubmitScore =
    scoreQuery.trim().length > 0 &&
    scoreItems.length > 0 &&
    (maskPreset !== "custom" || customMaskJson.trim().length > 0);

  return (
    <div className="studio-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">SGLang Studio</p>
          <h1>Interactive Parity and Performance Lab</h1>
          <p className="subtitle">
            Submit benchmark or structured score runs, inspect parity gates, and debug token-level score differences.
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <h2>Create Run</h2>
          <label>Backend</label>
          <select value={backend} onChange={(e) => setBackend(e.target.value as typeof backend)}>
            <option value="sglang-jax">sglang-jax</option>
            <option value="sglang-pytorch">sglang-pytorch</option>
            <option value="mock">mock</option>
          </select>

          <label>Mode</label>
          <select value={runMode} onChange={(e) => setRunMode(e.target.value as typeof runMode)}>
            <option value="benchmark">benchmark</option>
            <option value="score">score</option>
          </select>

          {runMode === "benchmark" ? (
            <>
              <label>Prompt</label>
              <textarea rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} />

              <label>Multi-item count</label>
              <input
                type="number"
                min={1}
                max={128}
                value={multiItemCount}
                onChange={(e) => setMultiItemCount(Number(e.target.value))}
              />
            </>
          ) : (
            <>
              <label>Score Query</label>
              <textarea rows={4} value={scoreQuery} onChange={(e) => setScoreQuery(e.target.value)} />

              <label>Score Items (one per line)</label>
              <textarea rows={4} value={scoreItemsText} onChange={(e) => setScoreItemsText(e.target.value)} />

              <label>Label Token IDs (comma-separated, optional)</label>
              <input
                value={labelTokenIds}
                onChange={(e) => setLabelTokenIds(e.target.value)}
                placeholder="9454,2753"
              />

              <div className="inline-row">
                <div>
                  <label>Apply Softmax</label>
                  <select value={String(applySoftmax)} onChange={(e) => setApplySoftmax(e.target.value === "true")}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                </div>
                <div>
                  <label>Item First</label>
                  <select value={String(itemFirst)} onChange={(e) => setItemFirst(e.target.value === "true")}>
                    <option value="false">false</option>
                    <option value="true">true</option>
                  </select>
                </div>
              </div>

              <label>Mask Preset</label>
              <select value={maskPreset} onChange={(e) => setMaskPreset(e.target.value as typeof maskPreset)}>
                <option value="none">none</option>
                <option value="causal">causal</option>
                <option value="bidirectional-prefix">bidirectional-prefix</option>
                <option value="doc-isolation">doc-isolation</option>
                <option value="custom">custom</option>
              </select>

              {maskPreset === "custom" ? (
                <>
                  <label>Custom Mask JSON</label>
                  <textarea rows={4} value={customMaskJson} onChange={(e) => setCustomMaskJson(e.target.value)} />
                </>
              ) : null}

              <div className="inline-row">
                <div>
                  <label>Abs Epsilon</label>
                  <input value={absEpsilon} onChange={(e) => setAbsEpsilon(e.target.value)} />
                </div>
                <div>
                  <label>Rel Epsilon</label>
                  <input value={relEpsilon} onChange={(e) => setRelEpsilon(e.target.value)} />
                </div>
              </div>
            </>
          )}

          <button disabled={loading || (runMode === "benchmark" ? !canSubmitBenchmark : !canSubmitScore)} onClick={onSubmitRun}>
            {loading ? "Submitting..." : "Submit Run"}
          </button>
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Compare Runs</h2>
            <button className="ghost" onClick={useLatestSucceededPair}>Use Latest Succeeded Pair</button>
          </div>

          <label>Left Run ID</label>
          <input value={leftRunId} onChange={(e) => setLeftRunId(e.target.value)} placeholder="JAX run id" />

          <label>Right Run ID</label>
          <input value={rightRunId} onChange={(e) => setRightRunId(e.target.value)} placeholder="PyTorch run id" />

          <button disabled={loading || !leftRunId || !rightRunId} onClick={onCompare}>
            Compare
          </button>

          {compareResult ? (
            <>
              <div className="chip-row">
                <span className={`chip ${compareResult.overall_pass ? "pass" : "fail"}`}>
                  overall: {compareResult.overall_pass ? "PASS" : "FAIL"}
                </span>
                <span className={`chip ${compareResult.score_parity_pass ? "pass" : "fail"}`}>
                  score gate: {compareResult.score_parity_pass ? "PASS" : "FAIL"}
                </span>
                <span className={`chip ${compareResult.latency_regression_pass ? "pass" : "fail"}`}>
                  latency gate: {compareResult.latency_regression_pass ? "PASS" : "FAIL"}
                </span>
                <span className={`chip ${compareResult.token_parity_pass ? "pass" : "fail"}`}>
                  token gate: {compareResult.token_parity_pass ? "PASS" : "FAIL"}
                </span>
              </div>

              <div className="compare-box">
                <div>
                  <span>Score abs diff</span>
                  <strong>{compareResult.score_abs_diff.toFixed(6)}</strong>
                </div>
                <div>
                  <span>Latency diff (ms)</span>
                  <strong>{compareResult.latency_ms_diff.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Latency diff (%)</span>
                  <strong>{compareResult.latency_pct_diff.toFixed(2)}%</strong>
                </div>
                <div>
                  <span>Throughput diff</span>
                  <strong>{compareResult.throughput_items_per_s_diff.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Token mismatches</span>
                  <strong>{compareResult.token_mismatch_count}</strong>
                </div>
                <div>
                  <span>First divergence index</span>
                  <strong>{compareResult.first_divergence_index ?? "-"}</strong>
                </div>
              </div>

              {compareResult.token_diffs.length > 0 ? (
                <div className="table-wrap token-diff-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Token</th>
                        <th>Left</th>
                        <th>Right</th>
                        <th>Abs Diff</th>
                        <th>Rel Diff</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.token_diffs.slice(0, 80).map((row) => (
                        <tr key={row.index} className={row.is_match ? "" : "mismatch-row"}>
                          <td>{row.index}</td>
                          <td>{row.token}</td>
                          <td>{row.left_logprob.toFixed(6)}</td>
                          <td>{row.right_logprob.toFixed(6)}</td>
                          <td>{row.abs_diff.toExponential(2)}</td>
                          <td>{row.rel_diff.toExponential(2)}</td>
                          <td>{row.is_match ? "PASS" : "FAIL"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted-note">No token-level payload available in one or both runs.</p>
              )}
            </>
          ) : null}
        </section>

        <section className="card wide">
          <div className="card-head">
            <h2>Run History</h2>
            <button className="ghost" onClick={() => void refreshRuns()}>Refresh</button>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Backend</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Score</th>
                  <th>Latency (ms)</th>
                  <th>Throughput</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const result = run.result_json ?? {};
                  return (
                    <tr key={run.id}>
                      <td title={run.id}>{shortId(run.id)}</td>
                      <td>{run.backend}</td>
                      <td>{run.mode}</td>
                      <td>
                        <span className={`status ${run.status}`}>{run.status}</span>
                      </td>
                      <td>{typeof result.score === "number" ? result.score.toFixed(6) : "-"}</td>
                      <td>{typeof result.latency_ms === "number" ? result.latency_ms.toFixed(3) : "-"}</td>
                      <td>
                        {typeof result.throughput_items_per_s === "number"
                          ? result.throughput_items_per_s.toFixed(3)
                          : "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {error ? <aside className="error">{error}</aside> : null}
    </div>
  );
}
