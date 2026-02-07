import { useEffect, useMemo, useState } from "react";
import { compareRuns, createRun, listRuns } from "./api";
import type { CompareResponse, Run } from "./types";

function shortId(id: string): string {
  return id.slice(0, 8);
}

export function App() {
  const [prompt, setPrompt] = useState("Rate this answer quality from 0-1: Paris is the capital of France.");
  const [backend, setBackend] = useState<"sglang-jax" | "sglang-pytorch" | "mock">("sglang-jax");
  const [multiItemCount, setMultiItemCount] = useState(1);
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

  const latestByBackend = useMemo(() => {
    const latestJax = runs.find((r) => r.backend === "sglang-jax" && r.status === "succeeded");
    const latestTorch = runs.find((r) => r.backend === "sglang-pytorch" && r.status === "succeeded");
    return { latestJax, latestTorch };
  }, [runs]);

  async function onSubmitRun() {
    setLoading(true);
    setError(null);
    try {
      const run = await createRun({
        backend,
        prompt,
        parameters: {
          multi_item_count: multiItemCount
        }
      });
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

  return (
    <div className="studio-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">SGLang Studio</p>
          <h1>Interactive Parity and Performance Lab</h1>
          <p className="subtitle">
            Milestone 0 vertical slice: submit mock scoring runs, inspect status, and compare JAX vs PyTorch style outputs.
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

          <button disabled={loading || prompt.trim().length === 0} onClick={onSubmitRun}>
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
            </div>
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
