import { useState } from "react";

const V_FIELDS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const EMPTY = Object.fromEntries(["Amount", ...V_FIELDS].map((k) => [k, "0"]));
const pct = (p) => (p * 100 < 0.1 ? "<0.1%" : `${(p * 100).toFixed(1)}%`);

async function api(path, options) {
  const res = await fetch(`/api${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function RiskBar({ probability, threshold }) {
  return (
    <div className="riskbar" role="img" aria-label={`Fraud probability ${pct(probability)}, flag threshold ${pct(threshold)}`}>
      <div className="riskbar-fill" style={{ width: `${Math.max(probability * 100, 1)}%` }} />
      <div className="riskbar-mark" style={{ left: `${threshold * 100}%` }}>
        <span>Flag at {pct(threshold)}</span>
      </div>
    </div>
  );
}

function SingleReview() {
  const [values, setValues] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async (kind) => {
    setError("");
    try {
      const s = await api(`/sample?kind=${kind}`);
      setValues(Object.fromEntries(Object.entries(s).map(([k, v]) => [k, String(v)])));
      setResult(null);
    } catch (e) {
      setError(e.message);
    }
  };

  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const features = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v) || 0]));
      setResult(await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }) }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="split">
      <section>
        <h2>Transaction</h2>
        <div className="row">
          <button className="ghost" onClick={() => load("legit")}>Load a normal transaction</button>
          <button className="ghost" onClick={() => load("fraud")}>Load a known fraud case</button>
        </div>
        <div className="grid two">
          <label>Amount
            <input value={values.Amount} onChange={(e) => setValues({ ...values, Amount: e.target.value })} inputMode="decimal" />
          </label>
        </div>
        <details>
          <summary>Anonymized features V1–V28</summary>
          <div className="grid vgrid">
            {V_FIELDS.map((k) => (
              <label key={k}>{k}
                <input value={values[k]} onChange={(e) => setValues({ ...values, [k]: e.target.value })} inputMode="decimal" />
              </label>
            ))}
          </div>
        </details>
        <button className="primary" onClick={submit} disabled={busy}>{busy ? "Scoring…" : "Score transaction"}</button>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      <section className="verdict" aria-live="polite">
        <h2>Result</h2>
        {!result ? (
          <p className="empty">Load a sample or enter values, then score the transaction.</p>
        ) : (
          <>
            <p className={`verdict-word ${result.is_fraud ? "bad" : "good"}`}>
              {result.is_fraud ? "Likely fraud" : "Looks legitimate"}
            </p>
            <p className="verdict-sub">Fraud probability {pct(result.probability)}</p>
            <RiskBar probability={result.probability} threshold={result.threshold} />
            <h3>What drove this score</h3>
            <ul className="drivers">
              {result.drivers.map((d) => (
                <li key={d.feature}>
                  <span>{d.feature}</span>
                  <span className={d.impact > 0 ? "bad" : "good"}>
                    {d.impact > 0 ? "raises" : "lowers"} risk ({d.impact > 0 ? "+" : ""}{d.impact.toFixed(2)})
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}

function BatchReview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const upload = async (file) => {
    if (!file) return;
    setBusy(true);
    setError("");
    setData(null);
    try {
      const form = new FormData();
      form.append("file", file);
      setData(await api("/predict/batch", { method: "POST", body: form }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h2>Review a file</h2>
      <p className="hint">Upload a CSV with the columns V1–V28 and Amount (an id column is optional). Flagged transactions appear below, highest risk first.</p>
      <label className="file">
        <input type="file" accept=".csv" onChange={(e) => upload(e.target.files[0])} />
      </label>
      {busy && <p>Scoring…</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {data && (
        <>
          <p className="verdict-sub">{data.flagged} of {data.total} transactions flagged</p>
          <div className="tablewrap">
            <table>
              <thead><tr><th>ID</th><th>Amount</th><th>Fraud probability</th></tr></thead>
              <tbody>
                {data.rows.map((r, i) => (
                  <tr key={i}><td>{r.id}</td><td>{r.Amount.toFixed(2)}</td><td>{pct(r.probability)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState("single");
  return (
    <main>
      <header>
        <h1>Transaction review</h1>
        <nav role="tablist">
          <button role="tab" aria-selected={tab === "single"} onClick={() => setTab("single")}>One transaction</button>
          <button role="tab" aria-selected={tab === "batch"} onClick={() => setTab("batch")}>Upload a file</button>
        </nav>
      </header>
      {tab === "single" ? <SingleReview /> : <BatchReview />}
    </main>
  );
}
