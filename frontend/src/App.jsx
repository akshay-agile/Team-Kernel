import { useState, useEffect } from "react";
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
} from "recharts";

// ─── FIX 1: Correct port (5000) and API prefix (/api/) ───────────
const API = "http://localhost:5000";

// ─── Design tokens ───────────────────────────────────────────────
const C = {
  ocean:   "#0369a1",
  sky:     "#0ea5e9",
  seafoam: "#06b6d4",
  jade:    "#059669",
  amber:   "#d97706",
  coral:   "#dc2626",
  slate:   "#1e293b",
  mist:    "#f1f5f9",
  white:   "#ffffff",
  border:  "#e2e8f0",
  text:    "#0f172a",
  subtle:  "#64748b",
};

const RISK_PIE_DATA = [
  { name: "Low",    value: 50 },
  { name: "Medium", value: 40 },
  { name: "High",   value: 34 },
];
const PIE_COLORS = [C.jade, C.amber, C.coral];

const MONTHLY_DATA = [
  { month: "Jan", claims: 4,  detections: 12 },
  { month: "Feb", claims: 6,  detections: 18 },
  { month: "Mar", claims: 3,  detections: 15 },
  { month: "Apr", claims: 9,  detections: 22 },
  { month: "May", claims: 5,  detections: 19 },
  { month: "Jun", claims: 11, detections: 30 },
];

// ─── Shared Components ────────────────────────────────────────────
function MetricCard({ value, label, accent = C.ocean, icon }) {
  return (
    <div
      style={{
        background: C.white, border: `1px solid ${C.border}`,
        borderRadius: 16, padding: "24px 28px",
        display: "flex", alignItems: "center", gap: 20,
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        transition: "transform .2s, box-shadow .2s",
      }}
      onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 6px 20px rgba(0,0,0,0.1)"; }}
      onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.06)"; }}
    >
      <div style={{ width: 52, height: 52, borderRadius: 14, background: `${accent}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 30, fontWeight: 800, color: C.text, fontFamily: "'Syne', sans-serif", letterSpacing: "-1px" }}>{value}</div>
        <div style={{ fontSize: 13, color: C.subtle, marginTop: 2, fontWeight: 500 }}>{label}</div>
      </div>
    </div>
  );
}

function Badge({ level }) {
  const map = { HIGH: [C.coral, "#fff5f5"], MEDIUM: [C.amber, "#fffbeb"], LOW: [C.jade, "#f0fdf4"] };
  const [color, bg] = map[level] || [C.subtle, C.mist];
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}30`, borderRadius: 20, padding: "3px 12px", fontSize: 12, fontWeight: 700, letterSpacing: "0.04em" }}>
      {level}
    </span>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h2 style={{ fontSize: 22, fontWeight: 800, color: C.text, fontFamily: "'Syne', sans-serif", marginBottom: 20, letterSpacing: "-0.5px" }}>{title}</h2>
      {children}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 16, padding: 28, boxShadow: "0 1px 3px rgba(0,0,0,0.06)", ...style }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, variant = "primary", disabled = false, style = {} }) {
  const base = {
    padding: "10px 22px", borderRadius: 10, fontWeight: 700, fontSize: 14,
    cursor: disabled ? "not-allowed" : "pointer", border: "none",
    fontFamily: "inherit", transition: "all .18s",
    opacity: disabled ? 0.55 : 1,
    ...style,
  };
  const variants = {
    primary:   { background: C.ocean, color: "#fff" },
    secondary: { background: C.mist, color: C.slate, border: `1px solid ${C.border}` },
    danger:    { background: "#fef2f2", color: C.coral, border: `1px solid ${C.coral}30` },
  };
  return (
    <button style={{ ...base, ...variants[variant] }} onClick={onClick} disabled={disabled}
      onMouseEnter={e => { if (!disabled) { e.currentTarget.style.opacity = ".85"; e.currentTarget.style.transform = "translateY(-1px)"; } }}
      onMouseLeave={e => { e.currentTarget.style.opacity = disabled ? "0.55" : "1"; e.currentTarget.style.transform = ""; }}
    >{children}</button>
  );
}

function AlertBox({ type, children }) {
  const map = { success: [C.jade, "#f0fdf4"], warning: [C.amber, "#fffbeb"], error: [C.coral, "#fef2f2"], info: [C.ocean, "#eff6ff"] };
  const [color, bg] = map[type] || [C.subtle, C.mist];
  const icons = { success: "✓", warning: "⚠", error: "✕", info: "ℹ" };
  return (
    <div style={{ background: bg, border: `1px solid ${color}30`, borderRadius: 10, padding: "14px 18px", display: "flex", gap: 12, alignItems: "flex-start", marginBottom: 12 }}>
      <span style={{ color, fontWeight: 900, fontSize: 16, marginTop: 1 }}>{icons[type]}</span>
      <span style={{ color: C.text, fontSize: 14, lineHeight: 1.6 }}>{children}</span>
    </div>
  );
}

function FileUpload({ label, accept, onFile, fileName }) {
  return (
    <label style={{ display: "block", cursor: "pointer" }}>
      <div
        style={{ border: `2px dashed ${C.border}`, borderRadius: 12, padding: "32px 24px", textAlign: "center", background: C.mist, transition: "all .2s" }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = C.ocean; e.currentTarget.style.background = "#eff6ff"; }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = C.mist; }}
      >
        <div style={{ fontSize: 32, marginBottom: 8 }}>📁</div>
        <div style={{ fontWeight: 600, color: C.text, marginBottom: 4 }}>{fileName || label}</div>
        <div style={{ fontSize: 12, color: C.subtle }}>Click to browse or drag & drop</div>
        <input type="file" accept={accept} style={{ display: "none" }}
          onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
      </div>
    </label>
  );
}

// ─── Views ───────────────────────────────────────────────────────

function Dashboard() {
  return (
    <div>
      <Section title="Intelligence Overview">
        {/* FIX 2: 4-col grid stretches to full container width */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
          <MetricCard value="124" label="Properties Analysed" icon="🏠" accent={C.ocean} />
          <MetricCard value="38"  label="Pools Detected"       icon="🏊" accent={C.seafoam} />
          <MetricCard value="3"   label="Fraud Alerts"          icon="⚠️" accent={C.coral} />
          <MetricCard value="12"  label="IoT Inspections"       icon="📡" accent={C.jade} />
        </div>
      </Section>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 20, fontFamily: "'Syne',sans-serif" }}>Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={RISK_PIE_DATA} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={4} dataKey="value">
                {RISK_PIE_DATA.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 20, fontFamily: "'Syne',sans-serif" }}>Monthly Activity</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={MONTHLY_DATA}>
              <defs>
                <linearGradient id="gClaims" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.coral} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={C.coral} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gDetect" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.ocean} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={C.ocean} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="month" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="detections" stroke={C.ocean} fill="url(#gDetect)" strokeWidth={2} />
              <Area type="monotone" dataKey="claims"     stroke={C.coral} fill="url(#gClaims)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card style={{ marginTop: 24 }}>
        <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>System Pipeline</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "wrap" }}>
          {["Satellite Image","AI Pool Detection","Risk Engine","Insurance Dashboard","IoT Verification","Compliance Score","SMS Alert","Claims Analysis"].map((step, i, arr) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ background: `${C.ocean}12`, border: `1px solid ${C.ocean}30`, borderRadius: 10, padding: "8px 16px", fontSize: 13, fontWeight: 600, color: C.ocean, whiteSpace: "nowrap" }}>
                {step}
              </div>
              {i < arr.length - 1 && <span style={{ color: C.subtle, margin: "0 6px", fontSize: 18 }}>→</span>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function PoolDetection() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleFile = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const runDetection = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const fd = new FormData();
      // FIX: backend expects field name "image"
      fd.append("image", file);
      // FIX: correct path → /api/detect on port 5000
      const res = await fetch(`${API}/api/detect`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Detection failed (${res.status})`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Section title="Satellite Pool Detection">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <Card>
            <FileUpload label="Upload Satellite Image" accept="image/*" onFile={handleFile} fileName={file?.name} />
            {preview && (
              <div style={{ marginTop: 16 }}>
                <img src={preview} alt="preview" style={{ width: "100%", borderRadius: 10, maxHeight: 260, objectFit: "cover" }} />
              </div>
            )}
            <Btn style={{ marginTop: 16, width: "100%" }} onClick={runDetection} disabled={!file || loading}>
              {loading ? "⏳ Running AI Detection…" : "🔍 Run Detection"}
            </Btn>
            {error && <div style={{ marginTop: 12 }}><AlertBox type="error">{error}</AlertBox></div>}
          </Card>

          <Card>
            <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Risk Assessment</h3>
            {!result && <div style={{ color: C.subtle, fontSize: 14 }}>Upload an image and run detection to see results.</div>}
            {result && (
              <>
                {/* summary */}
                <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
                  <div style={{ background: C.mist, borderRadius: 10, padding: "10px 18px", fontSize: 13 }}>
                    <b>{result.summary?.total_pools ?? 0}</b> pools detected
                  </div>
                  <div style={{ background: C.mist, borderRadius: 10, padding: "10px 18px", fontSize: 13 }}>
                    Overall risk: <b>{result.summary?.overall_risk ?? "–"}</b>
                  </div>
                </div>

                {result.property_risks?.length === 0 && (
                  <AlertBox type="success">No pools detected on this property.</AlertBox>
                )}
                {result.property_risks?.map((r, i) => (
                  <div key={i} style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: 16, marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontWeight: 700, color: C.text }}>Property {r.property_id}</span>
                      <Badge level={r.risk_level} />
                    </div>
                    <div style={{ fontSize: 13, color: C.subtle }}>Risk Score: <b style={{ color: C.text }}>{r.risk_score}</b></div>
                    {r.risk_reasons?.map((reason, j) => (
                      <div key={j} style={{ fontSize: 12, color: C.subtle, marginTop: 4 }}>• {reason}</div>
                    ))}
                  </div>
                ))}

                {/* annotated image returned by backend */}
                {result.annotated_image && (
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image}`}
                    alt="annotated detection"
                    style={{ width: "100%", borderRadius: 10, marginTop: 12 }}
                  />
                )}
              </>
            )}
          </Card>
        </div>
      </Section>
    </div>
  );
}

function RiskAnalytics() {
  return (
    <Section title="Risk Analytics">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 20, fontFamily: "'Syne',sans-serif" }}>Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={RISK_PIE_DATA} cx="50%" cy="50%" outerRadius={100} paddingAngle={3} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                {RISK_PIE_DATA.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Risk Breakdown</h3>
          {[["Low Risk", 50, C.jade], ["Medium Risk", 40, C.amber], ["High Risk", 34, C.coral]].map(([label, count, color]) => (
            <div key={label} style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontWeight: 600, color: C.text, fontSize: 14 }}>{label}</span>
                <span style={{ color, fontWeight: 700 }}>{count} properties</span>
              </div>
              <div style={{ height: 8, background: C.mist, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${(count / 124) * 100}%`, height: "100%", background: color, borderRadius: 4, transition: "width 1s ease" }} />
              </div>
            </div>
          ))}
          <div style={{ marginTop: 24, padding: 16, background: C.mist, borderRadius: 12 }}>
            <div style={{ fontSize: 13, color: C.subtle, fontWeight: 600, marginBottom: 8 }}>KEY INSIGHTS</div>
            <div style={{ fontSize: 13, color: C.text, lineHeight: 1.7 }}>
              • 27% of properties flagged HIGH risk<br />
              • Uncovered pools account for 68% of high-risk cases<br />
              • Above-ground pools show 2.3× more claims frequency
            </div>
          </div>
        </Card>
      </div>
    </Section>
  );
}

function TimeSeries() {
  const [before, setBefore] = useState(null);
  const [after,  setAfter]  = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!before || !after) return;
    setLoading(true);
    try {
      const fd = new FormData();
      // FIX: backend expects "image_t1" and "image_t2"
      fd.append("image_t1", before);
      fd.append("image_t2", after);
      // FIX: correct path → /api/compare
      const res = await fetch(`${API}/api/compare`, { method: "POST", body: fd });
      const data = await res.json();
      // normalise response shape
      setResult({
        added:     data.comparison?.added     ?? 0,
        removed:   data.comparison?.removed   ?? 0,
        unchanged: data.comparison?.unchanged ?? 0,
        status:    data.claims_validation?.status   ?? "UNKNOWN",
        message:   data.claims_validation?.message  ?? "",
        severity:  data.claims_validation?.severity ?? "LOW",
        annotated_t1: data.t1?.annotated_image,
        annotated_t2: data.t2?.annotated_image,
      });
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Section title="Time-Series Claims Fraud Detection">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
        <Card>
          <h4 style={{ fontWeight: 700, color: C.text, marginBottom: 12 }}>Before Image (Pre-loss)</h4>
          <FileUpload label="Upload Before Image" accept="image/*" onFile={setBefore} fileName={before?.name} />
        </Card>
        <Card>
          <h4 style={{ fontWeight: 700, color: C.text, marginBottom: 12 }}>After Image (Post-loss)</h4>
          <FileUpload label="Upload After Image" accept="image/*" onFile={setAfter} fileName={after?.name} />
        </Card>
      </div>

      <Btn onClick={run} disabled={!before || !after || loading} style={{ marginBottom: 24 }}>
        {loading ? "⏳ Comparing Images…" : "🔍 Run Fraud Analysis"}
      </Btn>

      {result && !result.error && (
        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Analysis Result</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 20 }}>
            {[["Added Pools", result.added, C.coral], ["Removed Pools", result.removed, C.amber], ["Unchanged", result.unchanged, C.jade]].map(([label, val, color]) => (
              <div key={label} style={{ textAlign: "center", padding: 20, background: `${color}10`, borderRadius: 12, border: `1px solid ${color}30` }}>
                <div style={{ fontSize: 32, fontWeight: 900, color }}>{val}</div>
                <div style={{ fontSize: 13, color: C.subtle, marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>
          <AlertBox type={result.severity === "CRITICAL" || result.severity === "HIGH" ? "error" : result.severity === "MEDIUM" ? "warning" : "success"}>
            <b>{result.status}:</b> {result.message}
          </AlertBox>
          {(result.annotated_t1 || result.annotated_t2) && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
              {result.annotated_t1 && (
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: C.subtle, marginBottom: 6 }}>PRE-LOSS</div>
                  <img src={`data:image/jpeg;base64,${result.annotated_t1}`} alt="before" style={{ width: "100%", borderRadius: 10 }} />
                </div>
              )}
              {result.annotated_t2 && (
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: C.subtle, marginBottom: 6 }}>POST-LOSS</div>
                  <img src={`data:image/jpeg;base64,${result.annotated_t2}`} alt="after" style={{ width: "100%", borderRadius: 10 }} />
                </div>
              )}
            </div>
          )}
        </Card>
      )}
      {result?.error && <AlertBox type="error">{result.error}</AlertBox>}
    </Section>
  );
}

function IoTInspection() {
  const [sensorVal, setSensorVal] = useState(512);
  const [claimId,   setClaimId]   = useState("");
  const [propAddr,  setPropAddr]  = useState("");
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      // FIX: correct path → /api/iot/reading
      const res = await fetch(`${API}/api/iot/reading`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sensor_value:   sensorVal,
          claim_id:       claimId   || null,
          property_addr:  propAddr  || null,
          send_sms_if_high: true,
        }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const levelColor = sensorVal >= 700 ? C.coral : sensorVal >= 300 ? C.amber : C.jade;
  const levelLabel = sensorVal >= 700 ? "HIGH" : sensorVal >= 300 ? "MEDIUM" : "LOW";

  return (
    <Section title="IoT Sensor Inspection">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          {/* sensor value slider */}
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: "block", fontWeight: 600, color: C.text, marginBottom: 8 }}>
              Sensor Value (0–1023): <span style={{ color: levelColor, fontWeight: 900 }}>{sensorVal}</span>
              <span style={{ marginLeft: 10, fontSize: 12, background: `${levelColor}18`, color: levelColor, padding: "2px 10px", borderRadius: 20, fontWeight: 700 }}>{levelLabel}</span>
            </label>
            <input type="range" min={0} max={1023} value={sensorVal} onChange={e => setSensorVal(+e.target.value)}
              style={{ width: "100%", accentColor: levelColor }} />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span style={{ fontSize: 12, color: C.jade }}>Low Risk (0)</span>
              <span style={{ fontSize: 12, color: C.coral }}>High Risk (1023)</span>
            </div>
          </div>

          <div style={{ height: 10, background: C.mist, borderRadius: 5, marginBottom: 24, overflow: "hidden" }}>
            <div style={{ width: `${(sensorVal / 1023) * 100}%`, height: "100%", background: levelColor, borderRadius: 5, transition: "all .3s" }} />
          </div>

          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", fontWeight: 600, color: C.text, marginBottom: 8 }}>Claim ID (optional)</label>
            <input value={claimId} onChange={e => setClaimId(e.target.value)} placeholder="CLM-XXXXXXXX"
              style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit", boxSizing: "border-box" }} />
          </div>

          <div style={{ marginBottom: 24 }}>
            <label style={{ display: "block", fontWeight: 600, color: C.text, marginBottom: 8 }}>Property Address (optional)</label>
            <input value={propAddr} onChange={e => setPropAddr(e.target.value)} placeholder="123 Ocean Dr"
              style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit", boxSizing: "border-box" }} />
          </div>

          <Btn onClick={submit} style={{ width: "100%" }} disabled={loading}>
            {loading ? "Submitting…" : "📡 Submit Sensor Reading"}
          </Btn>

          {result && !result.error && (
            <div style={{ marginTop: 16 }}>
              <AlertBox type={result.risk_level === "HIGH" ? "error" : result.risk_level === "MEDIUM" ? "warning" : "success"}>
                {result.message} LED: {result.led_pct}%{result.sms_sent ? " | ✅ SMS sent" : ""}
              </AlertBox>
            </div>
          )}
          {result?.error && <div style={{ marginTop: 12 }}><AlertBox type="error">{result.error}</AlertBox></div>}
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Risk Thresholds</h3>
          {[
            ["0 – 299",   "LOW risk",    C.jade,  "No SMS alert. Standard monitoring."],
            ["300 – 699", "MEDIUM risk", C.amber, "Flagged for review. Inspector follow-up."],
            ["700 – 1023","HIGH risk",   C.coral, "Twilio SMS dispatched to insurer immediately."],
          ].map(([range, label, color, desc]) => (
            <div key={range} style={{ display: "flex", gap: 14, marginBottom: 20, alignItems: "flex-start" }}>
              <div style={{ minWidth: 72, padding: "6px 0", textAlign: "center", borderRadius: 10, background: `${color}15`, fontWeight: 900, color, fontSize: 11 }}>{range}</div>
              <div>
                <div style={{ fontWeight: 700, color: C.text, fontSize: 14 }}>{label}</div>
                <div style={{ fontSize: 13, color: C.subtle, marginTop: 2 }}>{desc}</div>
              </div>
            </div>
          ))}
        </Card>
      </div>
    </Section>
  );
}

function FraudAnalysis() {
  const [file,    setFile]    = useState(null);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      // FIX: correct path → /api/documents/analyze
      const res = await fetch(`${API}/api/documents/analyze`, { method: "POST", body: fd });
      const data = await res.json();
      // normalise response
      setResult({
        risk_score: data.final_risk_score,
        risk_level: data.risk_level?.toUpperCase(),
        reasons:    data.risk_reasons,
        ml_flag:    data.ml_flag,
        recommendation: data.recommendation,
        text_preview:   data.ocr_text_preview,
      });
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Section title="Document Fraud Detection">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <FileUpload label="Upload Property Document" accept="image/*,.pdf" onFile={setFile} fileName={file?.name} />
          <Btn style={{ marginTop: 16, width: "100%" }} onClick={analyze} disabled={!file || loading}>
            {loading ? "⏳ Analysing…" : "🔍 Analyse Document"}
          </Btn>
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Analysis Result</h3>
          {!result && <div style={{ color: C.subtle, fontSize: 14 }}>Upload a document to analyse for fraud indicators.</div>}
          {result && !result.error && (
            <>
              <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
                <div style={{ flex: 1, textAlign: "center", padding: 20, background: `${result.risk_level === "HIGH" ? C.coral : result.risk_level === "MEDIUM" ? C.amber : C.jade}10`, borderRadius: 12 }}>
                  <div style={{ fontSize: 24, fontWeight: 900, color: result.risk_level === "HIGH" ? C.coral : result.risk_level === "MEDIUM" ? C.amber : C.jade }}>{result.risk_score}</div>
                  <div style={{ fontSize: 12, color: C.subtle, marginTop: 4 }}>Risk Score</div>
                </div>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <Badge level={result.risk_level} />
                  {result.ml_flag && (
                    <span style={{ fontSize: 11, color: result.ml_flag === "Anomaly Detected" ? C.coral : C.jade, fontWeight: 600 }}>
                      ML: {result.ml_flag}
                    </span>
                  )}
                </div>
              </div>
              {result.recommendation && (
                <div style={{ padding: "10px 14px", background: C.mist, borderRadius: 10, marginBottom: 14, fontSize: 13, fontWeight: 600, color: C.text }}>
                  {result.recommendation}
                </div>
              )}
              <div style={{ fontWeight: 700, color: C.text, marginBottom: 10 }}>Fraud Indicators</div>
              {result.reasons?.map((r, i) => (
                <div key={i} style={{ fontSize: 13, color: C.text, padding: "8px 12px", background: C.mist, borderRadius: 8, marginBottom: 6 }}>• {r}</div>
              ))}
              {result.text_preview && (
                <>
                  <div style={{ fontWeight: 700, color: C.text, marginTop: 16, marginBottom: 8 }}>OCR Preview</div>
                  <div style={{ fontSize: 12, fontFamily: "monospace", background: C.mist, padding: 12, borderRadius: 8, maxHeight: 120, overflow: "auto", color: C.subtle }}>
                    {result.text_preview}
                  </div>
                </>
              )}
            </>
          )}
          {result?.error && <AlertBox type="error">{result.error}</AlertBox>}
        </Card>
      </div>
    </Section>
  );
}

function AIAssistant() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hello! I'm your AquaIntelligence Risk Assistant. Ask me about property risks, pool insurance, underwriting, or fraud detection." }
  ]);
  const [input,   setInput]   = useState("");
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(m => [...m, { role: "user", text: userMsg }]);
    setLoading(true);
    try {
      // FIX: correct path → /api/ai-assistant (if you add this route; falls back to error msg)
      const res = await fetch(`${API}/api/ai-assistant`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg }),
      });
      const data = await res.json();
      setMessages(m => [...m, { role: "assistant", text: data.answer ?? JSON.stringify(data) }]);
    } catch {
      setMessages(m => [...m, { role: "assistant", text: "Sorry, I couldn't connect to the AI service. Please ensure the backend is running on port 5000." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Section title="AI Risk Assistant">
      <Card style={{ display: "flex", flexDirection: "column", height: 520 }}>
        <div style={{ flex: 1, overflowY: "auto", marginBottom: 16, paddingRight: 4 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start", marginBottom: 14 }}>
              {m.role === "assistant" && (
                <div style={{ width: 34, height: 34, borderRadius: 10, background: `${C.ocean}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, marginRight: 10, flexShrink: 0 }}>🤖</div>
              )}
              <div style={{
                maxWidth: "72%", padding: "12px 16px", borderRadius: 14,
                background: m.role === "user" ? C.ocean : C.mist,
                color: m.role === "user" ? "#fff" : C.text,
                fontSize: 14, lineHeight: 1.6,
                borderBottomRightRadius: m.role === "user" ? 4 : 14,
                borderBottomLeftRadius: m.role === "assistant" ? 4 : 14,
              }}>
                {m.text}
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 34, height: 34, borderRadius: 10, background: `${C.ocean}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>🤖</div>
              <div style={{ padding: "12px 16px", background: C.mist, borderRadius: 14, borderBottomLeftRadius: 4, fontSize: 20, letterSpacing: 2 }}>···</div>
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <input value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && send()}
            placeholder="Ask about risk, underwriting, fraud detection…"
            style={{ flex: 1, padding: "12px 16px", borderRadius: 12, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit", outline: "none" }} />
          <Btn onClick={send} disabled={loading || !input.trim()}>Send →</Btn>
        </div>
      </Card>
    </Section>
  );
}

function PropertyRegister() {
  const [name,   setName]   = useState("");
  const [email,  setEmail]  = useState("");
  const [phone,  setPhone]  = useState("");
  const [policy, setPolicy] = useState("");
  const [addr,   setAddr]   = useState("");
  const [pool,   setPool]   = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      // FIX: correct path → /api/policyholder/register
      const res = await fetch(`${API}/api/policyholder/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, phone, policy_number: policy, property_addr: addr, pool_declared: pool }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = { width: "100%", padding: "11px 14px", borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit", boxSizing: "border-box", marginBottom: 16 };

  return (
    <Section title="Register Property">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Full Name</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="Jane Smith" style={inputStyle} />

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Email</label>
          <input value={email} onChange={e => setEmail(e.target.value)} placeholder="jane@example.com" style={inputStyle} />

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Phone</label>
          <input value={phone} onChange={e => setPhone(e.target.value)} placeholder="+1 555 000 0000" style={inputStyle} />

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Policy Number</label>
          <input value={policy} onChange={e => setPolicy(e.target.value)} placeholder="POL-000000" style={inputStyle} />

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Property Address</label>
          <input value={addr} onChange={e => setAddr(e.target.value)} placeholder="123 Ocean Drive, Miami, FL" style={inputStyle} />

          <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", marginBottom: 24 }}>
            <input type="checkbox" checked={pool} onChange={e => setPool(e.target.checked)} style={{ width: 18, height: 18, accentColor: C.ocean }} />
            <span style={{ fontWeight: 600, color: C.text }}>Pool Present on Property</span>
          </label>

          <Btn onClick={submit} disabled={loading || !name || !addr}>
            {loading ? "Registering…" : "Register Property"}
          </Btn>

          {result && !result.error && (
            <div style={{ marginTop: 12 }}>
              <AlertBox type="success">
                Property registered! ID: <b>{result.policyholder_id}</b>
              </AlertBox>
            </div>
          )}
          {result?.error && <div style={{ marginTop: 12 }}><AlertBox type="error">{result.error}</AlertBox></div>}
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>What Happens Next</h3>
          {[
            ["1", "AI satellite verification", "Our system scans your property via satellite imagery within 2–3 business days."],
            ["2", "Pool risk assessment",       "If a pool is present, an automated risk score is assigned."],
            ["3", "Premium calculation",        "Your premium is adjusted based on pool type, size, and cover status."],
            ["4", "IoT inspection offer",       "You may be offered an IoT-assisted inspection to lower your risk score."],
          ].map(([num, title, desc]) => (
            <div key={num} style={{ display: "flex", gap: 14, marginBottom: 20 }}>
              <div style={{ minWidth: 32, height: 32, borderRadius: 8, background: `${C.ocean}15`, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 900, color: C.ocean, fontSize: 14 }}>{num}</div>
              <div>
                <div style={{ fontWeight: 700, color: C.text, fontSize: 14 }}>{title}</div>
                <div style={{ fontSize: 13, color: C.subtle, marginTop: 2 }}>{desc}</div>
              </div>
            </div>
          ))}
        </Card>
      </div>
    </Section>
  );
}

function SubmitClaim() {
  const [form, setForm] = useState({ name: "", email: "", phone: "", property_addr: "", claim_type: "Water Damage", incident_date: "", estimated_loss: "", description: "", pool_declared: false });
  const [file, setFile]     = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const submit = async () => {
    setLoading(true);
    try {
      // FIX: correct path → /api/claims
      const res = await fetch(`${API}/api/claims`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, estimated_loss: parseFloat(form.estimated_loss) || 0 }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = { width: "100%", padding: "11px 14px", borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit", boxSizing: "border-box", marginBottom: 16 };

  return (
    <Section title="Submit Insurance Claim">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Full Name</label>
          <input value={form.name} onChange={e => set("name", e.target.value)} placeholder="Jane Smith" style={inputStyle} />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Email</label>
              <input value={form.email} onChange={e => set("email", e.target.value)} placeholder="jane@example.com" style={{ ...inputStyle }} />
            </div>
            <div>
              <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Phone</label>
              <input value={form.phone} onChange={e => set("phone", e.target.value)} placeholder="+1 555 000 0000" style={{ ...inputStyle }} />
            </div>
          </div>

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Property Address</label>
          <input value={form.property_addr} onChange={e => set("property_addr", e.target.value)} placeholder="123 Ocean Drive" style={inputStyle} />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Claim Type</label>
              <select value={form.claim_type} onChange={e => set("claim_type", e.target.value)} style={{ ...inputStyle, background: C.white }}>
                {["Water Damage","Pool Damage","Flood","Structural","Liability"].map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Incident Date</label>
              <input type="date" value={form.incident_date} onChange={e => set("incident_date", e.target.value)} style={{ ...inputStyle }} />
            </div>
          </div>

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Estimated Loss ($)</label>
          <input type="number" value={form.estimated_loss} onChange={e => set("estimated_loss", e.target.value)} placeholder="5000" style={inputStyle} />

          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 6 }}>Incident Description</label>
          <textarea value={form.description} onChange={e => set("description", e.target.value)} rows={3}
            placeholder="Describe the incident in detail…"
            style={{ ...inputStyle, resize: "vertical" }} />

          <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", marginBottom: 20 }}>
            <input type="checkbox" checked={form.pool_declared} onChange={e => set("pool_declared", e.target.checked)} style={{ width: 18, height: 18, accentColor: C.ocean }} />
            <span style={{ fontWeight: 600, color: C.text }}>Pool Involved in Incident</span>
          </label>

          <Btn onClick={submit} disabled={loading || !form.name || !form.property_addr}>
            {loading ? "Submitting…" : "Submit Claim"}
          </Btn>

          {result && !result.error && (
            <div style={{ marginTop: 12 }}>
              <AlertBox type="success">
                Claim submitted! Reference: <b>{result.claim_id}</b><br />
                {result.message}
              </AlertBox>
            </div>
          )}
          {result?.error && <div style={{ marginTop: 12 }}><AlertBox type="error">{result.error}</AlertBox></div>}
        </Card>

        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Upload Evidence</h3>
          <FileUpload label="Upload Evidence (photos, documents)" accept="image/*,.pdf" onFile={setFile} fileName={file?.name} />
          <div style={{ marginTop: 24, padding: 16, background: C.mist, borderRadius: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 8 }}>After Submission</div>
            <div style={{ fontSize: 13, color: C.subtle, lineHeight: 1.8 }}>
              1. AI satellite verification within 24 hours<br />
              2. IoT inspection may be scheduled<br />
              3. Underwriter reviews risk report<br />
              4. Decision communicated via email & SMS
            </div>
          </div>
        </Card>
      </div>
    </Section>
  );
}

function TrackClaim() {
  const [claimId, setClaimId] = useState("");
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  const lookup = async () => {
    if (!claimId.trim()) return;
    setLoading(true);
    try {
      // FIX: correct path → /api/claims/{id}
      const res = await fetch(`${API}/api/claims/${claimId.trim()}`);
      if (!res.ok) throw new Error("Claim not found");
      setResult(await res.json());
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const STEPS = ["PENDING","AI_VERIFICATION","IOT_INSPECTION","UNDERWRITING_REVIEW","APPROVED / REJECTED"];

  return (
    <Section title="Track Claim Status">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <Card>
          <label style={{ fontWeight: 600, color: C.text, display: "block", marginBottom: 8 }}>Enter Claim Reference</label>
          <div style={{ display: "flex", gap: 10 }}>
            <input value={claimId} onChange={e => setClaimId(e.target.value)}
              onKeyDown={e => e.key === "Enter" && lookup()}
              placeholder="CLM-XXXXXXXX"
              style={{ flex: 1, padding: "11px 14px", borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 14, fontFamily: "inherit" }} />
            <Btn onClick={lookup} disabled={loading || !claimId.trim()}>
              {loading ? "…" : "Track"}
            </Btn>
          </div>

          {result && !result.error && (
            <div style={{ marginTop: 24 }}>
              {/* progress bar */}
              <div style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  {STEPS.map((s, i) => (
                    <div key={i} style={{ fontSize: 10, color: (result.progress_step ?? 1) > i ? C.ocean : C.subtle, fontWeight: (result.progress_step ?? 1) > i ? 700 : 400, textAlign: "center", flex: 1 }}>
                      {i + 1}
                    </div>
                  ))}
                </div>
                <div style={{ height: 6, background: C.mist, borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${((result.progress_step ?? 1) / 5) * 100}%`, height: "100%", background: C.ocean, borderRadius: 3, transition: "width .5s" }} />
                </div>
              </div>

              {[
                ["Claim ID",        result.id],
                ["Name",            result.name],
                ["Property",        result.property],
                ["Type",            result.claim_type],
                ["Incident Date",   result.incident_date],
                ["Estimated Loss",  result.estimated_loss ? `$${result.estimated_loss.toLocaleString()}` : "–"],
                ["Status",          result.status],
                ["Risk Level",      result.risk_level],
                ["Fraud Flag",      result.fraud_flag ? "⚠️ Yes" : "No"],
                ["Submitted",       result.submitted_at?.split("T")[0]],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: `1px solid ${C.border}`, fontSize: 14 }}>
                  <span style={{ color: C.subtle, fontWeight: 500 }}>{k}</span>
                  <span style={{ color: C.text, fontWeight: 600 }}>{v ?? "–"}</span>
                </div>
              ))}

              {result.uw_notes && (
                <div style={{ marginTop: 12, padding: 12, background: C.mist, borderRadius: 10, fontSize: 13, color: C.text }}>
                  <b>Underwriter Notes:</b> {result.uw_notes}
                </div>
              )}
            </div>
          )}
          {result?.error && <div style={{ marginTop: 12 }}><AlertBox type="error">{result.error}</AlertBox></div>}
        </Card>

        {/* sample recent claims */}
        <Card>
          <h3 style={{ fontWeight: 700, color: C.text, marginBottom: 16, fontFamily: "'Syne',sans-serif" }}>Sample References</h3>
          {[
            { ref: "CLM-884231", status: "Under Review",      date: "2025-12-01", level: "warning" },
            { ref: "CLM-771045", status: "Approved",           date: "2025-11-15", level: "success" },
            { ref: "CLM-663910", status: "Pending Documents",  date: "2025-10-29", level: "info" },
          ].map(c => (
            <div key={c.ref} onClick={() => setClaimId(c.ref)}
              style={{ border: `1px solid ${C.border}`, borderRadius: 12, padding: 18, marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", transition: "all .15s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.ocean; e.currentTarget.style.background = "#eff6ff"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = ""; }}
            >
              <div>
                <div style={{ fontWeight: 700, color: C.text, marginBottom: 4 }}>{c.ref}</div>
                <div style={{ fontSize: 13, color: C.subtle }}>{c.date}</div>
              </div>
              <AlertBox type={c.level} style={{ margin: 0, padding: "6px 14px" }}>{c.status}</AlertBox>
            </div>
          ))}
        </Card>
      </div>
    </Section>
  );
}

// ─── App Shell ────────────────────────────────────────────────────
const INSURER_MENU = [
  { id: "dashboard",  icon: "🏠", label: "Dashboard" },
  { id: "detection",  icon: "🛰",  label: "Pool Detection" },
  { id: "analytics",  icon: "📊", label: "Risk Analytics" },
  { id: "timeseries", icon: "⏳", label: "Time-Series" },
  { id: "iot",        icon: "📡", label: "IoT Inspection" },
  { id: "fraud",      icon: "📄", label: "Fraud Analysis" },
  { id: "assistant",  icon: "🤖", label: "AI Assistant" },
];

const HOLDER_MENU = [
  { id: "register", icon: "🏠", label: "Register Property" },
  { id: "claim",    icon: "📋", label: "Submit Claim" },
  { id: "track",    icon: "🔍", label: "Track Claim" },
];

export default function App() {
  const [role, setRole] = useState(null);
  const [view, setView] = useState("dashboard");

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=DM+Sans:wght@400;500;600&display=swap";
    document.head.appendChild(link);
    // FIX 3: ensure html/body fill the viewport with no overflow issues
    document.documentElement.style.cssText = "margin:0;padding:0;width:100%;height:100%;";
    document.body.style.cssText           = "margin:0;padding:0;width:100%;min-height:100vh;";
  }, []);

  const menu = role === "insurer" ? INSURER_MENU : HOLDER_MENU;

  const views = {
    dashboard:  <Dashboard />,
    detection:  <PoolDetection />,
    analytics:  <RiskAnalytics />,
    timeseries: <TimeSeries />,
    iot:        <IoTInspection />,
    fraud:      <FraudAnalysis />,
    assistant:  <AIAssistant />,
    register:   <PropertyRegister />,
    claim:      <SubmitClaim />,
    track:      <TrackClaim />,
  };

  const SIDEBAR_W = 240;

  // ── Landing ──────────────────────────────────────────────────────
  if (!role) return (
    <div style={{
      minHeight: "100vh", width: "100%",
      background: "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%)",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: "'DM Sans', sans-serif",
    }}>
      <div style={{ textAlign: "center", maxWidth: 620, width: "100%", padding: "0 24px" }}>
        <div style={{ fontSize: 64, marginBottom: 16 }}>🌊</div>
        <h1 style={{ fontSize: 48, fontWeight: 900, color: C.slate, fontFamily: "'Syne', sans-serif", letterSpacing: "-2px", marginBottom: 12 }}>AquaIntelligence</h1>
        <p style={{ fontSize: 18, color: C.subtle, marginBottom: 48, lineHeight: 1.6 }}>AI-Powered Swimming Pool Risk Intelligence Platform</p>
        <div style={{ display: "flex", gap: 20, justifyContent: "center" }}>
          {[
            { id: "insurer",      icon: "🏢", label: "Insurance Company", sub: "Risk analysis & detection tools" },
            { id: "policyholder", icon: "🏠", label: "Policyholder",       sub: "Manage property & claims" },
          ].map(r => (
            <div key={r.id}
              onClick={() => { setRole(r.id); setView(r.id === "insurer" ? "dashboard" : "register"); }}
              style={{ background: C.white, border: `2px solid ${C.border}`, borderRadius: 20, padding: "32px 36px", cursor: "pointer", flex: 1, transition: "all .2s", boxShadow: "0 2px 8px rgba(0,0,0,0.06)" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.ocean; e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 12px 32px rgba(3,105,161,.15)"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.06)"; }}
            >
              <div style={{ fontSize: 40, marginBottom: 12 }}>{r.icon}</div>
              <div style={{ fontWeight: 800, fontSize: 18, color: C.text, fontFamily: "'Syne',sans-serif", marginBottom: 6 }}>{r.label}</div>
              <div style={{ fontSize: 13, color: C.subtle }}>{r.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // ── Main App ─────────────────────────────────────────────────────
  return (
    <div style={{ display: "flex", minHeight: "100vh", width: "100%", fontFamily: "'DM Sans', sans-serif", background: C.mist, boxSizing: "border-box" }}>

      {/* Sidebar — fixed, always 240 px wide */}
      <aside style={{
        width: SIDEBAR_W, minWidth: SIDEBAR_W, flexShrink: 0,
        background: C.white, borderRight: `1px solid ${C.border}`,
        display: "flex", flexDirection: "column",
        position: "fixed", top: 0, bottom: 0, left: 0, zIndex: 10,
      }}>
        <div style={{ padding: "28px 24px 20px", borderBottom: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 22, marginBottom: 4 }}>🌊</div>
          <div style={{ fontWeight: 900, fontSize: 18, color: C.text, fontFamily: "'Syne',sans-serif", letterSpacing: "-0.5px" }}>AquaIntelligence</div>
          <div style={{ fontSize: 11, color: C.subtle, marginTop: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            {role === "insurer" ? "Insurance Company" : "Policyholder"}
          </div>
        </div>

        <nav style={{ flex: 1, padding: "16px 12px", overflowY: "auto" }}>
          {menu.map(item => (
            <div key={item.id} onClick={() => setView(item.id)}
              style={{
                display: "flex", alignItems: "center", gap: 12, padding: "11px 14px",
                borderRadius: 10, cursor: "pointer", marginBottom: 2, transition: "all .15s",
                background: view === item.id ? `${C.ocean}12` : "transparent",
                color:      view === item.id ? C.ocean : C.subtle,
                fontWeight: view === item.id ? 700 : 500,
                fontSize: 14,
              }}
              onMouseEnter={e => { if (view !== item.id) e.currentTarget.style.background = C.mist; }}
              onMouseLeave={e => { if (view !== item.id) e.currentTarget.style.background = "transparent"; }}
            >
              <span style={{ fontSize: 18 }}>{item.icon}</span>
              {item.label}
            </div>
          ))}
        </nav>

        <div style={{ padding: "16px 12px", borderTop: `1px solid ${C.border}` }}>
          <div
            onClick={() => { setRole(null); setView("dashboard"); }}
            style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 10, cursor: "pointer", color: C.subtle, fontSize: 13, fontWeight: 500 }}
            onMouseEnter={e => { e.currentTarget.style.background = C.mist; }}
            onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
          >
            ← Switch Role
          </div>
        </div>
      </aside>

      {/*
        FIX 4: Main content uses marginLeft = sidebar width and fills remaining space.
        We also cap max-width so content never looks too narrow on wide screens,
        and set width: 100% + box-sizing so padding doesn't cause overflow.
      */}
      <main style={{
        marginLeft: SIDEBAR_W,
        flex: 1,
        minWidth: 0,           // prevents flex child from overflowing
        width: `calc(100% - ${SIDEBAR_W}px)`,
        boxSizing: "border-box",
        padding: "40px 48px",
        minHeight: "100vh",
      }}>
        {views[view] || <Dashboard />}
      </main>
    </div>
  );
}