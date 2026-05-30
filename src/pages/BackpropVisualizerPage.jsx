import React, { useState, useEffect, useRef, useCallback } from 'react'

// ─── Constants ────────────────────────────────────────────────────────────────
const LR_DEFAULT = 0.2

// Training samples from Excel: 10 samples (5 zeros, 5 ones)
// Each is a 4×3 = 12-pixel pattern (row-major: 4 rows × 3 cols)
const SAMPLES = [
  // "0" patterns (target: [1, 0])
  { pixels: [1,1,1, 1,0,1, 1,0,1, 1,1,1], target: [1,0], label:'0', id:'s1' },
  { pixels: [0,1,1, 1,0,1, 1,0,1, 1,1,1], target: [1,0], label:'0', id:'s2' },
  { pixels: [1,1,0, 1,0,1, 1,0,1, 1,1,1], target: [1,0], label:'0', id:'s3' },
  { pixels: [1,1,1, 1,0,1, 1,0,1, 1,1,0], target: [1,0], label:'0', id:'s4' },
  { pixels: [1,1,1, 1,0,1, 1,0,1, 0,1,1], target: [1,0], label:'0', id:'s5' },
  // "1" patterns (target: [0, 1])
  { pixels: [0,0,0, 0,1,1, 0,1,1, 0,1,1], target: [0,1], label:'1', id:'s6' },
  { pixels: [0,1,1, 1,1,1, 0,1,1, 0,1,1], target: [0,1], label:'1', id:'s7' },
  { pixels: [0,0,0, 1,1,0, 1,1,0, 1,1,0], target: [0,1], label:'1', id:'s8' },
  { pixels: [0,0,0, 1,1,1, 1,1,1, 1,1,0], target: [0,1], label:'1', id:'s9' },
  { pixels: [0,0,0, 1,1,1, 1,1,1, 0,1,1], target: [0,1], label:'1', id:'s10'},
]

const STEP_LABELS = [
  { id: 0, title: 'INPUT',       icon: '🖼️' },
  { id: 1, title: 'FORWARD',     icon: '→'  },
  { id: 2, title: 'OUTPUT',      icon: '📊' },
  { id: 3, title: 'LOSS',        icon: '⚡' },
  { id: 4, title: 'BACKPROP',    icon: '←'  },
  { id: 5, title: 'UPDATE W',    icon: '↺'  },
]

const STEP_DESCRIPTIONS = [
  '입력 이미지는 12개의 숫자(0 또는 1)로 펼쳐져 신경망에 들어갑니다. 각 픽셀이 하나의 입력 노드가 됩니다.',
  '각 입력 픽셀에 weight를 곱해 더한 뒤 sigmoid를 적용합니다. 은닉층은 픽셀들의 조합에서 특징을 추출합니다.',
  '은닉층의 출력이 다시 weight를 거쳐 출력층으로 전달됩니다. out₀는 숫자 0일 가능성, out₁는 숫자 1일 가능성입니다.',
  'Loss는 예측값이 정답과 얼마나 다른지를 나타냅니다. Loss가 클수록 예측이 틀린 것입니다.',
  'Backpropagation은 오차를 뒤로 보내어 어떤 weight를 얼마나 고칠지 계산하는 과정입니다. chain rule로 각 weight의 기여를 계산합니다.',
  'gradient 방향 반대로 weight를 수정합니다. 이 과정을 반복하면 점점 정답에 가까워집니다.',
]

const FORMULAS = [
  ['입력', 'x₁, x₂, ..., x₁₂ ∈ {0, 1}'],
  ['sigmoid', 'σ(z) = 1 / (1 + e⁻ᶻ)'],
  ['은닉층', 'z₂ᵢ = Σ wᵢⱼ·xⱼ + bᵢ\na₂ᵢ = σ(z₂ᵢ)'],
  ['출력층', 'z₃ₖ = Σ wₖᵢ·a₂ᵢ + bₖ\noutₖ = σ(z₃ₖ)'],
  ['Loss', 'C = ½ Σ (target − out)²'],
  ['δ₃ (출력)', 'δ₃ₖ = (targetₖ − outₖ) · outₖ · (1 − outₖ)'],
  ['δ₂ (은닉)', 'δ₂ᵢ = (Σ δ₃ₖ · wₖᵢ) · a₂ᵢ · (1 − a₂ᵢ)'],
  ['w 업데이트', 'w_new = w_old + η · δ · a_prev'],
]

// ─── Math ────────────────────────────────────────────────────────────────────
const sigmoid = x => 1 / (1 + Math.exp(-x))
const r4 = v => Math.round(v * 10000) / 10000
const r3 = v => Math.round(v * 1000) / 1000

function initWeights() {
  // w1: [3][12], b1: [3]
  // w2: [2][3],  b2: [2]
  const rand = () => r4((Math.random() - 0.5) * 2)
  const w1 = Array.from({ length: 3 }, () => Array.from({ length: 12 }, rand))
  const b1 = Array.from({ length: 3 }, rand)
  const w2 = Array.from({ length: 2 }, () => Array.from({ length: 3 }, rand))
  const b2 = Array.from({ length: 2 }, rand)
  return { w1, b1, w2, b2 }
}

function forward(inputs, w1, b1, w2, b2) {
  // Hidden layer
  const z2 = b1.map((b, i) => r4(b + w1[i].reduce((s, w, j) => s + w * inputs[j], 0)))
  const a2 = z2.map(z => r4(sigmoid(z)))
  // Output layer
  const z3 = b2.map((b, i) => r4(b + w2[i].reduce((s, w, j) => s + w * a2[j], 0)))
  const a3 = z3.map(z => r4(sigmoid(z)))
  return { z2, a2, z3, a3 }
}

function computeGrads(inputs, a2, a3, target, w2) {
  // δ₃
  const d3 = a3.map((o, k) => r4((target[k] - o) * o * (1 - o)))
  // δ₂
  const d2 = a2.map((a, i) => {
    const sum = d3.reduce((s, d, k) => s + d * w2[k][i], 0)
    return r4(sum * a * (1 - a))
  })
  // Loss
  const loss = r4(0.5 * a3.reduce((s, o, k) => s + (target[k] - o) ** 2, 0))
  return { d3, d2, loss }
}

function updateWeights({ w1, b1, w2, b2 }, inputs, a2, d2, d3, lr) {
  const nw2 = w2.map((row, k) => row.map((w, i) => r4(w + lr * d3[k] * a2[i])))
  const nb2 = b2.map((b, k) => r4(b + lr * d3[k]))
  const nw1 = w1.map((row, i) => row.map((w, j) => r4(w + lr * d2[i] * inputs[j])))
  const nb1 = b1.map((b, i) => r4(b + lr * d2[i]))
  return { w1: nw1, b1: nb1, w2: nw2, b2: nb2 }
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function PixelGrid({ pixels, highlight = false }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2, width: 72 }}>
      {pixels.map((v, i) => (
        <div key={i} style={{
          width: 22, height: 22, borderRadius: 3,
          background: v === 1 ? (highlight ? '#7c6ff7' : '#1a1523') : '#e8e4f0',
          border: '1px solid #c9c2e0',
          transition: 'background 0.3s',
        }} />
      ))}
    </div>
  )
}

function NodeCircle({ value, size = 38, color = '#7c6ff7', label, glow = false, style = {} }) {
  const alpha = Math.min(0.15 + Math.abs(value ?? 0) * 0.7, 0.95)
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      background: `rgba(${color === '#7c6ff7' ? '124,111,247' : color === '#d97706' ? '217,119,6' : '220,38,38'},${alpha})`,
      border: `2px solid ${glow ? '#d97706' : color}`,
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      boxShadow: glow ? `0 0 12px ${color}88` : 'none',
      transition: 'all 0.4s',
      ...style,
    }}>
      {value !== undefined && (
        <span style={{ fontSize: 9, fontFamily: "'DM Mono',monospace", color: '#1a1523', fontWeight: 700, lineHeight: 1 }}>
          {r3(value)}
        </span>
      )}
      {label && <span style={{ fontSize: 7, color: '#6b6580', marginTop: 1 }}>{label}</span>}
    </div>
  )
}

function FormulaBox({ step }) {
  const f = FORMULAS[Math.min(step, FORMULAS.length - 1)]
  return (
    <div style={{
      background: '#1a1523', borderRadius: 10, padding: '12px 16px',
      border: '1px solid #3a3050', minHeight: 80,
    }}>
      <div style={{ fontSize: 9, color: '#7c6ff7', letterSpacing: 2, marginBottom: 6 }}>FORMULA</div>
      <div style={{ fontSize: 10, color: '#aaa3c0', fontWeight: 600, marginBottom: 4 }}>{f[0]}</div>
      <pre style={{ fontSize: 11, color: '#e8e4f0', fontFamily: "'DM Mono',monospace", margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>
        {f[1]}
      </pre>
    </div>
  )
}

function StepBadge({ label, icon, active, done }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, cursor: 'default',
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: '50%',
        background: active ? '#7c6ff7' : done ? '#22c55e22' : '#2a2040',
        border: `2px solid ${active ? '#7c6ff7' : done ? '#22c55e' : '#3a3050'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 14, color: active ? '#fff' : done ? '#22c55e' : '#6b6580',
        boxShadow: active ? '0 0 16px #7c6ff755' : 'none',
        transition: 'all 0.3s',
      }}>
        {done && !active ? '✓' : icon}
      </div>
      <span style={{ fontSize: 8, letterSpacing: 1, color: active ? '#7c6ff7' : done ? '#22c55e' : '#6b6580', fontWeight: active ? 700 : 400 }}>
        {label}
      </span>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function BackpropVisualizerPage() {
  const [weights, setWeights] = useState(() => initWeights())
  const [sampleIdx, setSampleIdx] = useState(0)
  const [lr, setLr] = useState(LR_DEFAULT)
  const [step, setStep] = useState(0)            // 0–5
  const [fwd, setFwd] = useState(null)           // { z2, a2, z3, a3 }
  const [grads, setGrads] = useState(null)       // { d3, d2, loss }
  const [lossHistory, setLossHistory] = useState([])
  const [epoch, setEpoch] = useState(0)
  const [autoRunning, setAutoRunning] = useState(false)
  const [hoveredWeight, setHoveredWeight] = useState(null) // { layer, j, i, value }
  const autoRef = useRef(false)
  const lossCanvasRef = useRef()

  const sample = SAMPLES[sampleIdx]
  const inputs = sample.pixels
  const target = sample.target

  // Recompute forward when sample or weights change
  useEffect(() => {
    const f = forward(inputs, weights.w1, weights.b1, weights.w2, weights.b2)
    setFwd(f)
    setGrads(null)
    setStep(0)
  }, [sampleIdx, weights])

  // Draw loss curve
  useEffect(() => {
    const canvas = lossCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#1a1523'; ctx.fillRect(0, 0, W, H)
    if (lossHistory.length < 2) return
    const max = Math.max(...lossHistory, 0.01)
    const pts = lossHistory.slice(-200).map((v, i, arr) => ({
      x: 8 + (i / (arr.length - 1)) * (W - 16),
      y: H - 8 - (v / max) * (H - 16),
    }))
    ctx.beginPath()
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
    ctx.strokeStyle = '#7c6ff7'; ctx.lineWidth = 1.5; ctx.stroke()
    const last = lossHistory[lossHistory.length - 1]
    ctx.fillStyle = '#7c6ff7'; ctx.font = '9px Courier New'
    ctx.fillText(`loss: ${last.toFixed(5)}`, 8, 14)
    ctx.fillStyle = '#6b6580'; ctx.fillText(`epoch: ${epoch}`, 8, 26)
  }, [lossHistory, epoch])

  function nextStep() {
    if (step === 0) {
      // Already have fwd from useEffect; move to step 1
      setStep(1)
    } else if (step === 1) {
      setStep(2)
    } else if (step === 2) {
      setStep(3)
    } else if (step === 3) {
      const g = computeGrads(inputs, fwd.a2, fwd.a3, target, weights.w2)
      setGrads(g)
      setLossHistory(h => [...h.slice(-299), g.loss])
      setStep(4)
    } else if (step === 4) {
      setStep(5)
    } else if (step === 5) {
      // Apply update
      const g = grads || computeGrads(inputs, fwd.a2, fwd.a3, target, weights.w2)
      const nw = updateWeights(weights, inputs, fwd.a2, g.d2, g.d3, lr)
      setWeights(nw)
      setEpoch(e => e + 1)
      setStep(0)
    }
  }

  function trainOneEpoch() {
    let w = { ...weights }
    let totalLoss = 0
    for (const s of SAMPLES) {
      const f = forward(s.pixels, w.w1, w.b1, w.w2, w.b2)
      const g = computeGrads(s.pixels, f.a2, f.a3, s.target, w.w2)
      w = updateWeights(w, s.pixels, f.a2, g.d2, g.d3, lr)
      totalLoss += g.loss
    }
    setWeights(w)
    setEpoch(e => e + 1)
    setLossHistory(h => [...h.slice(-299), r4(totalLoss / SAMPLES.length)])
    setStep(0)
  }

  async function trainN(n) {
    autoRef.current = true
    setAutoRunning(true)
    let w = { ...weights }
    for (let ep = 0; ep < n; ep++) {
      if (!autoRef.current) break
      let totalLoss = 0
      for (const s of SAMPLES) {
        const f = forward(s.pixels, w.w1, w.b1, w.w2, w.b2)
        const g = computeGrads(s.pixels, f.a2, f.a3, s.target, w.w2)
        w = updateWeights(w, s.pixels, f.a2, g.d2, g.d3, lr)
        totalLoss += g.loss
      }
      if (ep % 10 === 0) {
        setWeights({ ...w })
        setEpoch(e => e + n > e ? e + 10 : e)
        setLossHistory(h => [...h.slice(-299), r4(totalLoss / SAMPLES.length)])
        await new Promise(r => setTimeout(r, 1))
      }
    }
    setWeights({ ...w })
    setEpoch(e => e + n)
    autoRef.current = false
    setAutoRunning(false)
  }

  function resetAll() {
    autoRef.current = false
    setAutoRunning(false)
    setWeights(initWeights())
    setFwd(null); setGrads(null); setStep(0)
    setLossHistory([]); setEpoch(0)
  }

  // Prediction
  const out = fwd?.a3 ?? [0.5, 0.5]
  const pred = out[0] > out[1] ? 0 : 1
  const isCorrect = pred === parseInt(sample.label)
  const totalLoss = grads?.loss ?? (fwd ? r4(0.5 * fwd.a3.reduce((s, o, k) => s + (target[k] - o) ** 2, 0)) : null)

  // Colors
  const c = { accent: '#7c6ff7', orange: '#d97706', red: '#dc2626', green: '#22c55e', bg: '#0e0c16', bg2: '#1a1523', bg3: '#22193a', border: '#2a2040', text: '#e8e4f0', text2: '#aaa3c0', text3: '#6b6580' }

  // Weight color helper
  const wColor = (v) => v > 0
    ? `rgba(100,120,255,${Math.min(0.2 + Math.abs(v) * 0.5, 0.9)})`
    : `rgba(220,60,60,${Math.min(0.2 + Math.abs(v) * 0.5, 0.9)})`

  return (
    <div style={{ background: c.bg, minHeight: '100vh', color: c.text, fontFamily: "'DM Mono', monospace", padding: '20px 16px' }}>

      {/* ── Header ── */}
      <div style={{ marginBottom: 20, borderBottom: `1px solid ${c.border}`, paddingBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 4 }}>
          <h2 style={{ fontSize: 18, fontWeight: 800, color: c.text, margin: 0, letterSpacing: -0.5 }}>
            🧠 Backpropagation Visualizer
          </h2>
          <span style={{ fontSize: 10, color: c.accent, letterSpacing: 2, fontWeight: 600 }}>
            EXCEL LECTURE 07 · 12→3→2
          </span>
        </div>
        <p style={{ fontSize: 11, color: c.text2, margin: 0, lineHeight: 1.6 }}>
          Excel 예제의 역전파 계산 과정을 웹에서 단계별로 재현합니다. 4×3 픽셀 입력 → 은닉층 3개 → 출력층 2개 → 숫자 0 또는 1 예측
        </p>
      </div>

      {/* ── Step progress bar ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 16, background: c.bg2, borderRadius: 12, padding: '10px 16px', border: `1px solid ${c.border}` }}>
        {STEP_LABELS.map((s, i) => (
          <React.Fragment key={s.id}>
            <StepBadge label={s.title} icon={s.icon} active={step === i} done={step > i} />
            {i < STEP_LABELS.length - 1 && (
              <div style={{ flex: 1, height: 2, background: step > i ? c.accent : c.border, margin: '0 4px', borderRadius: 2, transition: 'background 0.4s', marginBottom: 12 }} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* ── Main layout: Network | Controls ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 14 }}>

        {/* ── LEFT: Network visualization ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Step description */}
          <div style={{ background: c.bg3, borderRadius: 10, padding: '10px 14px', border: `1px solid ${c.border}`, borderLeft: `3px solid ${c.accent}` }}>
            <div style={{ fontSize: 9, color: c.accent, letterSpacing: 2, marginBottom: 4 }}>
              STEP {step + 1} · {STEP_LABELS[step]?.title}
            </div>
            <p style={{ fontSize: 11, color: c.text2, margin: 0, lineHeight: 1.7 }}>
              {STEP_DESCRIPTIONS[step]}
            </p>
          </div>

          {/* Network visualization area */}
          <div style={{ background: c.bg2, borderRadius: 14, border: `1px solid ${c.border}`, padding: 16, display: 'flex', gap: 20, alignItems: 'center', minHeight: 340 }}>

            {/* INPUT LAYER — pixel grid + 12 nodes */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 8, color: c.text3, letterSpacing: 2 }}>INPUT</div>
              <PixelGrid pixels={inputs} highlight={step >= 1} />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 3, marginTop: 4 }}>
                {inputs.map((v, i) => (
                  <div key={i} style={{
                    width: 28, height: 18, borderRadius: 4,
                    background: v === 1 ? (step >= 1 ? '#7c6ff722' : '#22193a') : '#0e0c16',
                    border: `1px solid ${v === 1 ? c.accent : c.border}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 9, color: v === 1 ? c.accent : c.text3,
                    fontWeight: 700, transition: 'all 0.4s',
                  }}>{v}</div>
                ))}
              </div>
              <div style={{ fontSize: 8, color: c.text3 }}>12 pixels</div>
            </div>

            {/* Arrow input → hidden */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <div style={{ fontSize: 9, color: step >= 1 ? c.accent : c.border }}>
                {step >= 1 ? '━━▶' : '───'}
              </div>
              {step >= 1 && (
                <div style={{ fontSize: 8, color: c.text3, textAlign: 'center', maxWidth: 50 }}>
                  w₁<br/>×x
                </div>
              )}
            </div>

            {/* HIDDEN LAYER — 3 nodes */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <div style={{ fontSize: 8, color: c.text3, letterSpacing: 2 }}>HIDDEN</div>
              {[0, 1, 2].map(i => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <NodeCircle
                    value={fwd ? fwd.a2[i] : undefined}
                    size={48}
                    color={step === 4 ? '#d97706' : '#7c6ff7'}
                    glow={step === 1 || step === 4}
                    label={`h${i+1}`}
                  />
                  {step >= 1 && fwd && (
                    <div style={{ fontSize: 8, color: c.text3, textAlign: 'center', lineHeight: 1.4 }}>
                      <div>z={r3(fwd.z2[i])}</div>
                      {step === 4 && grads && (
                        <div style={{ color: c.orange }}>δ={r3(grads.d2[i])}</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div style={{ fontSize: 8, color: c.text3 }}>3 nodes</div>
            </div>

            {/* Arrow hidden → output */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <div style={{ fontSize: 9, color: step >= 2 ? c.accent : c.border }}>
                {step >= 2 ? '━━▶' : '───'}
              </div>
              {step >= 2 && (
                <div style={{ fontSize: 8, color: c.text3, textAlign: 'center', maxWidth: 50 }}>
                  w₂<br/>×a₂
                </div>
              )}
            </div>

            {/* OUTPUT LAYER — 2 nodes */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <div style={{ fontSize: 8, color: c.text3, letterSpacing: 2 }}>OUTPUT</div>
              {[0, 1].map(k => (
                <div key={k} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <NodeCircle
                    value={fwd ? fwd.a3[k] : undefined}
                    size={52}
                    color={step === 3 || step === 4 ? (k === target.indexOf(Math.max(...target)) ? '#22c55e' : '#dc2626') : '#7c6ff7'}
                    glow={step === 2 || step === 3}
                    label={`out${k}`}
                  />
                  <div style={{ fontSize: 8, textAlign: 'center', lineHeight: 1.5 }}>
                    {step >= 2 && fwd && <div style={{ color: c.text3 }}>z={r3(fwd.z3[k])}</div>}
                    <div style={{ color: target[k] === 1 ? c.green : c.text3, fontWeight: 700 }}>
                      t={target[k]}
                    </div>
                    {step >= 3 && grads && (
                      <div style={{ color: c.orange }}>δ={r3(grads.d3[k])}</div>
                    )}
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 8, color: c.text3 }}>2 nodes</div>
            </div>

            {/* PREDICTION RESULT */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 8, color: c.text3, letterSpacing: 2 }}>RESULT</div>
              <div style={{
                width: 60, height: 60, borderRadius: '50%',
                background: step >= 2 ? (isCorrect ? '#22c55e22' : '#dc262622') : '#1a1523',
                border: `3px solid ${step >= 2 ? (isCorrect ? c.green : c.red) : c.border}`,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                transition: 'all 0.4s',
                boxShadow: step >= 2 ? `0 0 20px ${isCorrect ? c.green : c.red}44` : 'none',
              }}>
                <span style={{ fontSize: 22, fontWeight: 800, color: step >= 2 ? (isCorrect ? c.green : c.red) : c.text3 }}>
                  {step >= 2 ? pred : '?'}
                </span>
                <span style={{ fontSize: 8, color: step >= 2 ? (isCorrect ? c.green : c.red) : c.text3 }}>
                  {step >= 2 ? (isCorrect ? '✓ OK' : '✗ ERR') : ''}
                </span>
              </div>
              {step >= 3 && totalLoss !== null && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 8, color: c.text3 }}>LOSS</div>
                  <div style={{ fontSize: 14, fontWeight: 800, color: totalLoss > 0.1 ? c.red : c.green }}>
                    {totalLoss.toFixed(5)}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Weight matrices ── */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>

            {/* W1 matrix (3×12, shown compactly) */}
            <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 12 }}>
              <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 8 }}>
                W₁ HIDDEN WEIGHTS (3×12)
                {step === 5 && <span style={{ color: c.orange, marginLeft: 8 }}>← updating</span>}
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ borderCollapse: 'collapse', fontSize: 9, fontFamily: "'DM Mono',monospace" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '2px 4px', color: c.text3, textAlign: 'left' }}>h</th>
                      {Array.from({ length: 12 }, (_, j) => (
                        <th key={j} style={{ padding: '2px 3px', color: c.text3, fontSize: 8 }}>x{j+1}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {weights.w1.map((row, i) => (
                      <tr key={i}>
                        <td style={{ padding: '2px 4px', color: c.accent, fontWeight: 700 }}>h{i+1}</td>
                        {row.map((w, j) => {
                          const dw = grads && step === 5 ? r4(lr * grads.d2[i] * inputs[j]) : null
                          return (
                            <td key={j}
                              onMouseEnter={() => setHoveredWeight({ layer: 1, i, j, value: w, dw })}
                              onMouseLeave={() => setHoveredWeight(null)}
                              style={{
                                padding: '2px 3px', textAlign: 'center', cursor: 'default',
                                background: hoveredWeight?.layer === 1 && hoveredWeight.i === i && hoveredWeight.j === j ? '#3a3050' : wColor(w),
                                color: c.text, fontSize: 8, borderRadius: 2,
                                outline: step === 5 && dw ? `1px solid ${c.orange}` : 'none',
                                transition: 'background 0.2s',
                              }}>
                              {w.toFixed(2)}
                              {step === 5 && dw !== null && (
                                <div style={{ fontSize: 7, color: c.orange }}>{dw > 0 ? '+' : ''}{dw.toFixed(3)}</div>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* W2 matrix (2×3) */}
            <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 12 }}>
              <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 8 }}>
                W₂ OUTPUT WEIGHTS (2×3)
                {step === 5 && <span style={{ color: c.orange, marginLeft: 8 }}>← updating</span>}
              </div>
              <table style={{ borderCollapse: 'collapse', fontSize: 10, fontFamily: "'DM Mono',monospace" }}>
                <thead>
                  <tr>
                    <th style={{ padding: '3px 6px', color: c.text3 }}>out</th>
                    {['h1','h2','h3'].map(h => (
                      <th key={h} style={{ padding: '3px 8px', color: c.text3 }}>{h}</th>
                    ))}
                    <th style={{ padding: '3px 6px', color: c.text3 }}>bias</th>
                  </tr>
                </thead>
                <tbody>
                  {weights.w2.map((row, k) => {
                    return (
                      <tr key={k}>
                        <td style={{ padding: '3px 6px', color: c.accent, fontWeight: 700 }}>out{k}</td>
                        {row.map((w, i) => {
                          const dw = grads && step === 5 ? r4(lr * grads.d3[k] * fwd.a2[i]) : null
                          return (
                            <td key={i}
                              onMouseEnter={() => setHoveredWeight({ layer: 2, k, i, value: w, dw })}
                              onMouseLeave={() => setHoveredWeight(null)}
                              style={{
                                padding: '3px 8px', textAlign: 'center', cursor: 'default',
                                background: hoveredWeight?.layer === 2 && hoveredWeight.k === k && hoveredWeight.i === i ? '#3a3050' : wColor(w),
                                color: c.text, fontSize: 10, borderRadius: 3,
                                outline: step === 5 && dw ? `1px solid ${c.orange}` : 'none',
                                transition: 'background 0.2s',
                              }}>
                              {w.toFixed(3)}
                              {step === 5 && dw !== null && (
                                <div style={{ fontSize: 8, color: c.orange }}>{dw > 0 ? '+' : ''}{dw.toFixed(4)}</div>
                              )}
                            </td>
                          )
                        })}
                        <td style={{ padding: '3px 8px', color: c.text2, fontSize: 10 }}>
                          {weights.b2[k].toFixed(3)}
                          {step === 5 && grads && (
                            <div style={{ fontSize: 8, color: c.orange }}>
                              {(r4(lr * grads.d3[k]) > 0 ? '+' : '')}{r4(lr * grads.d3[k]).toFixed(4)}
                            </div>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>

              {/* Hover tooltip */}
              {hoveredWeight && (
                <div style={{ marginTop: 8, background: '#3a3050', borderRadius: 6, padding: '6px 10px', fontSize: 10, color: c.text2 }}>
                  w = {hoveredWeight.value.toFixed(4)}
                  {hoveredWeight.dw !== null && <span style={{ color: c.orange }}> → Δ={hoveredWeight.dw.toFixed(4)}</span>}
                </div>
              )}

              {/* Delta display in backprop step */}
              {step >= 3 && grads && (
                <div style={{ marginTop: 10, borderTop: `1px solid ${c.border}`, paddingTop: 8 }}>
                  <div style={{ fontSize: 9, color: c.orange, letterSpacing: 1, marginBottom: 4 }}>GRADIENTS</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    {grads.d3.map((d, k) => (
                      <div key={k} style={{ background: '#d9770611', borderRadius: 4, padding: '4px 8px', fontSize: 10 }}>
                        <span style={{ color: c.text3 }}>δ₃[{k}] =</span>
                        <span style={{ color: c.orange, fontWeight: 700 }}> {d.toFixed(4)}</span>
                      </div>
                    ))}
                    {grads.d2.map((d, i) => (
                      <div key={i} style={{ background: '#7c6ff711', borderRadius: 4, padding: '4px 8px', fontSize: 10 }}>
                        <span style={{ color: c.text3 }}>δ₂[{i}] =</span>
                        <span style={{ color: c.accent, fontWeight: 700 }}> {d.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Loss curve */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 12 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 8 }}>LOSS CURVE</div>
            <canvas ref={lossCanvasRef} width={640} height={70} style={{ width: '100%', borderRadius: 6 }} />
            {lossHistory.length === 0 && (
              <div style={{ fontSize: 10, color: c.text3, textAlign: 'center', marginTop: 4 }}>
                학습을 시작하면 Loss 그래프가 표시됩니다
              </div>
            )}
          </div>
        </div>

        {/* ── RIGHT: Controls ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Sample selector */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 14 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 10 }}>TRAINING SAMPLE</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, marginBottom: 10 }}>
              {SAMPLES.map((s, i) => (
                <button key={i} onClick={() => { setSampleIdx(i); setStep(0) }}
                  style={{
                    border: `2px solid ${sampleIdx === i ? c.accent : c.border}`,
                    background: sampleIdx === i ? '#7c6ff722' : c.bg3,
                    borderRadius: 8, padding: '6px 4px', cursor: 'pointer',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                    transition: 'all 0.2s',
                  }}>
                  <PixelGrid pixels={s.pixels} highlight={sampleIdx === i} />
                  <span style={{ fontSize: 9, color: sampleIdx === i ? c.accent : c.text3, fontWeight: 700 }}>
                    "{s.label}"
                  </span>
                </button>
              ))}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div style={{ background: c.bg3, borderRadius: 8, padding: '8px 10px', textAlign: 'center' }}>
                <div style={{ fontSize: 8, color: c.text3, marginBottom: 2 }}>TARGET</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: c.accent }}>
                  [{target.join(', ')}]
                </div>
                <div style={{ fontSize: 9, color: c.text3 }}>숫자 "{sample.label}"</div>
              </div>
              <div style={{ background: c.bg3, borderRadius: 8, padding: '8px 10px', textAlign: 'center', border: `1px solid ${isCorrect ? c.green : c.red}` }}>
                <div style={{ fontSize: 8, color: c.text3, marginBottom: 2 }}>PREDICTION</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: isCorrect ? c.green : c.red }}>
                  [{out.map(v => r3(v)).join(', ')}]
                </div>
                <div style={{ fontSize: 9, color: isCorrect ? c.green : c.red }}>
                  {isCorrect ? '✓ Correct' : '✗ Wrong'}
                </div>
              </div>
            </div>
          </div>

          {/* Step-by-step control */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 14 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 10 }}>STEP CONTROL</div>
            <button onClick={nextStep} disabled={autoRunning}
              style={{
                width: '100%', padding: '12px', borderRadius: 8, cursor: autoRunning ? 'not-allowed' : 'pointer',
                background: '#7c6ff7', border: 'none', color: '#fff', fontSize: 13, fontWeight: 700,
                letterSpacing: 1, marginBottom: 8, opacity: autoRunning ? 0.5 : 1,
                boxShadow: '0 2px 12px #7c6ff744', transition: 'all 0.2s',
              }}>
              {step === 5 ? '↺ APPLY & NEXT SAMPLE' : `▶ NEXT STEP → ${STEP_LABELS[step + 1]?.title ?? ''}`}
            </button>
            <div style={{ fontSize: 9, color: c.text3, background: c.bg3, borderRadius: 6, padding: '6px 10px', lineHeight: 1.7 }}>
              현재: <span style={{ color: c.accent }}>{STEP_LABELS[step]?.title}</span>
              {step < 5 && <> → 다음: <span style={{ color: c.text }}>{STEP_LABELS[step + 1]?.title}</span></>}
            </div>
          </div>

          {/* Auto train */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 14 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 10 }}>AUTO TRAIN</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button onClick={trainOneEpoch} disabled={autoRunning}
                style={{ padding: '9px', borderRadius: 7, cursor: autoRunning ? 'not-allowed' : 'pointer', background: '#22193a', border: `1px solid ${c.accent}`, color: c.accent, fontSize: 11, fontWeight: 600, opacity: autoRunning ? 0.5 : 1, transition: 'all 0.2s' }}>
                ⚡ Train 1 Epoch (10 samples)
              </button>
              <button onClick={() => trainN(100)} disabled={autoRunning}
                style={{ padding: '9px', borderRadius: 7, cursor: autoRunning ? 'not-allowed' : 'pointer', background: '#22193a', border: `1px solid ${c.green}`, color: c.green, fontSize: 11, fontWeight: 600, opacity: autoRunning ? 0.5 : 1, transition: 'all 0.2s' }}>
                🚀 Train 100 Epochs
              </button>
              <button onClick={() => trainN(1000)} disabled={autoRunning}
                style={{ padding: '9px', borderRadius: 7, cursor: autoRunning ? 'not-allowed' : 'pointer', background: '#22193a', border: `1px solid ${c.orange}`, color: c.orange, fontSize: 11, fontWeight: 600, opacity: autoRunning ? 0.5 : 1, transition: 'all 0.2s' }}>
                ⚡ Train 1000 Epochs
              </button>
              {autoRunning && (
                <button onClick={() => { autoRef.current = false; setAutoRunning(false) }}
                  style={{ padding: '9px', borderRadius: 7, cursor: 'pointer', background: '#dc262622', border: `1px solid ${c.red}`, color: c.red, fontSize: 11, fontWeight: 600 }}>
                  ⬛ STOP
                </button>
              )}
              <button onClick={resetAll}
                style={{ padding: '8px', borderRadius: 7, cursor: 'pointer', background: 'transparent', border: `1px solid ${c.border}`, color: c.text2, fontSize: 10 }}>
                ↺ Reset Weights
              </button>
            </div>
            {autoRunning && (
              <div style={{ marginTop: 8, fontSize: 10, color: c.orange, textAlign: 'center' }}>
                학습 중... epoch {epoch}
              </div>
            )}
          </div>

          {/* Learning rate */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 14 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 6 }}>LEARNING RATE η</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: c.orange, marginBottom: 6 }}>{lr.toFixed(2)}</div>
            <input type="range" min="0.01" max="1.0" step="0.01" value={lr}
              onChange={e => setLr(Number(e.target.value))}
              style={{ width: '100%', marginBottom: 6 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: c.text3, marginBottom: 6 }}>
              <span>0.01</span><span>0.2 (Excel)</span><span>1.0</span>
            </div>
            <div style={{
              fontSize: 10, padding: '5px 8px', borderRadius: 5,
              background: lr > 0.5 ? '#dc262611' : lr < 0.05 ? '#22c55e11' : '#7c6ff711',
              color: lr > 0.5 ? c.red : lr < 0.05 ? c.green : c.text2,
              borderLeft: `2px solid ${lr > 0.5 ? c.red : lr < 0.05 ? c.green : c.accent}`,
            }}>
              {lr > 0.5 ? '⚠ 크면 발산할 수 있습니다' : lr < 0.05 ? '매우 느리게 수렴합니다' : `Excel 예제 기본값: 0.2`}
            </div>
          </div>

          {/* Formula box */}
          <FormulaBox step={Math.min(
            step === 0 ? 0 : step === 1 ? 1 : step === 2 ? 2 : step === 2 ? 3 : step === 3 ? 4 : step === 4 ? 5 : 7,
            FORMULAS.length - 1
          )} />

          {/* Stats */}
          <div style={{ background: c.bg2, borderRadius: 12, border: `1px solid ${c.border}`, padding: 14 }}>
            <div style={{ fontSize: 9, color: c.text3, letterSpacing: 2, marginBottom: 8 }}>STATISTICS</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              <div style={{ background: c.bg3, borderRadius: 6, padding: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: 8, color: c.text3 }}>EPOCH</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: c.accent }}>{epoch}</div>
              </div>
              <div style={{ background: c.bg3, borderRadius: 6, padding: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: 8, color: c.text3 }}>LOSS</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: totalLoss !== null && totalLoss < 0.05 ? c.green : c.red }}>
                  {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '—'}
                </div>
              </div>
            </div>

            {/* Accuracy check across all samples */}
            {epoch > 0 && (() => {
              const correct = SAMPLES.filter(s => {
                const f = forward(s.pixels, weights.w1, weights.b1, weights.w2, weights.b2)
                const p = f.a3[0] > f.a3[1] ? 0 : 1
                return p === parseInt(s.label)
              }).length
              return (
                <div style={{ marginTop: 8, background: correct === SAMPLES.length ? '#22c55e11' : '#dc262611', borderRadius: 6, padding: '6px 10px', fontSize: 10, textAlign: 'center', color: correct === SAMPLES.length ? c.green : c.text2 }}>
                  정확도: {correct}/{SAMPLES.length} 샘플 정답
                  {correct === SAMPLES.length && <span style={{ color: c.green }}> 🎉 완벽!</span>}
                </div>
              )
            })()}
          </div>
        </div>
      </div>

      {/* ── Bottom: concept cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginTop: 14 }}>
        {[
          { title: '왜 역전파가 필요한가?', color: c.accent, body: 'weight를 무작위로 수정하면 수백만 번의 시도가 필요합니다. 역전파는 각 weight가 loss에 얼마나 기여했는지를 chain rule로 정확히 계산합니다.' },
          { title: 'sigmoid의 역할', color: c.orange, body: 'σ(z) = 1/(1+e⁻ᶻ) — 임의의 실수를 0~1 범위로 압축합니다. 미분이 σ(1-σ)로 간단해서 역전파 계산이 쉬워집니다.' },
          { title: 'weight 업데이트 원칙', color: c.green, body: 'w_new = w_old + η · δ · a_prev\ngradient 반대 방향으로 weight를 수정합니다. learning rate η가 수정 폭을 결정합니다.' },
        ].map(({ title, color, body }) => (
          <div key={title} style={{ background: c.bg2, borderRadius: 10, padding: '12px 14px', border: `1px solid ${c.border}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 6 }}>{title}</div>
            <pre style={{ fontSize: 10, color: c.text2, lineHeight: 1.7, margin: 0, whiteSpace: 'pre-wrap', fontFamily: "'DM Mono',monospace" }}>{body}</pre>
          </div>
        ))}
      </div>
    </div>
  )
}
