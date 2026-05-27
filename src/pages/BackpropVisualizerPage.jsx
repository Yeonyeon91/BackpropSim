import React, { useState, useEffect, useRef, useCallback } from 'react'

// ─── Math helpers ─────────────────────────────────────────────────────────────
const sigmoid = x => 1 / (1 + Math.exp(-x))
const sigmoidDeriv = x => { const s = sigmoid(x); return s * (1 - s) }
const relu = x => Math.max(0, x)
const reluDeriv = x => x > 0 ? 1 : 0
const tanh_ = x => Math.tanh(x)
const tanhDeriv = x => 1 - Math.tanh(x) ** 2

const ACTIVATIONS = {
  sigmoid: { fn: sigmoid, deriv: sigmoidDeriv, label: 'Sigmoid' },
  relu:    { fn: relu,    deriv: reluDeriv,    label: 'ReLU'    },
  tanh:    { fn: tanh_,   deriv: tanhDeriv,    label: 'Tanh'    },
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
const round4 = v => Math.round(v * 10000) / 10000
const round2 = v => Math.round(v * 100) / 100

// ─── Network architecture: 2 → 3 → 2 → 1 ────────────────────────────────────
// Fixed small network so everything is visible
const ARCH = [2, 3, 2, 1]

function initNetwork() {
  // weights[l][j][i] = weight from neuron i in layer l to neuron j in layer l+1
  const weights = []
  const biases  = []
  for (let l = 0; l < ARCH.length - 1; l++) {
    const W = []
    const B = []
    for (let j = 0; j < ARCH[l + 1]; j++) {
      const row = []
      for (let i = 0; i < ARCH[l]; i++) {
        row.push(round4((Math.random() - 0.5) * 2))
      }
      W.push(row)
      B.push(round4((Math.random() - 0.5) * 0.5))
    }
    weights.push(W)
    biases.push(B)
  }
  return { weights, biases }
}

function forwardPass(inputs, weights, biases, activationKey) {
  const act = ACTIVATIONS[activationKey]
  // zs[l] = pre-activation values at layer l+1
  // as[l] = post-activation values at layer l (as[0] = inputs)
  const as = [inputs.slice()]
  const zs = []

  for (let l = 0; l < weights.length; l++) {
    const z = []
    const a = []
    for (let j = 0; j < ARCH[l + 1]; j++) {
      let sum = biases[l][j]
      for (let i = 0; i < ARCH[l]; i++) {
        sum += weights[l][j][i] * as[l][i]
      }
      z.push(round4(sum))
      // Last layer: no activation (raw output)
      a.push(l === weights.length - 1 ? round4(sum) : round4(act.fn(sum)))
    }
    zs.push(z)
    as.push(a)
  }

  return { as, zs }
}

function backwardPass(as, zs, weights, target, activationKey) {
  const act = ACTIVATIONS[activationKey]
  const numLayers = weights.length

  // MSE loss: L = (pred - target)^2 / 2
  const pred = as[as.length - 1][0]
  const loss = round4(0.5 * (pred - target) ** 2)

  // dLoss/da for each layer (gradient of loss w.r.t. activations)
  const dAs = Array.from({ length: numLayers + 1 }, (_, i) => new Array(ARCH[i]).fill(0))
  dAs[numLayers][0] = round4(pred - target)  // dL/dpred

  // dW[l][j][i], dB[l][j]
  const dWeights = weights.map(W => W.map(row => row.map(() => 0)))
  const dBiases  = weights.map((W, l) => new Array(ARCH[l + 1]).fill(0))

  for (let l = numLayers - 1; l >= 0; l--) {
    for (let j = 0; j < ARCH[l + 1]; j++) {
      // dL/dz_j
      const dz = l === numLayers - 1
        ? dAs[l + 1][j]
        : round4(dAs[l + 1][j] * act.deriv(zs[l][j]))

      dBiases[l][j] = round4(dz)

      for (let i = 0; i < ARCH[l]; i++) {
        dWeights[l][j][i] = round4(dz * as[l][i])
        dAs[l][i] = round4(dAs[l][i] + dz * weights[l][j][i])
      }
    }
  }

  return { loss, dWeights, dBiases, dAs }
}

// ─── Canvas Neural Network Renderer ──────────────────────────────────────────
function NetworkCanvas({
  weights, as, dAs, dWeights,
  highlightLayer, highlightNeuron,
  phase, activeStep,
  activationKey,
  onNeuronClick,
}) {
  const canvasRef = useRef()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#f8f7ff'
    ctx.fillRect(0, 0, W, H)

    const layerX = ARCH.map((_, l) => 60 + (l / (ARCH.length - 1)) * (W - 120))
    const neuronY = ARCH.map((n, l) => {
      const spacing = Math.min(80, (H - 60) / (n + 1))
      return Array.from({ length: n }, (_, j) => H / 2 + (j - (n - 1) / 2) * spacing)
    })

    // ── Draw connections ──────────────────────────────────────────────────────
    for (let l = 0; l < ARCH.length - 1; l++) {
      for (let j = 0; j < ARCH[l + 1]; j++) {
        for (let i = 0; i < ARCH[l]; i++) {
          const x1 = layerX[l], y1 = neuronY[l][i]
          const x2 = layerX[l + 1], y2 = neuronY[l + 1][j]

          const w = weights[l][j][i]
          const dw = dWeights?.[l]?.[j]?.[i] ?? 0

          // Color by weight sign
          let alpha = clamp(Math.abs(w) * 0.6 + 0.1, 0.05, 0.8)
          let color = w >= 0 ? `rgba(124,111,247,${alpha})` : `rgba(220,38,38,${alpha})`
          let lineW = clamp(Math.abs(w) * 1.5, 0.5, 3)

          // Highlight gradient flow during backprop
          if (phase === 'backward' && dWeights) {
            const adw = Math.abs(dw)
            if (adw > 0.001) {
              alpha = clamp(adw * 3, 0.3, 1)
              color = dw > 0
                ? `rgba(217,119,6,${alpha})`
                : `rgba(5,150,105,${alpha})`
              lineW = clamp(adw * 5, 1, 4)
            }
          }

          // Animate active step
          const isActive =
            (phase === 'forward' && activeStep === l) ||
            (phase === 'backward' && activeStep === (ARCH.length - 2 - l))

          if (isActive) {
            ctx.shadowBlur = 8
            ctx.shadowColor = phase === 'forward' ? '#4ade80' : '#f59e0b'
          } else {
            ctx.shadowBlur = 0
          }

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = color
          ctx.lineWidth = lineW
          ctx.stroke()
          ctx.shadowBlur = 0

          // Weight label on hover (for highlighted connections)
          if (highlightLayer === l && (highlightNeuron === i || highlightNeuron === j)) {
            const mx = (x1 + x2) / 2
            const my = (y1 + y2) / 2
            ctx.fillStyle = '#d97706'
            ctx.font = '9px Courier New'
            ctx.fillText(w.toFixed(2), mx - 8, my - 4)
          }
        }
      }
    }

    // ── Draw neurons ──────────────────────────────────────────────────────────
    for (let l = 0; l < ARCH.length; l++) {
      for (let j = 0; j < ARCH[l]; j++) {
        const x = layerX[l]
        const y = neuronY[l][j]
        const activation = as?.[l]?.[j] ?? 0
        const gradient = dAs?.[l]?.[j] ?? 0

        const r = 18
        const isHighlighted = highlightLayer === l && highlightNeuron === j

        // Outer glow for active neurons
        if (Math.abs(activation) > 0.3 || isHighlighted) {
          ctx.beginPath()
          ctx.arc(x, y, r + 5, 0, Math.PI * 2)
          const glowAlpha = clamp(Math.abs(activation) * 0.3, 0, 0.4)
          ctx.fillStyle = phase === 'backward' && Math.abs(gradient) > 0.01
            ? `rgba(217,119,6,${glowAlpha})`
            : `rgba(124,111,247,${glowAlpha})`
          ctx.fill()
        }

        // Neuron fill based on activation value
        const intensity = clamp(Math.abs(activation), 0, 1)
        const isNeg = activation < 0
        const fillAlpha = 0.15 + intensity * 0.5

        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        if (phase === 'backward' && Math.abs(gradient) > 0.001) {
          ctx.fillStyle = gradient > 0
            ? `rgba(217,119,6,${fillAlpha + 0.2})`
            : `rgba(5,150,105,${fillAlpha + 0.2})`
        } else {
          ctx.fillStyle = isNeg
            ? `rgba(220,38,38,${fillAlpha})`
            : `rgba(124,111,247,${fillAlpha})`
        }
        ctx.fill()

        // Border
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.strokeStyle = isHighlighted ? '#1a1523' : (phase === 'backward' ? '#d97706' : '#7c6ff7')
        ctx.lineWidth = isHighlighted ? 2.5 : 1
        ctx.stroke()

        // Activation value text
        ctx.fillStyle = '#1a1523'
        ctx.font = `bold 10px Courier New`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(
          as ? activation.toFixed(2) : '?',
          x, y
        )

        // Gradient overlay during backprop
        if (phase === 'backward' && Math.abs(gradient) > 0.001) {
          ctx.fillStyle = gradient > 0 ? '#f59e0b' : '#22d3ee'
          ctx.font = '8px Courier New'
          ctx.fillText(`∇${gradient.toFixed(2)}`, x, y + 26)
        }
      }
    }

    // ── Layer labels ──────────────────────────────────────────────────────────
    const labels = ['INPUT', ...Array.from({ length: ARCH.length - 2 }, (_, i) => `HIDDEN ${i + 1}`), 'OUTPUT']
    labels.forEach((label, l) => {
      ctx.fillStyle = '#6b6580'
      ctx.font = '9px Courier New'
      ctx.textAlign = 'center'
      ctx.fillText(label, layerX[l], H - 12)
    })

    // ── Phase indicator arrow ─────────────────────────────────────────────────
    if (phase === 'forward' && activeStep >= 0) {
      const fromX = layerX[activeStep] + 22
      const toX   = layerX[activeStep + 1] - 22
      const midY  = 20
      ctx.beginPath()
      ctx.moveTo(fromX, midY)
      ctx.lineTo(toX - 8, midY)
      ctx.strokeStyle = '#7c6ff7'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.stroke()
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(toX - 8, midY - 5)
      ctx.lineTo(toX, midY)
      ctx.lineTo(toX - 8, midY + 5)
      ctx.fillStyle = '#7c6ff7'
      ctx.fill()
    }
    if (phase === 'backward' && activeStep >= 0) {
      const layerIdx = ARCH.length - 2 - activeStep
      const fromX = layerX[layerIdx + 1] - 22
      const toX   = layerX[layerIdx] + 22
      const midY  = 20
      ctx.beginPath()
      ctx.moveTo(fromX, midY)
      ctx.lineTo(toX + 8, midY)
      ctx.strokeStyle = '#d97706'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.stroke()
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(toX + 8, midY - 5)
      ctx.lineTo(toX, midY)
      ctx.lineTo(toX + 8, midY + 5)
      ctx.fillStyle = '#d97706'
      ctx.fill()
    }

  }, [weights, as, dAs, dWeights, highlightLayer, highlightNeuron, phase, activeStep, activationKey])

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={320}
      style={{ width: '100%', height: 'auto', cursor: 'crosshair' }}
    />
  )
}

// ─── Loss Curve mini-chart ────────────────────────────────────────────────────
function LossCurve({ history }) {
  const canvasRef = useRef()
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !history.length) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#f8f7ff'
    ctx.fillRect(0, 0, W, H)

    if (history.length < 2) return
    const maxLoss = Math.max(...history, 0.1)
    const pts = history.map((v, i) => ({
      x: 8 + (i / (history.length - 1)) * (W - 16),
      y: H - 8 - (v / maxLoss) * (H - 16),
    }))

    ctx.beginPath()
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
    ctx.strokeStyle = '#7c6ff7'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Fill under curve
    ctx.beginPath()
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
    ctx.lineTo(pts[pts.length - 1].x, H - 8)
    ctx.lineTo(pts[0].x, H - 8)
    ctx.fillStyle = 'rgba(124,111,247,0.07)'
    ctx.fill()

    // Current loss label
    const last = history[history.length - 1]
    ctx.fillStyle = '#7c6ff7'
    ctx.font = '9px Courier New'
    ctx.fillText(`loss: ${last.toFixed(4)}`, 8, 14)
  }, [history])
  return <canvas ref={canvasRef} width={260} height={80} style={{ width: '100%' }} />
}

// ─── Weight matrix display ────────────────────────────────────────────────────
function WeightMatrix({ weights, dWeights, layerIdx, label }) {
  const W = weights[layerIdx]
  const dW = dWeights?.[layerIdx]

  return (
    <div>
      <div style={{fontSize:10,color:"var(--text3)",marginBottom:4,letterSpacing:"0.5px"}}>{label}</div>
      <div className="overflow-x-auto">
        <table className="text-[11px] font-mono border-collapse">
          <tbody>
            {W.map((row, j) => (
              <tr key={j}>
                {row.map((w, i) => {
                  const dw = dW?.[j]?.[i] ?? 0
                  const isLarge = Math.abs(dw) > 0.05
                  return (
                    <td key={i} className={`px-1.5 py-0.5 border border-[var(--border)] text-center min-w-[52px] ${
                      isLarge
                        ? dw > 0 ? 'text-[#d97706] bg-[#fffbeb]' : 'text-[#059669] bg-[#eff6ff]'
                        : 'text-[var(--text2)]'
                    }`}>
                      {w.toFixed(3)}
                      {dW && (
                        <div className={`text-[9px] ${dw > 0 ? 'text-[#d97706]' : 'text-[#059669]'}`}>
                          {dw > 0 ? '↑' : '↓'}{Math.abs(dw).toFixed(3)}
                        </div>
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
  )
}

// ─── Activation function visualizer ──────────────────────────────────────────
function ActivationViz({ activationKey }) {
  const canvasRef = useRef()
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#f8f7ff'
    ctx.fillRect(0, 0, W, H)

    const act = ACTIVATIONS[activationKey]
    const range = 4
    const toX = v => (v / range + 1) / 2 * W
    const toY = v => (1 - (v + 1) / 2) * H

    // Axes
    ctx.strokeStyle = '#e8e4f0'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke()

    // Function curve
    ctx.beginPath()
    for (let px = 0; px <= W; px++) {
      const x = (px / W) * range * 2 - range
      const y = act.fn(x)
      const cy = (1 - (y + 1) / 2) * H
      px === 0 ? ctx.moveTo(px, cy) : ctx.lineTo(px, cy)
    }
    ctx.strokeStyle = '#7c6ff7'
    ctx.lineWidth = 2
    ctx.stroke()

    // Derivative curve
    ctx.beginPath()
    for (let px = 0; px <= W; px++) {
      const x = (px / W) * range * 2 - range
      const dy = act.deriv(x)
      const cy = (1 - (dy + 1) / 2) * H
      px === 0 ? ctx.moveTo(px, cy) : ctx.lineTo(px, cy)
    }
    ctx.strokeStyle = 'rgba(217,119,6,0.6)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([3, 3])
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = '#6b6580'
    ctx.font = '8px Courier New'
    ctx.fillText('f(x)', 4, 12)
    ctx.fillStyle = '#a89fc0'
    ctx.fillText("f'(x)", 4, 22)
  }, [activationKey])
  return <canvas ref={canvasRef} width={160} height={80} style={{ width: '100%' }} />
}

// ─── Main Page ────────────────────────────────────────────────────────────────
const SAMPLE_DATA = [
  { inputs: [0.8, 0.2], target: 0.9, label: 'Pattern A' },
  { inputs: [0.1, 0.9], target: 0.1, label: 'Pattern B' },
  { inputs: [0.6, 0.6], target: 0.7, label: 'Pattern C' },
  { inputs: [0.3, 0.7], target: 0.3, label: 'Pattern D' },
]

const PHASE_STEPS = {
  idle:     { label: 'IDLE',        color: "var(--text3)" },
  forward:  { label: 'FORWARD →',   color: "var(--accent)" },
  loss:     { label: 'LOSS ✕',       color: "#dc2626" },
  backward: { label: '← BACKWARD',  color: "#d97706" },
  update:   { label: 'UPDATE ↺',    color: "#059669" },
}

export default function BackpropVisualizerPage() {
  const [net, setNet] = useState(() => initNetwork())
  const [activationKey, setActivationKey] = useState('sigmoid')
  const [lr, setLr] = useState(0.5)
  const [sampleIdx, setSampleIdx] = useState(0)
  const [phase, setPhase] = useState('idle')
  const [activeStep, setActiveStep] = useState(-1)
  const [fwdResult, setFwdResult] = useState(null)   // { as, zs }
  const [bwdResult, setBwdResult] = useState(null)   // { loss, dWeights, dBiases, dAs }
  const [lossHistory, setLossHistory] = useState([])
  const [epoch, setEpoch] = useState(0)
  const [prevWeights, setPrevWeights] = useState(null)
  const [highlightLayer, setHighlightLayer] = useState(null)
  const [highlightNeuron, setHighlightNeuron] = useState(null)
  const [autoRunning, setAutoRunning] = useState(false)
  const [log, setLog] = useState([])
  const autoRef = useRef(false)

  const sample = SAMPLE_DATA[sampleIdx]

  // ── Step: Forward ──────────────────────────────────────────────────────────
  async function stepForward() {
    setPhase('forward')
    setBwdResult(null)
    const fwd = forwardPass(sample.inputs, net.weights, net.biases, activationKey)
    // Animate layer by layer
    for (let l = 0; l < ARCH.length - 1; l++) {
      setActiveStep(l)
      await sleep(420)
    }
    setFwdResult(fwd)
    setActiveStep(-1)
    setPhase('loss')
    addLog(`▶ Forward: pred=${fwd.as[fwd.as.length - 1][0].toFixed(4)}, target=${sample.target}`)
  }

  // ── Step: Backward ─────────────────────────────────────────────────────────
  async function stepBackward() {
    if (!fwdResult) return
    setPhase('backward')
    const bwd = backwardPass(fwdResult.as, fwdResult.zs, net.weights, sample.target, activationKey)
    for (let l = 0; l < ARCH.length - 1; l++) {
      setActiveStep(l)
      await sleep(420)
    }
    setBwdResult(bwd)
    setActiveStep(-1)
    setLossHistory(h => [...h.slice(-79), bwd.loss])
    addLog(`◀ Backward: loss=${bwd.loss.toFixed(4)}, maxGrad=${maxGrad(bwd.dWeights).toFixed(4)}`)
  }

  // ── Step: Update weights ───────────────────────────────────────────────────
  function stepUpdate() {
    if (!bwdResult) return
    setPrevWeights(JSON.parse(JSON.stringify(net.weights)))
    setNet(prev => {
      const newW = prev.weights.map((layer, l) =>
        layer.map((row, j) =>
          row.map((w, i) => round4(w - lr * bwdResult.dWeights[l][j][i]))
        )
      )
      const newB = prev.biases.map((layer, l) =>
        layer.map((b, j) => round4(b - lr * bwdResult.dBiases[l][j]))
      )
      return { weights: newW, biases: newB }
    })
    setEpoch(e => e + 1)
    setFwdResult(null)
    setBwdResult(null)
    setPhase('idle')
    addLog(`↺ Update: lr=${lr}, epoch=${epoch + 1}`)
  }

  // ── Auto run (train) ───────────────────────────────────────────────────────
  async function runOneEpoch(currentNet) {
    const fwd = forwardPass(sample.inputs, currentNet.weights, currentNet.biases, activationKey)
    const bwd = backwardPass(fwd.as, fwd.zs, currentNet.weights, sample.target, activationKey)
    const newWeights = currentNet.weights.map((layer, l) =>
      layer.map((row, j) =>
        row.map((w, i) => round4(w - lr * bwd.dWeights[l][j][i]))
      )
    )
    const newBiases = currentNet.biases.map((layer, l) =>
      layer.map((b, j) => round4(b - lr * bwd.dBiases[l][j]))
    )
    return { weights: newWeights, biases: newBiases, loss: bwd.loss }
  }

  async function startAutoTrain() {
    if (autoRunning) { autoRef.current = false; setAutoRunning(false); return }
    autoRef.current = true
    setAutoRunning(true)
    setPhase('forward')

    let currentNet = { ...net, weights: JSON.parse(JSON.stringify(net.weights)), biases: JSON.parse(JSON.stringify(net.biases)) }
    let ep = epoch

    while (autoRef.current) {
      const result = await runOneEpoch(currentNet)
      currentNet = { weights: result.weights, biases: result.biases }
      ep++
      setNet({ ...currentNet })
      setEpoch(ep)
      setLossHistory(h => [...h.slice(-79), result.loss])
      setFwdResult(forwardPass(sample.inputs, currentNet.weights, currentNet.biases, activationKey))
      if (result.loss < 0.0001 || ep > 2000) break
      await sleep(80)
    }
    autoRef.current = false
    setAutoRunning(false)
    setPhase('idle')
    addLog(`⬛ Auto-train stopped: epoch=${ep}, loss=${lossHistory[lossHistory.length - 1]?.toFixed(5) ?? '?'}`)
  }

  function resetAll() {
    autoRef.current = false
    setAutoRunning(false)
    setNet(initNetwork())
    setFwdResult(null)
    setBwdResult(null)
    setPhase('idle')
    setActiveStep(-1)
    setLossHistory([])
    setEpoch(0)
    setPrevWeights(null)
    setLog([])
  }

  function addLog(msg) {
    setLog(l => [`[ep${epoch}] ${msg}`, ...l.slice(0, 19)])
  }

  const phaseInfo = PHASE_STEPS[phase] || PHASE_STEPS.idle
  const pred = fwdResult ? fwdResult.as[fwdResult.as.length - 1][0] : null
  const currentLoss = bwdResult?.loss ?? null

  return (
    <div>
      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h2 style={{fontSize:22,fontWeight:700,color:'var(--text)',marginBottom:4}}>🧠 Backpropagation Visualizer</h2>
          <p style={{fontSize:13,color:'var(--text2)',maxWidth:540}}>
            역전파 알고리즘의 동작을 단계별로 직접 체험합니다. 순전파 → 오차 계산 → 역전파 → 가중치 업데이트의 전 과정을 눈으로 확인하세요.
          </p>
        </div>
        <div className="flex items-center gap-2 ml-4">
          <span className="text-[10px] tracking-[1px] text-[var(--text3)]">PHASE</span>
          <span className="text-[13px] font-bold tracking-[2px]" style={{ color: phaseInfo.color }}>
            {phaseInfo.label}
          </span>
        </div>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 260px' }}>

        {/* ── LEFT COLUMN ── */}
        <div className="space-y-4">

          {/* Network canvas */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] tracking-[2px] text-[var(--text3)]">NEURAL NETWORK  2→3→2→1</span>
              <div className="flex items-center gap-3 text-[10px] text-[var(--text3)]">
                <span><span className="text-[#7c6ff7]">━</span> positive weight</span>
                <span><span className="text-[#dc2626]">━</span> negative weight</span>
                <span><span className="text-[#d97706]">━</span> gradient flow</span>
              </div>
            </div>
            <NetworkCanvas
              weights={net.weights}
              as={fwdResult?.as}
              dAs={bwdResult?.dAs}
              dWeights={bwdResult?.dWeights}
              highlightLayer={highlightLayer}
              highlightNeuron={highlightNeuron}
              phase={phase}
              activeStep={activeStep}
              activationKey={activationKey}
            />
          </div>

          {/* Step buttons */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-3">STEP-BY-STEP CONTROL</div>

            <div className="grid grid-cols-5 gap-2 mb-3">
              {/* 1 Forward */}
              <button
                onClick={stepForward}
                disabled={phase === 'forward' || autoRunning}
                className="flex flex-col items-center gap-1 bg-[#ecfdf5] border border-[#86efac] text-[#7c6ff7] py-2.5 px-2 hover:bg-[#d1fae5] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <span className="text-[18px]">▶</span>
                <span className="text-[9px] tracking-[1px]">FORWARD</span>
              </button>

              {/* Arrow */}
              <div className="flex items-center justify-center text-[var(--text3)] text-[20px]">→</div>

              {/* 2 Backward */}
              <button
                onClick={stepBackward}
                disabled={!fwdResult || phase === 'backward' || autoRunning}
                className="flex flex-col items-center gap-1 bg-[#fffbeb] border border-[#fcd34d] text-[#d97706] py-2.5 px-2 hover:bg-[#fef3c7] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <span className="text-[18px]">◀</span>
                <span className="text-[9px] tracking-[1px]">BACKWARD</span>
              </button>

              {/* Arrow */}
              <div className="flex items-center justify-center text-[var(--text3)] text-[20px]">→</div>

              {/* 3 Update */}
              <button
                onClick={stepUpdate}
                disabled={!bwdResult || autoRunning}
                className="flex flex-col items-center gap-1 bg-[#eff6ff] border border-[#93c5fd] text-[#059669] py-2.5 px-2 hover:bg-[#dbeafe] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <span className="text-[18px]">↺</span>
                <span className="text-[9px] tracking-[1px]">UPDATE W</span>
              </button>
            </div>

            {/* Phase explanation */}
            <div className="grid grid-cols-3 gap-2 text-[10px]">
              <div className={`p-2 border ${phase === 'forward' || phase === 'loss' ? 'border-[#7c6ff7] bg-[#f5f3ff]' : 'border-[var(--border)]'}`}>
                <div className="text-[#7c6ff7] font-bold mb-1">① FORWARD</div>
                <div className="text-[var(--text2)] leading-relaxed">입력값이 레이어를 통과하며 예측값을 생성합니다. 각 뉴런은 weighted sum + activation을 계산합니다.</div>
              </div>
              <div className={`p-2 border ${phase === 'backward' ? 'border-[#fcd34d] bg-[#fffbeb]' : 'border-[var(--border)]'}`}>
                <div style={{color:"#d97706",fontWeight:600,marginBottom:4}}>② BACKWARD</div>
                <div className="text-[var(--text2)] leading-relaxed">오차가 뒤에서 앞으로 전파됩니다. chain rule로 각 weight가 loss에 얼마나 기여했는지 계산합니다.</div>
              </div>
              <div className={`p-2 border ${phase === 'update' ? 'border-[#93c5fd] bg-[#eff6ff]' : 'border-[var(--border)]'}`}>
                <div className="text-[#059669] font-bold mb-1">③ UPDATE</div>
                <div className="text-[var(--text2)] leading-relaxed">gradient 방향 반대로 weight를 수정합니다. learning rate가 수정 폭을 결정합니다.</div>
              </div>
            </div>
          </div>

          {/* Weight matrices */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-3">
              WEIGHT MATRICES
              {bwdResult && <span className="text-[#d97706] ml-2">← gradient 크기에 따라 색상 변화</span>}
            </div>
            <div className="grid grid-cols-3 gap-4">
              {net.weights.map((_, l) => (
                <WeightMatrix
                  key={l}
                  weights={net.weights}
                  dWeights={bwdResult?.dWeights}
                  layerIdx={l}
                  label={`W${l + 1}  (${ARCH[l + 1]}×${ARCH[l]})`}
                />
              ))}
            </div>
            {prevWeights && (
              <div className="mt-2 text-[10px] text-[#059669] border-l-2 border-[#059669] pl-2">
                ↺ weight updated — 위 행렬 값이 gradient × lr만큼 수정되었습니다
              </div>
            )}
          </div>

        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="space-y-3">

          {/* Sample & config */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-3">INPUT / CONFIG</div>

            {/* Sample selector */}
            <div className="text-[11px] text-[var(--text2)] mb-1">TRAINING SAMPLE</div>
            <select
              value={sampleIdx}
              onChange={e => { setSampleIdx(Number(e.target.value)); setFwdResult(null); setBwdResult(null); setPhase('idle') }}
              className="w-full bg-white text-[var(--text)] border border-[var(--border)] px-2 py-1.5 text-[11px] font-mono mb-3 focus:outline-none focus:border-[#4ade80]"
            >
              {SAMPLE_DATA.map((s, i) => (
                <option key={i} value={i}>{s.label}: [{s.inputs.join(', ')}] → {s.target}</option>
              ))}
            </select>

            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="bg-white border border-[var(--border)] p-2 text-center">
                <div className="text-[9px] text-[var(--text3)] mb-1">INPUT</div>
                <div className="text-[13px] font-bold text-[var(--text)]">[{sample.inputs.join(', ')}]</div>
              </div>
              <div className="bg-white border border-[var(--border)] p-2 text-center">
                <div className="text-[9px] text-[var(--text3)] mb-1">TARGET</div>
                <div className="text-[18px] font-bold text-[#7c6ff7]">{sample.target}</div>
              </div>
            </div>

            {/* Prediction result */}
            {pred !== null && (
              <div className="grid grid-cols-2 gap-2 mb-3">
                <div className="bg-white border border-[var(--border)] p-2 text-center">
                  <div className="text-[9px] text-[var(--text3)] mb-1">PRED</div>
                  <div className={`text-[18px] font-bold ${Math.abs(pred - sample.target) < 0.05 ? 'text-[#7c6ff7]' : 'text-[#dc2626]'}`}>
                    {pred.toFixed(4)}
                  </div>
                </div>
                <div className="bg-white border border-[var(--border)] p-2 text-center">
                  <div className="text-[9px] text-[var(--text3)] mb-1">LOSS</div>
                  <div className={`text-[18px] font-bold ${currentLoss !== null ? (currentLoss < 0.01 ? 'text-[#7c6ff7]' : 'text-[#dc2626]') : 'text-[var(--text2)]'}`}>
                    {currentLoss !== null ? currentLoss.toFixed(5) : '—'}
                  </div>
                </div>
              </div>
            )}

            {/* Activation */}
            <div className="text-[11px] text-[var(--text2)] mb-1">ACTIVATION FUNCTION</div>
            <div className="flex gap-1 mb-2">
              {Object.entries(ACTIVATIONS).map(([key, { label }]) => (
                <button
                  key={key}
                  onClick={() => setActivationKey(key)}
                  className={`flex-1 py-1 text-[10px] tracking-[1px] border transition-colors ${
                    activationKey === key
                      ? 'bg-[#ecfdf5] border-[#86efac] text-[#7c6ff7]'
                      : 'bg-transparent border-[var(--border)] text-[var(--text2)] hover:border-[#333]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <ActivationViz activationKey={activationKey} />
            <div className="flex gap-3 text-[9px] text-[var(--text3)] mt-1">
              <span><span className="text-[#7c6ff7]">—</span> f(x)</span>
              <span><span className="text-[#d97706]">- -</span> f'(x) derivative</span>
            </div>
          </div>

          {/* Learning rate */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-2">LEARNING RATE</div>
            <div className="text-[28px] font-bold text-[#d97706] mb-1">{lr.toFixed(2)}</div>
            <input
              type="range" min="0.01" max="2.0" step="0.01" value={lr}
              onChange={e => setLr(Number(e.target.value))}
              className="w-full mb-1"
            />
            <div className="flex justify-between text-[9px] text-[var(--text3)] mb-2">
              <span>0.01 (느림)</span><span>1.0</span><span>2.0 (발산)</span>
            </div>
            <div className={`text-[10px] border-l-2 pl-2 ${
              lr > 1.5 ? 'text-[#dc2626] border-[#f87171]' :
              lr > 0.8 ? 'text-[#d97706] border-[#f59e0b]' :
              lr < 0.05 ? 'text-[#059669] border-[#22d3ee]' :
              'text-[var(--text2)] border-[#333]'
            }`}>
              {lr > 1.5 ? '⚠ 너무 크면 loss가 발산할 수 있습니다' :
               lr > 0.8 ? '주의: 불안정할 수 있습니다' :
               lr < 0.05 ? '매우 느리게 수렴합니다' :
               '적절한 학습률입니다'}
            </div>
          </div>

          {/* Loss curve */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="flex justify-between items-center mb-2">
              <div className="text-[10px] tracking-[2px] text-[var(--text3)]">LOSS CURVE</div>
              <div className="text-[10px] text-[var(--text2)]">epoch {epoch}</div>
            </div>
            <LossCurve history={lossHistory} />
            {lossHistory.length === 0 && (
              <div className="text-[10px] text-[var(--text3)] text-center mt-1">학습 시작 후 표시됩니다</div>
            )}
          </div>

          {/* Auto train */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-2">AUTO TRAINING</div>
            <button
              onClick={startAutoTrain}
              className={`w-full py-2.5 font-bold text-[12px] tracking-[2px] transition-colors mb-2 border ${
                autoRunning
                  ? 'bg-[#fee2e2] border-[#fca5a5] text-[#dc2626] hover:bg-[#fecaca]'
                  : 'bg-[#ecfdf5] border-[#86efac] text-[#7c6ff7] hover:bg-[#d1fae5]'
              }`}
            >
              {autoRunning ? '⬛ STOP' : '⚡ AUTO TRAIN'}
            </button>
            <button
              onClick={resetAll}
              className="w-full py-2 bg-transparent border border-[var(--border)] text-[var(--text2)] text-[11px] tracking-[1px] hover:border-[#333] hover:text-[#777] transition-colors"
            >
              ↺ RESET NETWORK
            </button>
          </div>

          {/* Console log */}
          <div style={{background:"var(--bg2)",borderRadius:14,border:"1px solid var(--border)",padding:14}}>
            <div className="text-[10px] tracking-[2px] text-[var(--text3)] mb-2">CONSOLE</div>
            <div className="h-[100px] overflow-y-auto font-mono">
              {log.length === 0
                ? <div className="text-[10px] text-[var(--text3)]">_ waiting for input</div>
                : log.map((l, i) => (
                  <div key={i} className="text-[10px] text-[var(--text2)] leading-relaxed">{l}</div>
                ))
              }
            </div>
          </div>

        </div>
      </div>

      {/* Bottom: explanation panels */}
      <div className="grid grid-cols-3 gap-3 mt-4">
        <ConceptCard title="왜 역전파가 필요한가?" color="var(--accent)">
          <p>가중치를 <em>랜덤</em>으로 수정하면 평균 수백만 번의 시도가 필요합니다.</p>
          <p>역전파는 각 weight가 loss에 <strong>얼마나, 어느 방향으로</strong> 기여했는지를 수학적으로 계산하므로, 단 한 번의 계산으로 모든 weight의 수정 방향을 알 수 있습니다.</p>
        </ConceptCard>
        <ConceptCard title="Chain Rule이란?" color="#d97706">
          <p>역전파는 <em>연쇄 법칙(chain rule)</em>을 사용합니다.</p>
          <p>∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w</p>
          <p>출력층의 gradient가 hidden layer로 곱해지며 전파됩니다. 이 과정에서 activation 함수의 미분값이 핵심입니다.</p>
        </ConceptCard>
        <ConceptCard title="Gradient의 의미" color="#059669">
          <p><strong>양수 gradient</strong>: 이 weight를 키우면 loss도 커짐 → weight를 줄여야 함</p>
          <p><strong>음수 gradient</strong>: 이 weight를 키우면 loss가 줄어듦 → weight를 키워야 함</p>
          <p>그래서 <em>w = w - lr × gradient</em> 방향으로 업데이트합니다.</p>
        </ConceptCard>
      </div>
    </div>
  )
}

function ConceptCard({ title, color, children }) {
  return (
    <div className="bg-white border border-[var(--border)] p-4">
      <div className="text-[12px] font-bold mb-2" style={{ color }}>{title}</div>
      <div className="text-[11px] text-[var(--text2)] leading-relaxed space-y-1">
        {children}
      </div>
    </div>
  )
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

function maxGrad(dWeights) {
  let max = 0
  dWeights.forEach(layer => layer.forEach(row => row.forEach(v => { if (Math.abs(v) > max) max = Math.abs(v) })))
  return max
}
