import React, { useState } from 'react'
import ControlPanel from '../components/ControlPanel.jsx'
import ImageResultCard from '../components/ImageResultCard.jsx'
import GradientHeatmapCard from '../components/GradientHeatmapCard.jsx'
import PerturbationCard from '../components/PerturbationCard.jsx'
import PredictionSummary from '../components/PredictionSummary.jsx'
import AttackTimeline from '../components/AttackTimeline.jsx'
import { runAttack } from '../api/attackApi.js'
import { SAMPLES, ORIGINAL_PROBS } from '../data/mockAttackData.js'

const TIMELINE_STEPS = 6
const STEP_DELAY_MS = 500

export default function AttackLabPage({ onExperiment }) {
  const [sampleId, setSampleId] = useState('sample-1')
  const [epsilon, setEpsilon] = useState(0.10)
  const [targetLabel, setTargetLabel] = useState('auto')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [tlStep, setTlStep] = useState(-1)

  const sample = SAMPLES[sampleId]
  const origConf = ORIGINAL_PROBS[sampleId][String(sample.trueLabel)]

  async function handleAttack() {
    setLoading(true); setResult(null); setTlStep(-1)
    for (let i = 0; i < TIMELINE_STEPS; i++) {
      setTlStep(i)
      await new Promise(r => setTimeout(r, STEP_DELAY_MS))
    }
    try {
      const res = await runAttack({ sampleId, epsilon: parseFloat(epsilon), targetLabel })
      setResult(res); setTlStep(TIMELINE_STEPS)
      if (onExperiment) onExperiment(res)
    } catch (err) {
      console.error(err); setTlStep(-1)
    } finally { setLoading(false) }
  }

  function handleReset() { setResult(null); setTlStep(-1) }

  return (
    <div>
      {/* Page header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>⚗️ Attack Lab</h2>
        <p style={{ fontSize: 13, color: 'var(--text2)' }}>
          MNIST 숫자 이미지를 선택하고 epsilon을 조절한 뒤, 역전파 기반 공격을 실행합니다.
        </p>
      </div>

      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: '220px 1fr 260px' }}>
        {/* Control Panel */}
        <ControlPanel
          sampleId={sampleId} setSampleId={setSampleId}
          epsilon={epsilon} setEpsilon={setEpsilon}
          targetLabel={targetLabel} setTargetLabel={setTargetLabel}
          onAttack={handleAttack} onReset={handleReset} loading={loading}
        />

        {/* Center */}
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            <ImageResultCard stepNum="01 ·" title="ORIGINAL IMAGE" digit={sample.trueLabel} type="original" prediction={sample.trueLabel} confidence={origConf} />
            <GradientHeatmapCard digit={result ? result.trueLabel : undefined} gradientMin={result?.attack.gradientMin} gradientMax={result?.attack.gradientMax} topPixels={result?.attack.topInfluentialPixels} />
            <PerturbationCard digit={result ? result.trueLabel : undefined} epsilon={result?.attack.epsilon ?? epsilon} />
            <ImageResultCard stepNum="04 ·" title="ADVERSARIAL IMAGE" digit={result ? result.adversarial.prediction : undefined} type="adversarial" prediction={result?.adversarial.prediction} confidence={result?.adversarial.confidence} />
          </div>
          <AttackTimeline activeStep={tlStep} />
        </div>

        {/* Summary */}
        <PredictionSummary result={result} />
      </div>
    </div>
  )
}
