import React, { useRef, useEffect } from 'react'
import ProcessStepCard from '../components/ProcessStepCard.jsx'
import ProbabilityBarChart from '../components/ProbabilityBarChart.jsx'
import { drawDigit } from '../components/ImageResultCard.jsx'
import { ORIGINAL_PROBS, ADVERSARIAL_PROBS, ADVERSARIAL_PREDS } from '../data/mockAttackData.js'

const SAMPLE_ID = 'sample-1'
const TRUE_LABEL = 7
const ADV_PRED = ADVERSARIAL_PREDS[SAMPLE_ID]
const ORIG_PROBS = ORIGINAL_PROBS[SAMPLE_ID]
const ADV_PROBS = ADVERSARIAL_PROBS[SAMPLE_ID]

function StatCard({ label, value, color = 'var(--accent)', bg = 'var(--tag-bg)' }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: '10px 12px', textAlign: 'center', border: '1px solid var(--border)' }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

function Formula({ children }) {
  return (
    <div style={{
      display: 'inline-block', padding: '8px 16px', borderRadius: 8, margin: '8px 0',
      background: 'var(--tag-bg)', color: 'var(--accent)',
      fontSize: 15, fontFamily: "'DM Mono', monospace", fontWeight: 500,
      border: '1px solid var(--border2)',
    }}>{children}</div>
  )
}

function Note({ children }) {
  return (
    <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.7, borderLeft: '3px solid var(--accent2)', paddingLeft: 10, marginTop: 10 }}>
      {children}
    </p>
  )
}

export default function ProcessExplorerPage() {
  const origRef = useRef(); const advRef = useRef()
  const origRef2 = useRef()
  useEffect(() => {
    if (origRef.current) drawDigit(origRef.current, TRUE_LABEL, 'original')
    if (origRef2.current) drawDigit(origRef2.current, TRUE_LABEL, 'original')
    if (advRef.current) drawDigit(advRef.current, ADV_PRED, 'adversarial')
  }, [])

  const canvasStyle = { width: 90, height: 90, borderRadius: 10 }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>📖 Process Explorer</h2>
        <p style={{ fontSize: 13, color: 'var(--text2)', maxWidth: 600 }}>
          공격 과정은 단순히 이미지를 바꾸는 것이 아니라, 역전파로 계산된 gradient를 입력에 적용하는 과정입니다.
        </p>
      </div>

      <ProcessStepCard stepNum="STEP 01" title="Original Input">
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
          <div style={{ background: '#f8f7ff', borderRadius: 12, border: '1.5px solid #c7d2fe', padding: 4 }}>
            <canvas ref={origRef} width={56} height={56} style={canvasStyle} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
              <StatCard label="TRUE LABEL" value={TRUE_LABEL} color="var(--text)" bg="var(--bg2)" />
              <StatCard label="PRED" value={TRUE_LABEL} color="var(--text)" bg="var(--bg2)" />
              <StatCard label="CONFIDENCE" value={`${(ORIG_PROBS[String(TRUE_LABEL)] * 100).toFixed(1)}%`} color="var(--accent)" />
            </div>
            <Note>원본 이미지가 모델에 입력됩니다. 모델은 숫자 {TRUE_LABEL}임을 98.4%의 확률로 정확히 예측합니다.</Note>
          </div>
        </div>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 02" title="Forward Propagation">
        <Note>모델이 입력 이미지를 받아 각 클래스(0~9)의 예측 확률을 계산합니다.</Note>
        <div style={{ marginTop: 12, maxWidth: 400 }}>
          <ProbabilityBarChart probs={ORIG_PROBS} highlight={TRUE_LABEL} label="OUTPUT PROBABILITIES (BEFORE ATTACK)" />
        </div>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 03" title="Loss Calculation">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, maxWidth: 320, marginBottom: 4 }}>
          <StatCard label="LOSS BEFORE" value="0.018" color="var(--text)" bg="var(--bg2)" />
          <StatCard label="LOSS AFTER" value="2.414" color="#dc2626" bg="#fff1f2" />
          <StatCard label="DELTA" value="+2.396" color="#dc2626" bg="#fff1f2" />
        </div>
        <Note>공격에서는 모델이 기존 정답에서 멀어지도록 loss 방향을 사용합니다. 일반 학습과 반대 방향입니다.</Note>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 04" title="Backpropagation">
        <Formula>∂Loss / ∂Input</Formula>
        <Note>여기서는 weight가 아니라 input에 대한 gradient를 구합니다. 모델의 가중치는 고정된 채로 진행됩니다.</Note>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, maxWidth: 340, marginTop: 10 }}>
          <div style={{ background: 'var(--bg2)', borderRadius: 8, padding: '10px 12px', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', marginBottom: 4 }}>일반 학습</div>
            <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: "'DM Mono', monospace" }}>∂Loss / <span style={{ color: 'var(--accent)' }}>∂W</span></div>
          </div>
          <div style={{ background: '#fffbeb', borderRadius: 8, padding: '10px 12px', border: '1px solid #fed7aa' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#d97706', marginBottom: 4 }}>적대적 공격 ★</div>
            <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: "'DM Mono', monospace" }}>∂Loss / <span style={{ color: '#d97706' }}>∂X</span></div>
          </div>
        </div>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 05" title="Input Gradient Heatmap">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, maxWidth: 280, marginBottom: 8 }}>
          <StatCard label="GRAD MIN" value="-0.032" color="#7c6ff7" bg="var(--tag-bg)" />
          <StatCard label="GRAD MAX" value="+0.041" color="#059669" bg="#ecfdf5" />
          <StatCard label="TOP PIXELS" value="84" color="#d97706" bg="#fffbeb" />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <span style={{ fontSize: 10, color: 'var(--text3)' }}>LOW</span>
          <div style={{ width: 120, height: 8, borderRadius: 99, background: 'linear-gradient(to right, #c7d2fe, #7c6ff7, #f59e0b)' }} />
          <span style={{ fontSize: 10, color: 'var(--text3)' }}>HIGH</span>
        </div>
        <Note>양수 gradient 픽셀: 밝게 하면 loss 증가 → 음수 gradient: 어둡게 하면 loss 증가.</Note>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 06" title="Perturbation Generation">
        <Formula>x_adv = x + ε · sign(∇x Loss)</Formula>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, maxWidth: 280, marginTop: 10 }}>
          <StatCard label="EPSILON" value="0.10" color="#d97706" bg="#fffbeb" />
          <StatCard label="PIXEL RANGE" value="[0,255]" color="var(--text)" bg="var(--bg2)" />
          <StatCard label="METHOD" value="FGSM" color="var(--accent)" bg="var(--tag-bg)" />
        </div>
        <Note>epsilon은 gradient 방향을 원본 이미지에 얼마나 강하게 반영할지 결정합니다. FGSM은 gradient의 부호(방향)만 사용합니다.</Note>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 07" title="Adversarial Image">
        <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', marginBottom: 6 }}>ORIGINAL</div>
            <div style={{ background: '#f8f7ff', borderRadius: 12, border: '1.5px solid #c7d2fe', padding: 4 }}>
              <canvas ref={origRef2} width={56} height={56} style={canvasStyle} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>pred: {TRUE_LABEL}</div>
          </div>
          <div style={{ fontSize: 24, color: 'var(--text3)' }}>→</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#dc2626', marginBottom: 6 }}>ADVERSARIAL</div>
            <div style={{ background: '#fff1f2', borderRadius: 12, border: '1.5px solid #fecaca', padding: 4 }}>
              <canvas ref={advRef} width={56} height={56} style={canvasStyle} />
            </div>
            <div style={{ fontSize: 11, color: '#dc2626', marginTop: 4 }}>pred: {ADV_PRED}</div>
          </div>
          <div style={{ flex: 1 }}>
            <Note>사람이 보기에는 두 이미지가 거의 동일합니다. 그러나 모델의 내부 표현 공간에서는 결정 경계를 넘었습니다.</Note>
          </div>
        </div>
      </ProcessStepCard>

      <ProcessStepCard stepNum="STEP 08" title="Re-Prediction">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 14 }}>
          <ProbabilityBarChart probs={ORIG_PROBS} highlight={TRUE_LABEL} label="BEFORE ATTACK" />
          <ProbabilityBarChart probs={ADV_PROBS} highlight={ADV_PRED} label="AFTER ATTACK" />
        </div>
        <div style={{ display: 'inline-block', padding: '8px 18px', borderRadius: 99, background: '#fee2e2', color: '#dc2626', fontSize: 13, fontWeight: 600 }}>
          🎭 Prediction changed: {TRUE_LABEL} → {ADV_PRED}
        </div>
        <Note>공격 이전 {TRUE_LABEL}에 98.4%의 확률을 보였던 모델이, 공격 이후에는 {ADV_PRED}에 91.2%를 부여합니다.</Note>
      </ProcessStepCard>
    </div>
  )
}
