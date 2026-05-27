// attackApi.js
// 백엔드 API 추상화 레이어.
// USE_MOCK = true  → mock 데이터 사용 (백엔드 없이 개발)
// USE_MOCK = false → FastAPI 백엔드 실제 호출

import {
  SAMPLES,
  ORIGINAL_PROBS,
  ADVERSARIAL_PROBS,
  ADVERSARIAL_PREDS,
} from '../data/mockAttackData.js'

const USE_MOCK = true
const API_BASE_URL = 'http://localhost:8000' // FastAPI 서버 주소

// ─────────────────────────────────────────────
// 공격 실행
// input:  { sampleId, epsilon, targetLabel }
// output: AttackResult 객체
// ─────────────────────────────────────────────
export async function runAttack({ sampleId, epsilon, targetLabel }) {
  if (USE_MOCK) {
    return _mockRunAttack({ sampleId, epsilon, targetLabel })
  }

  // FastAPI 연결 시 이 부분만 활성화
  const response = await fetch(`${API_BASE_URL}/attack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_id: sampleId, epsilon, target_label: targetLabel }),
  })
  if (!response.ok) {
    throw new Error(`Attack API error: ${response.status}`)
  }
  return response.json()
}

// ─────────────────────────────────────────────
// Mock 구현 (실제 AI 처리를 시뮬레이션)
// ─────────────────────────────────────────────
async function _mockRunAttack({ sampleId, epsilon, targetLabel }) {
  // 처리 시간 시뮬레이션 (실제처럼 보이도록)
  await new Promise(r => setTimeout(r, 300))

  const sample = SAMPLES[sampleId]
  const trueLabel = sample.trueLabel
  const origProbs = ORIGINAL_PROBS[sampleId]
  const advProbs = ADVERSARIAL_PROBS[sampleId]
  const advPred = ADVERSARIAL_PREDS[sampleId]

  const origConf = origProbs[String(trueLabel)]
  const advConf = advProbs[String(advPred)]

  // epsilon 크기에 따라 loss, gradient 값 조정
  const lossBefore = 0.018 + Math.random() * 0.005
  const lossAfter = epsilon > 0.2 ? 3.1 + Math.random() * 0.3 : epsilon > 0.1 ? 2.414 + Math.random() * 0.2 : 1.8 + Math.random() * 0.3
  const gradientScale = epsilon * 10

  const attackSuccess = epsilon > 0.05

  return {
    id: `exp-${Date.now()}`,
    sampleId,
    trueLabel,

    original: {
      imageUrl: `/mock/original_${trueLabel}.png`,
      prediction: trueLabel,
      confidence: origConf,
      probabilities: origProbs,
    },

    attack: {
      epsilon: parseFloat(epsilon),
      targetLabel: targetLabel === 'auto' ? advPred : parseInt(targetLabel),
      lossBefore: parseFloat(lossBefore.toFixed(3)),
      lossAfter: parseFloat(lossAfter.toFixed(3)),
      gradientImageUrl: `/mock/gradient_${trueLabel}.png`,
      perturbationImageUrl: `/mock/perturbation_${trueLabel}.png`,
      differenceImageUrl: `/mock/diff_${trueLabel}.png`,
      gradientMin: parseFloat((-0.032 * gradientScale).toFixed(4)),
      gradientMax: parseFloat((0.041 * gradientScale).toFixed(4)),
      topInfluentialPixels: Math.round(84 * gradientScale),
    },

    adversarial: {
      imageUrl: `/mock/adversarial_${trueLabel}_to_${advPred}.png`,
      prediction: advPred,
      confidence: advConf,
      probabilities: advProbs,
    },

    result: {
      predictionChanged: attackSuccess,
      attackSuccess,
      confidenceDifference: parseFloat((advConf - origConf).toFixed(4)),
      predictionShiftText: `${trueLabel} → ${attackSuccess ? advPred : trueLabel}`,
    },
  }
}

// ─────────────────────────────────────────────
// 실험 로그 저장 (옵션 - 백엔드 연결 시 사용)
// ─────────────────────────────────────────────
export async function saveExperiment(result) {
  if (USE_MOCK) {
    console.log('[Mock] Experiment saved:', result.id)
    return { success: true }
  }

  const response = await fetch(`${API_BASE_URL}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result),
  })
  return response.json()
}

// ─────────────────────────────────────────────
// 실험 로그 조회 (옵션)
// ─────────────────────────────────────────────
export async function getExperiments() {
  if (USE_MOCK) {
    return []
  }

  const response = await fetch(`${API_BASE_URL}/experiments`)
  return response.json()
}
