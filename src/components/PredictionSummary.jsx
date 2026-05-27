import React from 'react'

export default function PredictionSummary({ result }) {
  if (!result) {
    return (
      <div style={{ background: 'var(--bg2)', borderRadius: 16, border: '1px solid var(--border)', padding: 18 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1px', marginBottom: 16 }}>
          📊 PREDICTION CHANGE
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 0', color: 'var(--text3)', fontSize: 13, textAlign: 'center', gap: 8 }}>
          <span style={{ fontSize: 32 }}>🎯</span>
          Run attack<br />to see results
        </div>
      </div>
    )
  }

  const { original, adversarial, attack, result: res } = result

  return (
    <div style={{ background: 'var(--bg2)', borderRadius: 16, border: '1px solid var(--border)', padding: 18 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1px', marginBottom: 16 }}>
        📊 PREDICTION CHANGE
      </div>

      {/* Big shift */}
      <div style={{ textAlign: 'center', padding: '16px 0', marginBottom: 14, borderRadius: 12, background: res.predictionChanged ? '#fef2f2' : '#f0fdf4', border: `1px solid ${res.predictionChanged ? '#fecaca' : '#bbf7d0'}` }}>
        <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>PREDICTION SHIFT</div>
        <div style={{ fontSize: 30, fontWeight: 700, color: res.predictionChanged ? '#dc2626' : '#16a34a', letterSpacing: 6 }}>
          {res.predictionShiftText}
        </div>
        <div style={{ marginTop: 10 }}>
          <span style={{
            display: 'inline-block', padding: '3px 12px', borderRadius: 99,
            fontSize: 11, fontWeight: 600,
            background: res.attackSuccess ? '#dcfce7' : '#fee2e2',
            color: res.attackSuccess ? '#15803d' : '#dc2626',
          }}>
            {res.attackSuccess ? '✓ Attack Success' : '✗ Attack Failed'}
          </span>
        </div>
      </div>

      {/* Stats */}
      {[
        { label: 'Before Pred', value: original.prediction, mono: true },
        { label: 'After Pred', value: adversarial.prediction, mono: true, red: res.predictionChanged },
        { label: 'Conf Before', value: `${(original.confidence * 100).toFixed(1)}%` },
        { label: 'Conf After', value: `${(adversarial.confidence * 100).toFixed(1)}%`, red: true },
        { label: 'Conf Δ', value: `${res.confidenceDifference > 0 ? '+' : ''}${(res.confidenceDifference * 100).toFixed(1)}%`, red: res.confidenceDifference < 0 },
        { label: 'Epsilon', value: attack.epsilon.toFixed(2), accent: true },
        { label: 'Loss Before', value: attack.lossBefore.toFixed(3) },
        { label: 'Loss After', value: attack.lossAfter.toFixed(3), red: true },
      ].map(({ label, value, red, accent, mono }) => (
        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
          <span style={{ fontSize: 11, color: 'var(--text2)' }}>{label}</span>
          <span style={{
            fontSize: 13, fontWeight: 600,
            color: red ? '#dc2626' : accent ? 'var(--accent)' : 'var(--text)',
            fontFamily: mono ? "'DM Mono', monospace" : 'inherit',
          }}>{value}</span>
        </div>
      ))}

      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text3)', lineHeight: 1.6, padding: '10px 12px', background: 'var(--bg)', borderRadius: 8 }}>
        입력 이미지는 거의 같지만, 모델의 예측 결과가 변경되었습니다.
      </div>
    </div>
  )
}
