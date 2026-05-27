import React, { useRef, useEffect } from 'react'

const DIGIT_STROKES = {
  7: [[28,12,36,12],[36,12,22,44]],
  3: [[18,12,38,12],[18,22,34,22],[18,32,38,32],[38,12,38,44],[18,44,38,44]],
  5: [[18,12,38,12],[18,12,18,28],[18,28,36,28],[36,28,38,44],[18,44,38,44]],
  1: [[28,12,28,44]],
  2: [[18,12,36,12],[36,12,36,28],[18,28,36,28],[18,28,18,44],[18,44,36,44]],
  6: [[36,12,18,12],[18,12,18,44],[18,44,36,44],[36,44,36,28],[18,28,36,28]],
}

export function drawDigit(canvas, digit, type = 'original') {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const s = 56
  canvas.width = s; canvas.height = s

  // Background
  ctx.fillStyle = type === 'original' ? '#f8f7ff' : type === 'adversarial' ? '#fff1f2' : type === 'gradient' ? '#f0fdf4' : '#fffbeb'
  ctx.fillRect(0, 0, s, s)

  const strokes = DIGIT_STROKES[digit] || DIGIT_STROKES[7]

  if (type === 'gradient') {
    for (let i = 0; i < 60; i++) {
      const alpha = Math.random() * 0.5
      ctx.fillStyle = Math.random() > 0.5 ? `rgba(124,111,247,${alpha})` : `rgba(52,211,153,${alpha})`
      ctx.fillRect(Math.random() * s, Math.random() * s, 3 + Math.random() * 2, 3 + Math.random() * 2)
    }
    ctx.strokeStyle = 'rgba(124,111,247,0.5)'
    ctx.lineWidth = 3; ctx.lineCap = 'round'
    strokes.forEach(([x1,y1,x2,y2]) => { ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke() })
    return
  }

  if (type === 'perturbation') {
    ctx.fillStyle = '#fffbeb'
    ctx.fillRect(0, 0, s, s)
    for (let i = 0; i < 400; i++) {
      const pos = Math.random() > 0.5
      const alpha = Math.random() * 0.7 + 0.1
      ctx.fillStyle = pos ? `rgba(249,115,22,${alpha})` : `rgba(124,111,247,${alpha})`
      ctx.fillRect(Math.random() * s, Math.random() * s, 1.5, 1.5)
    }
    return
  }

  const colorMap = { original: '#4f46e5', adversarial: '#dc2626' }
  ctx.strokeStyle = colorMap[type] || '#4f46e5'
  ctx.lineWidth = 4; ctx.lineCap = 'round'
  strokes.forEach(([x1,y1,x2,y2]) => { ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke() })

  if (type === 'adversarial') {
    ctx.strokeStyle = 'rgba(220,38,38,0.2)'; ctx.lineWidth = 1
    for (let i = 0; i < 10; i++) { ctx.beginPath(); ctx.moveTo(Math.random()*s, Math.random()*s); ctx.lineTo(Math.random()*s, Math.random()*s); ctx.stroke() }
  }
}

export default function ImageResultCard({ stepNum, title, digit, type = 'original', prediction, confidence, note, placeholder = 'Run attack first', showResult = true, extraInfo }) {
  const canvasRef = useRef()
  useEffect(() => { if (digit !== undefined && digit !== null) drawDigit(canvasRef.current, digit, type) }, [digit, type])
  const hasImage = digit !== undefined && digit !== null

  const bgColors = { original: '#f8f7ff', adversarial: '#fff1f2', gradient: '#f0fdf4', perturbation: '#fffbeb' }
  const borderColors = { original: '#c7d2fe', adversarial: '#fecaca', gradient: '#bbf7d0', perturbation: '#fed7aa' }
  const labelColors = { original: '#4f46e5', adversarial: '#dc2626', gradient: '#059669', perturbation: '#d97706' }

  return (
    <div style={{
      background: 'var(--bg2)', borderRadius: 14, border: '1px solid var(--border)', padding: 12,
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.8px', marginBottom: 8 }}>
        {stepNum && <span style={{ color: 'var(--accent2)', marginRight: 4 }}>{stepNum}</span>}
        {title}
      </div>

      <div style={{
        width: '100%', aspectRatio: '1', borderRadius: 10,
        background: hasImage ? bgColors[type] : 'var(--bg)',
        border: `1.5px solid ${hasImage ? borderColors[type] : 'var(--border)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        marginBottom: 10, overflow: 'hidden',
      }}>
        {hasImage
          ? <canvas ref={canvasRef} width={56} height={56} style={{ width: '100%', height: '100%' }} />
          : <span style={{ fontSize: 11, color: 'var(--text3)', textAlign: 'center', padding: 8 }}>{placeholder}</span>
        }
      </div>

      {showResult && prediction !== undefined && prediction !== null && (
        <>
          <div style={{ fontSize: 34, fontWeight: 700, textAlign: 'center', color: labelColors[type] }}>{prediction}</div>
          {confidence !== undefined && (
            <div style={{ fontSize: 12, color: 'var(--text2)', textAlign: 'center' }}>{(confidence * 100).toFixed(1)}%</div>
          )}
          <div style={{ fontSize: 10, color: 'var(--text3)', textAlign: 'center', marginTop: 2 }}>
            {type === 'adversarial' ? 'Adversarial Pred' : 'Original Pred'}
          </div>
        </>
      )}
      {extraInfo && <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text2)' }}>{extraInfo}</div>}
      {note && (
        <div style={{ marginTop: 8, fontSize: 10.5, color: 'var(--text2)', lineHeight: 1.6, padding: '6px 8px', background: 'var(--bg)', borderRadius: 6 }}>
          {note}
        </div>
      )}
    </div>
  )
}
