import React, { useRef, useEffect } from 'react'
import { drawDigit } from './ImageResultCard.jsx'

export default function PerturbationCard({ digit, epsilon }) {
  const canvasRef = useRef()
  useEffect(() => { if (digit !== undefined && digit !== null) drawDigit(canvasRef.current, digit, 'perturbation') }, [digit])
  const hasData = digit !== undefined && digit !== null

  return (
    <div style={{ background: 'var(--bg2)', borderRadius: 14, border: '1px solid var(--border)', padding: 12, display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.8px', marginBottom: 8 }}>
        <span style={{ color: 'var(--accent2)', marginRight: 4 }}>03 ·</span>PERTURBATION
      </div>

      <div style={{
        width: '100%', aspectRatio: '1', borderRadius: 10,
        background: hasData ? '#fffbeb' : 'var(--bg)',
        border: `1.5px solid ${hasData ? '#fed7aa' : 'var(--border)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        marginBottom: 10, overflow: 'hidden',
      }}>
        {hasData
          ? <canvas ref={canvasRef} width={56} height={56} style={{ width: '100%', height: '100%' }} />
          : <span style={{ fontSize: 11, color: 'var(--text3)' }}>Run attack</span>
        }
      </div>

      {hasData && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 11, color: 'var(--text2)' }}>ε =</span>
          <span style={{ fontSize: 22, fontWeight: 700, color: '#d97706' }}>{parseFloat(epsilon).toFixed(2)}</span>
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        {[['#f97316', '+ gradient'], ['#7c6ff7', '− gradient']].map(([c, l]) => (
          <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 10, height: 10, borderRadius: 3, background: c, opacity: 0.7 }} />
            <span style={{ fontSize: 10, color: 'var(--text3)' }}>{l}</span>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 10.5, color: 'var(--text2)', lineHeight: 1.6, padding: '6px 8px', background: 'var(--bg)', borderRadius: 6 }}>
        gradient 방향으로 생성된 픽셀 변화
      </div>
    </div>
  )
}
