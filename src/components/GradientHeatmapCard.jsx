import React, { useRef, useEffect } from 'react'
import { drawDigit } from './ImageResultCard.jsx'

export default function GradientHeatmapCard({ digit, gradientMin, gradientMax, topPixels }) {
  const canvasRef = useRef()
  useEffect(() => { if (digit !== undefined && digit !== null) drawDigit(canvasRef.current, digit, 'gradient') }, [digit])
  const hasData = digit !== undefined && digit !== null

  return (
    <div style={{ background: 'var(--bg2)', borderRadius: 14, border: '1px solid var(--border)', padding: 12, display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.8px', marginBottom: 8 }}>
        <span style={{ color: 'var(--accent2)', marginRight: 4 }}>02 ·</span>INPUT GRADIENT
      </div>

      <div style={{
        width: '100%', aspectRatio: '1', borderRadius: 10,
        background: hasData ? '#f0fdf4' : 'var(--bg)',
        border: `1.5px solid ${hasData ? '#bbf7d0' : 'var(--border)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        marginBottom: 10, overflow: 'hidden',
      }}>
        {hasData
          ? <canvas ref={canvasRef} width={56} height={56} style={{ width: '100%', height: '100%' }} />
          : <span style={{ fontSize: 11, color: 'var(--text3)' }}>Run attack</span>
        }
      </div>

      {hasData && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginBottom: 8 }}>
          {[
            { label: 'MIN', value: gradientMin?.toFixed(3), color: '#7c6ff7' },
            { label: 'MAX', value: `+${gradientMax?.toFixed(3)}`, color: '#059669' },
            { label: 'TOP PX', value: topPixels, color: '#d97706' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: 'var(--bg)', borderRadius: 6, padding: '5px 4px', textAlign: 'center' }}>
              <div style={{ fontSize: 9, color: 'var(--text3)', marginBottom: 2 }}>{label}</div>
              <div style={{ fontSize: 11, fontWeight: 700, color }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 6 }}>
        <span style={{ fontSize: 9, color: 'var(--text3)' }}>LOW</span>
        <div style={{ flex: 1, height: 6, borderRadius: 99, background: 'linear-gradient(to right, #c7d2fe, #7c6ff7, #f59e0b)' }} />
        <span style={{ fontSize: 9, color: 'var(--text3)' }}>HIGH</span>
      </div>

      <div style={{ fontSize: 10.5, color: 'var(--text2)', lineHeight: 1.6, padding: '6px 8px', background: 'var(--bg)', borderRadius: 6 }}>
        밝은 영역 = 예측에 큰 영향을 주는 픽셀
      </div>
    </div>
  )
}
