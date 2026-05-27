import React from 'react'

export default function ProbabilityBarChart({ probs, highlight, label }) {
  if (!probs) return null
  return (
    <div style={{ width: '100%' }}>
      {label && <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.8px', marginBottom: 8 }}>{label}</div>}
      {Object.entries(probs).map(([k, v]) => {
        const isHL = String(k) === String(highlight)
        const pct = (v * 100).toFixed(1)
        return (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <div style={{ fontSize: 11, color: 'var(--text2)', width: 14, textAlign: 'right', fontFamily: "'DM Mono', monospace", fontWeight: isHL ? 700 : 400 }}>{k}</div>
            <div style={{ flex: 1, height: 14, background: 'var(--bg)', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 99,
                width: `${pct}%`,
                background: isHL ? 'var(--accent)' : 'var(--border2)',
                transition: 'width .5s cubic-bezier(.4,0,.2,1)',
              }} />
            </div>
            <div style={{ fontSize: 11, width: 36, textAlign: 'right', fontFamily: "'DM Mono', monospace", color: isHL ? 'var(--accent)' : 'var(--text3)', fontWeight: isHL ? 700 : 400 }}>
              {pct}%
            </div>
          </div>
        )
      })}
    </div>
  )
}
