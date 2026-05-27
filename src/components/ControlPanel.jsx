import React from 'react'
import { SAMPLES } from '../data/mockAttackData.js'

const S = {
  panel: { background: 'var(--bg2)', borderRadius: 16, border: '1px solid var(--border)', padding: 18 },
  label: { fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.8px', marginBottom: 5, display: 'block' },
  select: {
    width: '100%', background: 'var(--bg)', color: 'var(--text)',
    border: '1.5px solid var(--border)', borderRadius: 10,
    padding: '8px 12px', fontSize: 13, fontFamily: 'inherit',
    marginBottom: 16, outline: 'none', transition: 'border-color .15s',
  },
  bigNum: { fontSize: 32, fontWeight: 700, color: 'var(--accent)', letterSpacing: '-0.5px', marginBottom: 4 },
  btnPrimary: {
    width: '100%', background: 'var(--accent)', color: '#fff',
    border: 'none', borderRadius: 10, padding: '10px 0',
    fontSize: 13, fontWeight: 600, cursor: 'pointer',
    marginBottom: 8, transition: 'all .15s',
    boxShadow: '0 4px 12px rgba(124,111,247,0.3)',
  },
  btnSecondary: {
    width: '100%', background: 'transparent', color: 'var(--text2)',
    border: '1.5px solid var(--border)', borderRadius: 10, padding: '9px 0',
    fontSize: 13, fontWeight: 500, cursor: 'pointer', transition: 'all .15s',
  },
}

export default function ControlPanel({ sampleId, setSampleId, epsilon, setEpsilon, targetLabel, setTargetLabel, onAttack, onReset, loading }) {
  return (
    <div style={S.panel}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1px', marginBottom: 18, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>⚙️</span> CONTROL PANEL
      </div>

      <label style={S.label}>SAMPLE</label>
      <select
        value={sampleId} onChange={e => setSampleId(e.target.value)}
        style={S.select}
        onFocus={e => e.target.style.borderColor = 'var(--accent)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      >
        {Object.entries(SAMPLES).map(([k, v]) => (
          <option key={k} value={k}>{k.replace('sample-', 'Sample #')} — {v.description}</option>
        ))}
      </select>

      <label style={S.label}>EPSILON</label>
      <div style={S.bigNum}>{parseFloat(epsilon).toFixed(2)}</div>
      <input
        type="range" min="0" max="0.30" step="0.01" value={epsilon}
        onChange={e => setEpsilon(e.target.value)}
        style={{ width: '100%', marginBottom: 4 }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text3)', marginBottom: 18 }}>
        <span>0.00</span><span>0.15</span><span>0.30</span>
      </div>

      <label style={S.label}>TARGET LABEL</label>
      <select
        value={targetLabel} onChange={e => setTargetLabel(e.target.value)}
        style={S.select}
        onFocus={e => e.target.style.borderColor = 'var(--accent)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      >
        <option value="auto">Auto</option>
        {Array.from({ length: 10 }, (_, i) => <option key={i} value={String(i)}>{i}</option>)}
      </select>

      <button
        onClick={onAttack} disabled={loading}
        style={{ ...S.btnPrimary, opacity: loading ? 0.6 : 1 }}
        onMouseEnter={e => { if (!loading) e.currentTarget.style.transform = 'translateY(-1px)' }}
        onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
      >
        {loading ? '⏳ Running...' : '▶ Run Attack'}
      </button>
      <button
        onClick={onReset} style={S.btnSecondary}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent2)'; e.currentTarget.style.color = 'var(--accent)' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text2)' }}
      >
        ↺ Reset
      </button>

      <div style={{ marginTop: 14, fontSize: 11, color: 'var(--text3)', lineHeight: 1.6, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
        💡 epsilon이 클수록 공격 성공률↑, 이미지 변형↑
      </div>
    </div>
  )
}
