import React, { useState, useRef, useEffect } from 'react'
import { drawDigit } from './ImageResultCard.jsx'
import { MOCK_EXPERIMENT_LOG } from '../data/mockAttackData.js'

function MiniCanvas({ digit, type }) {
  const ref = useRef()
  useEffect(() => { if (digit !== undefined && digit !== null) drawDigit(ref.current, digit, type) }, [digit, type])
  return <canvas ref={ref} width={56} height={56} style={{ width: 56, height: 56, borderRadius: 6 }} />
}

export default function ExperimentLogTable({ log = [] }) {
  const [selected, setSelected] = useState(null)
  const [filterSuccess, setFilterSuccess] = useState(false)
  const [epsFilter, setEpsFilter] = useState('all')

  const allLogs = [...log, ...MOCK_EXPERIMENT_LOG]
  const filtered = allLogs.filter(r => {
    if (filterSuccess && !r.success) return false
    if (epsFilter === 'low' && r.epsilon > 0.1) return false
    if (epsFilter === 'high' && r.epsilon <= 0.1) return false
    return true
  })
  const selRow = selected ? allLogs.find(r => r.id === selected) : null

  const selStyle = {
    select: { background: 'var(--bg2)', color: 'var(--text)', border: '1.5px solid var(--border)', borderRadius: 8, padding: '6px 10px', fontSize: 12, fontFamily: 'inherit', outline: 'none' },
  }

  return (
    <div>
      {/* Filters */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text2)', cursor: 'pointer' }}>
          <input type="checkbox" checked={filterSuccess} onChange={e => setFilterSuccess(e.target.checked)} style={{ accentColor: 'var(--accent)', width: 14, height: 14 }} />
          Success only
        </label>
        <select value={epsFilter} onChange={e => setEpsFilter(e.target.value)} style={selStyle.select}>
          <option value="all">All Epsilon</option>
          <option value="low">ε ≤ 0.10</option>
          <option value="high">ε &gt; 0.10</option>
        </select>
        <span style={{ fontSize: 12, color: 'var(--text3)', marginLeft: 'auto' }}>{filtered.length} experiments</span>
      </div>

      {/* Table */}
      <div style={{ background: 'var(--bg2)', borderRadius: 14, border: '1px solid var(--border)', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
              {['Time', 'Sample', 'Original', 'Adversarial', 'Epsilon', 'Conf Before', 'Conf After', 'Result'].map(col => (
                <th key={col} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.5px' }}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr><td colSpan={8} style={{ padding: '32px', textAlign: 'center', color: 'var(--text3)', fontSize: 13 }}>No experiments match the filter</td></tr>
            ) : filtered.map(r => (
              <tr
                key={r.id}
                onClick={() => setSelected(selected === r.id ? null : r.id)}
                style={{
                  borderBottom: '1px solid var(--border)',
                  cursor: 'pointer',
                  background: selected === r.id ? '#f5f3ff' : 'transparent',
                  transition: 'background .1s',
                }}
                onMouseEnter={e => { if (selected !== r.id) e.currentTarget.style.background = 'var(--bg)' }}
                onMouseLeave={e => { if (selected !== r.id) e.currentTarget.style.background = 'transparent' }}
              >
                <td style={{ padding: '10px 14px', fontFamily: "'DM Mono', monospace", fontSize: 12, color: 'var(--text2)' }}>{r.time || '--:--:--'}</td>
                <td style={{ padding: '10px 14px', color: 'var(--text2)', fontSize: 12 }}>{r.sample}</td>
                <td style={{ padding: '10px 14px', fontSize: 22, fontWeight: 700, color: 'var(--accent)' }}>{r.origPred}</td>
                <td style={{ padding: '10px 14px', fontSize: 22, fontWeight: 700, color: '#dc2626' }}>{r.advPred}</td>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: '#d97706', fontFamily: "'DM Mono', monospace" }}>{r.epsilon.toFixed(2)}</td>
                <td style={{ padding: '10px 14px', color: 'var(--text2)', fontFamily: "'DM Mono', monospace" }}>{(r.confBefore * 100).toFixed(1)}%</td>
                <td style={{ padding: '10px 14px', color: 'var(--text2)', fontFamily: "'DM Mono', monospace" }}>{(r.confAfter * 100).toFixed(1)}%</td>
                <td style={{ padding: '10px 14px' }}>
                  <span style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: 99, fontSize: 11, fontWeight: 600,
                    background: r.success ? '#dcfce7' : '#fee2e2',
                    color: r.success ? '#15803d' : '#dc2626',
                  }}>
                    {r.success ? '✓ Yes' : '✗ No'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail panel */}
      {selRow && (
        <div style={{ marginTop: 14, background: 'var(--bg2)', borderRadius: 14, border: '1.5px solid var(--accent)', padding: 18 }} className="fade-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>Detail: {selRow.id}</div>
            <span style={{
              padding: '3px 12px', borderRadius: 99, fontSize: 11, fontWeight: 600,
              background: selRow.success ? '#dcfce7' : '#fee2e2',
              color: selRow.success ? '#15803d' : '#dc2626',
            }}>{selRow.success ? '✓ Attack Success' : '✗ Attack Failed'}</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
            {[
              { label: 'Original', digit: selRow.origPred, type: 'original' },
              { label: 'Gradient', digit: selRow.origPred, type: 'gradient' },
              { label: 'Perturbation', digit: selRow.origPred, type: 'perturbation' },
              { label: 'Adversarial', digit: selRow.advPred, type: 'adversarial' },
            ].map(({ label, digit, type }) => (
              <div key={label} style={{ background: 'var(--bg)', borderRadius: 10, padding: 10, textAlign: 'center' }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', marginBottom: 6 }}>{label}</div>
                <MiniCanvas digit={digit} type={type} />
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
            {[
              { label: 'Epsilon', value: selRow.epsilon.toFixed(2), color: '#d97706' },
              { label: 'Conf Δ', value: `${((selRow.confAfter - selRow.confBefore) * 100).toFixed(1)}%`, color: selRow.confAfter < selRow.confBefore ? '#dc2626' : '#15803d' },
              { label: 'Shift', value: `${selRow.origPred} → ${selRow.advPred}`, color: 'var(--accent)' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: 'var(--bg)', borderRadius: 8, padding: '10px', textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
