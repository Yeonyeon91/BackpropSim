import React, { useState } from 'react'

export default function ProcessStepCard({ stepNum, title, children }) {
  return (
    <div style={{ borderRadius: 16, border: '1px solid var(--border)', marginBottom: 14, overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 18px', background: 'var(--bg2)', borderBottom: '1px solid var(--border)' }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1px', minWidth: 56 }}>{stepNum}</span>
        <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
      </div>
      <div style={{ padding: '16px 18px', background: 'var(--bg)' }}>
        {children}
      </div>
    </div>
  )
}
