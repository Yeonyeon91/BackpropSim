import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'

const NAV_TABS = [
  { path: '/', label: 'Home', exact: true },
  { path: '/lab', label: 'Attack Lab' },
  { path: '/process', label: 'Process Explorer' },
  { path: '/visualizer', label: 'Backprop Visualizer' },
  { path: '/log', label: 'Experiment Log' },
]

export default function Header() {
  return (
    <nav style={{
      background: 'rgba(247,246,243,0.85)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky', top: 0, zIndex: 50,
    }}>
      <div style={{ maxWidth: 1440, margin: '0 auto', padding: '0 24px', display: 'flex', alignItems: 'center', gap: 0 }}>
        {/* Brand */}
        <div style={{
          padding: '14px 20px 14px 0',
          marginRight: 20,
          borderRight: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 8,
          userSelect: 'none',
        }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 12, fontWeight: 700, color: '#fff',
          }}>∇</div>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', letterSpacing: '0.5px' }}>
            BPA Sim
          </span>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', overflowX: 'auto', gap: 2 }}>
          {NAV_TABS.map(tab => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end={tab.exact}
              style={({ isActive }) => ({
                padding: '14px 14px',
                fontSize: 13,
                fontWeight: isActive ? 500 : 400,
                color: isActive ? 'var(--accent)' : 'var(--text2)',
                textDecoration: 'none',
                whiteSpace: 'nowrap',
                borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                transition: 'all .15s',
                borderRadius: '0',
              })}
              onMouseEnter={e => { if (!e.currentTarget.classList.contains('active')) e.currentTarget.style.color = 'var(--text)' }}
              onMouseLeave={e => { if (!e.currentTarget.classList.contains('active')) e.currentTarget.style.color = 'var(--text2)' }}
            >
              {tab.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
