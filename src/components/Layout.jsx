import React from 'react'
import Header from './Header.jsx'

export default function Layout({ children }) {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <Header />
      <main style={{ maxWidth: 1440, margin: '0 auto', padding: '28px 24px 48px' }}>
        {children}
      </main>
    </div>
  )
}
