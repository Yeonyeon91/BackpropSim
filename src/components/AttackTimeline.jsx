import React from 'react'

const STEPS = ['Forward', 'Loss', 'Backprop', 'Gradient', 'Perturb', 'Re-Predict']

export default function AttackTimeline({ activeStep }) {
  const done = activeStep >= STEPS.length

  return (
    <div style={{ background: 'var(--bg2)', borderRadius: 16, border: '1px solid var(--border)', padding: '14px 18px', marginTop: 14 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1px', marginBottom: 14 }}>
        ⏱ ATTACK TIMELINE
      </div>
      <div style={{ display: 'flex', alignItems: 'center', overflowX: 'auto' }}>
        {STEPS.map((step, i) => {
          const isRunning = !done && activeStep === i
          const isDone = done || activeStep > i
          return (
            <React.Fragment key={step}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, minWidth: 72 }}>
                <div
                  className={isRunning ? 'tl-running' : ''}
                  style={{
                    width: 32, height: 32, borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, marginBottom: 6,
                    transition: 'all .3s',
                    background: isDone ? '#dcfce7' : isRunning ? '#fef3c7' : 'var(--bg)',
                    color: isDone ? '#15803d' : isRunning ? '#d97706' : 'var(--text3)',
                    border: `2px solid ${isDone ? '#86efac' : isRunning ? '#fcd34d' : 'var(--border)'}`,
                  }}
                >
                  {isDone ? '✓' : i + 1}
                </div>
                <div style={{ fontSize: 10, color: isDone ? '#15803d' : isRunning ? '#d97706' : 'var(--text3)', fontWeight: isDone || isRunning ? 600 : 400, textAlign: 'center' }}>
                  {step}
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div style={{ flex: '0 0 20px', height: 2, background: activeStep > i ? '#86efac' : 'var(--border)', marginBottom: 18, transition: 'background .3s' }} />
              )}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}
