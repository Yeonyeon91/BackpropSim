import React from 'react'
import { useNavigate } from 'react-router-dom'

const FLOW_STEPS = [
  { n: '01', emoji: '🖼️', t: 'Original Input',      d: '원본 이미지를 로드합니다. AI가 첫 번째 예측을 수행합니다.' },
  { n: '02', emoji: '➡️', t: 'Forward Propagation', d: '이미지가 레이어를 통과하며 각 클래스의 확률이 계산됩니다.' },
  { n: '03', emoji: '📉', t: 'Loss Calculation',     d: '예측과 목표 사이의 오차를 수치로 표현합니다.' },
  { n: '04', emoji: '🔁', t: 'Backpropagation',      d: '오차를 역방향으로 전파해 gradient를 계산합니다.' },
  { n: '05', emoji: '🌡️', t: 'Input Gradient',       d: '어떤 픽셀이 예측에 가장 큰 영향을 주는지 heatmap으로 표시합니다.' },
  { n: '06', emoji: '⚡', t: 'Perturbation',          d: 'gradient 방향으로 미세한 픽셀 변화를 원본에 추가합니다.' },
  { n: '07', emoji: '🎭', t: 'Prediction Change',     d: '변형된 이미지로 재예측 — AI의 판단이 바뀌었습니다.' },
]

const PAGES = [
  {
    path: '/lab',
    emoji: '⚗️',
    title: 'Attack Lab',
    desc: 'epsilon을 조절하며 FGSM 공격을 직접 실행. 이미지 변화와 예측 변화를 실시간으로 확인합니다.',
    color: '#7c6ff7',
    bg: '#f5f3ff',
  },
  {
    path: '/visualizer',
    emoji: '🧠',
    title: 'Backprop Visualizer',
    desc: '신경망 학습의 전 과정을 직접 체험. 순전파→오차→역전파→weight 업데이트를 단계별로 조작합니다.',
    color: '#059669',
    bg: '#ecfdf5',
  },
  {
    path: '/process',
    emoji: '📖',
    title: 'Process Explorer',
    desc: '공격 과정 8단계를 이론과 수식으로 설명하는 발표용 레퍼런스 페이지입니다.',
    color: '#d97706',
    bg: '#fffbeb',
  },
  {
    path: '/log',
    emoji: '📋',
    title: 'Experiment Log',
    desc: '실험 결과를 테이블로 비교하고 필터링. 성공/실패 조건을 분석합니다.',
    color: '#db2777',
    bg: '#fdf2f8',
  },
]

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div>
      {/* Hero */}
      <div style={{ textAlign: 'center', padding: '40px 0 36px' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          background: 'var(--tag-bg)', color: 'var(--tag-text)',
          padding: '4px 14px', borderRadius: 99,
          fontSize: 12, fontWeight: 500, marginBottom: 20,
        }}>
          ✦ AI Security · Education Tool
        </div>
        <h1 style={{
          fontSize: 36, fontWeight: 700, color: 'var(--text)',
          letterSpacing: '-0.5px', lineHeight: 1.2, marginBottom: 14,
        }}>
          Backpropagation Attack<br />
          <span style={{ color: 'var(--accent)' }}>Simulator</span>
        </h1>
        <p style={{ fontSize: 15, color: 'var(--text2)', maxWidth: 520, margin: '0 auto 28px', lineHeight: 1.7 }}>
          역전파로 계산된 Gradient가 AI의 판단을 어떻게 바꾸는지,<br />
          단계별로 직접 체험하는 교육용 시뮬레이터입니다.
        </p>
        <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
          <button onClick={() => navigate('/lab')} style={{
            background: 'var(--accent)', color: '#fff',
            border: 'none', borderRadius: 12,
            padding: '11px 28px', fontSize: 14, fontWeight: 600,
            cursor: 'pointer', transition: 'all .15s',
            boxShadow: '0 4px 14px rgba(124,111,247,0.35)',
          }}
          onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
          onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
          >
            ▶ Start Attack Lab
          </button>
          <button onClick={() => navigate('/visualizer')} style={{
            background: 'var(--bg2)', color: 'var(--text)',
            border: '1.5px solid var(--border2)', borderRadius: 12,
            padding: '11px 28px', fontSize: 14, fontWeight: 500,
            cursor: 'pointer', transition: 'all .15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text)' }}
          >
            🧠 Backprop Visualizer
          </button>
        </div>
      </div>

      {/* Flow steps */}
      <div style={{
        background: 'var(--bg2)', borderRadius: 20,
        border: '1px solid var(--border)', padding: '24px',
        marginBottom: 28,
      }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '1.5px', marginBottom: 16 }}>
          ATTACK PIPELINE
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 8 }}>
          {FLOW_STEPS.map((step, i) => (
            <div key={step.n} style={{ position: 'relative' }}>
              <div style={{
                background: i === 0 || i === 6 ? 'var(--tag-bg)' : 'var(--bg)',
                borderRadius: 12, padding: '12px 10px',
                border: '1px solid var(--border)',
                height: '100%',
              }}>
                <div style={{ fontSize: 18, marginBottom: 6 }}>{step.emoji}</div>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', marginBottom: 4, letterSpacing: '1px' }}>
                  {step.n}
                </div>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 6, lineHeight: 1.3 }}>
                  {step.t}
                </div>
                <div style={{ fontSize: 10.5, color: 'var(--text2)', lineHeight: 1.55 }}>
                  {step.d}
                </div>
              </div>
              {i < FLOW_STEPS.length - 1 && (
                <div style={{
                  position: 'absolute', right: -6, top: '40%',
                  fontSize: 13, color: 'var(--text3)', zIndex: 1,
                }}>›</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Page cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
        {PAGES.map(pg => (
          <div
            key={pg.path}
            onClick={() => navigate(pg.path)}
            style={{
              background: 'var(--bg2)', borderRadius: 16,
              border: '1px solid var(--border)', padding: '20px',
              cursor: 'pointer', transition: 'all .2s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.transform = 'translateY(-3px)'
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(124,111,247,0.12)'
              e.currentTarget.style.borderColor = pg.color
            }}
            onMouseLeave={e => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = 'none'
              e.currentTarget.style.borderColor = 'var(--border)'
            }}
          >
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: pg.bg, display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: 20, marginBottom: 12,
            }}>
              {pg.emoji}
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>
              {pg.title}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
              {pg.desc}
            </div>
            <div style={{ marginTop: 14, fontSize: 12, fontWeight: 500, color: pg.color }}>
              열기 →
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
