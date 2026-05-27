import React from 'react'
import ExperimentLogTable from '../components/ExperimentLogTable.jsx'

export default function ExperimentLogPage({ log = [] }) {
  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>📋 Experiment Log</h2>
        <p style={{ fontSize: 13, color: 'var(--text2)' }}>
          공격 실험 결과를 기록하고 비교합니다. Attack Lab에서 실행한 실험이 자동으로 추가됩니다.
        </p>
      </div>
      <ExperimentLogTable log={log} />
    </div>
  )
}
