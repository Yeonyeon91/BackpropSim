import React, { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import HomePage from './pages/HomePage.jsx'
import AttackLabPage from './pages/AttackLabPage.jsx'
import ProcessExplorerPage from './pages/ProcessExplorerPage.jsx'
import ExperimentLogPage from './pages/ExperimentLogPage.jsx'
import BackpropVisualizerPage from './pages/BackpropVisualizerPage.jsx'

export default function App() {
  // Shared experiment log state across pages
  const [experimentLog, setExperimentLog] = useState([])

  function addExperiment(result) {
    setExperimentLog(prev => [
      {
        id: result.id,
        time: new Date().toTimeString().slice(0, 8),
        sample: result.sampleId,
        origPred: result.original.prediction,
        advPred: result.adversarial.prediction,
        epsilon: result.attack.epsilon,
        confBefore: result.original.confidence,
        confAfter: result.adversarial.confidence,
        success: result.result.attackSuccess,
        fullResult: result,
      },
      ...prev,
    ])
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/lab" element={<AttackLabPage onExperiment={addExperiment} />} />
        <Route path="/process" element={<ProcessExplorerPage />} />
        <Route path="/visualizer" element={<BackpropVisualizerPage />} />
        <Route path="/log" element={<ExperimentLogPage log={experimentLog} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
