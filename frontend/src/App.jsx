import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import AnalysisResults from './pages/AnalysisResults';
import ComparisonView from './pages/ComparisonView';
import MonitoringView from './pages/MonitoringView';

export default function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [datasetFile, setDatasetFile] = useState(null);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route
          path="/"
          element={
            <Dashboard
              setAnalysisResult={setAnalysisResult}
              setDatasetFile={setDatasetFile}
            />
          }
        />
        <Route
          path="/results"
          element={
            <AnalysisResults
              result={analysisResult}
              datasetFile={datasetFile}
              setAnalysisResult={setAnalysisResult}
            />
          }
        />
        <Route path="/compare" element={<ComparisonView />} />
        <Route path="/monitor" element={<MonitoringView />} />
      </Routes>
    </Router>
  );
}
