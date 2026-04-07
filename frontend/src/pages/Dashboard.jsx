import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import FileUploader from '../components/FileUploader';
import Loader from '../components/Loader';
import { analyzeModel } from '../api/client';

export default function Dashboard({ setAnalysisResult, setDatasetFile }) {
  const [mode, setMode] = useState('model'); // 'model' | 'metrics'
  const [modelFile, setModelFile] = useState(null);
  const [metricsFile, setMetricsFile] = useState(null);
  const [datasetFileLocal, setDatasetFileLocal] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    setError('');

    if (!datasetFileLocal) {
      setError('Please upload a dataset file (.csv)');
      return;
    }
    if (mode === 'model' && !modelFile) {
      setError('Please upload a model file (.pkl or .joblib)');
      return;
    }
    if (mode === 'metrics' && !metricsFile) {
      setError('Please upload a metrics file (.json)');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('dataset_file', datasetFileLocal);

      if (mode === 'model') {
        formData.append('model_file', modelFile);
      } else {
        formData.append('metrics_file', metricsFile);
      }

      if (targetColumn.trim()) {
        formData.append('target_column', targetColumn.trim());
      }

      const result = await analyzeModel(formData);
      setAnalysisResult(result);
      setDatasetFile(datasetFileLocal);
      navigate('/results');
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Analysis failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Loader text="Analyzing your model... This may take a moment." />;
  }

  return (
    <div className="page-container" id="dashboard-page">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1>🩺 Model <span style={{ background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Doctor</span> AI+</h1>
        <p>Upload your ML model & dataset to diagnose issues, get suggestions, and improve performance automatically.</p>
      </motion.div>

      {/* Mode Selector */}
      <div className="tabs" id="mode-tabs">
        <button
          className={`tab ${mode === 'model' ? 'active' : ''}`}
          onClick={() => setMode('model')}
        >
          🤖 Mode A: Model + Dataset
        </button>
        <button
          className={`tab ${mode === 'metrics' ? 'active' : ''}`}
          onClick={() => setMode('metrics')}
        >
          📊 Mode B: Metrics + Dataset
        </button>
      </div>

      {/* Upload Section */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
        {mode === 'model' ? (
          <FileUploader
            label="Upload Model (.pkl / .joblib)"
            accept={{ 'application/octet-stream': ['.pkl', '.joblib'] }}
            file={modelFile}
            onFileSelect={setModelFile}
            icon="🤖"
            hint="Scikit-learn model (.pkl or .joblib file)"
          />
        ) : (
          <FileUploader
            label="Upload Metrics (.json)"
            accept={{ 'application/json': ['.json'] }}
            file={metricsFile}
            onFileSelect={setMetricsFile}
            icon="📋"
            hint="JSON with accuracy, precision, recall, f1_score"
          />
        )}

        <FileUploader
          label="Upload Dataset (.csv)"
          accept={{ 'text/csv': ['.csv'] }}
          file={datasetFileLocal}
          onFileSelect={setDatasetFileLocal}
          icon="📊"
          hint="CSV dataset with features and target column"
        />
      </div>

      {/* Target Column Input */}
      <motion.div
        className="glass-card"
        style={{ marginBottom: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label">Target Column (optional — auto-detected if empty)</label>
          <input
            type="text"
            className="form-input"
            placeholder="e.g. target, label, class, survived..."
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            id="target-column-input"
          />
        </div>
      </motion.div>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            padding: 'var(--space-md)',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--accent-red)',
            marginBottom: 'var(--space-lg)',
            fontSize: '0.9rem',
          }}
          id="error-message"
        >
          ❌ {error}
        </motion.div>
      )}

      {/* Analyze Button */}
      <motion.button
        className="btn btn-primary"
        onClick={handleAnalyze}
        disabled={loading}
        style={{ width: '100%', padding: '1rem', fontSize: '1.1rem' }}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        id="analyze-button"
      >
        🔬 Analyze Model
      </motion.button>

      {/* Feature Cards */}
      <div className="grid-3" style={{ marginTop: 'var(--space-2xl)' }}>
        {[
          { icon: '🔍', title: 'Smart Diagnosis', desc: 'Detects overfitting, underfitting, imbalance & more' },
          { icon: '💡', title: 'AI Suggestions', desc: 'Rule-based + AI-powered improvement recommendations' },
          { icon: '⭐', title: 'Health Score', desc: 'Comprehensive 0-100 model health assessment' },
          { icon: '🔄', title: 'Auto Retrain', desc: 'One-click retraining with applied improvements' },
          { icon: '📊', title: 'Version Compare', desc: 'Side-by-side model version comparison' },
          { icon: '⬇️', title: 'Model Download', desc: 'Download the best performing model as .pkl' },
        ].map((f, i) => (
          <motion.div
            key={i}
            className="glass-card"
            style={{ textAlign: 'center', padding: 'var(--space-lg)' }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.08 }}
          >
            <div style={{ fontSize: '2rem', marginBottom: 'var(--space-sm)' }}>{f.icon}</div>
            <h4 style={{ marginBottom: 'var(--space-xs)' }}>{f.title}</h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{f.desc}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
