import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import MetricsCard from '../components/MetricsCard';
import DiagnosisPanel from '../components/DiagnosisPanel';
import SuggestionPanel from '../components/SuggestionPanel';
import HealthGauge from '../components/HealthGauge';
import ConfusionMatrix from '../components/ConfusionMatrix';
import Loader from '../components/Loader';
import { retrainModel } from '../api/client';

export default function AnalysisResults({ result, datasetFile, setAnalysisResult }) {
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState(null);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  if (!result) {
    return (
      <div className="page-container">
        <div className="empty-state">
          <div className="empty-icon">🔬</div>
          <h2>No Analysis Results</h2>
          <p style={{ marginBottom: 'var(--space-lg)' }}>Upload a model and dataset to see results.</p>
          <button className="btn btn-primary" onClick={() => navigate('/')}>
            Go to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const { metrics, diagnosis, suggestions, health_score, model_version, class_distribution } = result;

  const handleRetrain = async () => {
    if (!datasetFile) {
      setError('Dataset file is required for retraining. Please go back and upload again.');
      return;
    }

    setRetraining(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('dataset_file', datasetFile);
      formData.append('model_version', model_version);

      const res = await retrainModel(formData);
      setRetrainResult(res);

      // Update the main analysis result with new version info
      setAnalysisResult(prev => ({
        ...prev,
        model_version: res.new_model_version,
        metrics: res.new_metrics,
        health_score: res.new_health_score,
      }));
    } catch (err) {
      setError(err.response?.data?.detail || 'Retraining failed');
    } finally {
      setRetraining(false);
    }
  };

  if (retraining) {
    return <Loader text="🔄 Retraining model with improvements... Please wait." />;
  }

  return (
    <div className="page-container" id="results-page">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>📋 Analysis Results</h1>
        <p>Model Version: <strong style={{ color: 'var(--accent-purple)' }}>{model_version}</strong></p>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
        <MetricsCard label="Accuracy" value={metrics.accuracy} delay={0} />
        <MetricsCard label="Precision" value={metrics.precision} delay={0.1} />
        <MetricsCard label="Recall" value={metrics.recall} delay={0.2} />
        <MetricsCard label="F1 Score" value={metrics.f1_score} delay={0.3} />
      </div>

      {/* Health Gauge + Confusion Matrix */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
        <HealthGauge
          score={health_score.score}
          status={health_score.status}
          breakdown={health_score.breakdown}
        />
        <div className="glass-card">
          <ConfusionMatrix matrix={metrics.confusion_matrix} />
          {/* Class Distribution */}
          {class_distribution && Object.keys(class_distribution).length > 0 && (
            <div style={{ marginTop: 'var(--space-lg)' }}>
              <h4 style={{ marginBottom: 'var(--space-sm)', color: 'var(--text-secondary)' }}>
                Class Distribution
              </h4>
              {Object.entries(class_distribution).map(([cls, count]) => (
                <div key={cls} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '0.3rem 0', fontSize: '0.9rem',
                }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Class {cls}</span>
                  <span style={{ fontWeight: 600 }}>{count} samples</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Diagnosis + Suggestions */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="glass-card">
          <DiagnosisPanel diagnosis={diagnosis} />
        </div>
        <div className="glass-card">
          <SuggestionPanel suggestions={suggestions} />
        </div>
      </div>

      {/* Retrain Result */}
      {retrainResult && (
        <motion.div
          className="glass-card pulse-glow"
          style={{ marginBottom: 'var(--space-xl)' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          id="retrain-result"
        >
          <h3 style={{ marginBottom: 'var(--space-md)' }}>
            🎉 Retraining Complete — Version {retrainResult.new_model_version}
          </h3>

          <div className="grid-4" style={{ marginBottom: 'var(--space-md)' }}>
            {['accuracy', 'precision', 'recall', 'f1_score'].map((key) => {
              const delta = retrainResult.improvements[`${key === 'f1_score' ? 'f1' : key}_delta`]
                ?? (retrainResult.new_metrics[key] - retrainResult.old_metrics[key]);
              return (
                <div key={key} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    {key.replace('_', ' ')}
                  </div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                    {(retrainResult.new_metrics[key] * 100).toFixed(1)}%
                  </div>
                  <div className={delta >= 0 ? 'improvement-positive' : 'improvement-negative'}>
                    {delta >= 0 ? '▲' : '▼'} {(Math.abs(delta) * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>

          {retrainResult.improvements?.actions_applied && (
            <div style={{ marginTop: 'var(--space-sm)' }}>
              <strong style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Actions Applied:
              </strong>
              <ul style={{ marginTop: '0.25rem', paddingLeft: '1.2rem', color: 'var(--accent-cyan)', fontSize: '0.9rem' }}>
                {retrainResult.improvements.actions_applied.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: 'var(--space-md)',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--accent-red)',
          marginBottom: 'var(--space-lg)',
        }}>
          ❌ {error}
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
        <motion.button
          className="btn btn-success"
          onClick={handleRetrain}
          disabled={retraining || !datasetFile}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          id="retrain-button"
        >
          🔄 Retrain with Improvements
        </motion.button>

        <button
          className="btn btn-secondary"
          onClick={() => navigate('/compare')}
        >
          📊 Compare Versions
        </button>

        <button
          className="btn btn-secondary"
          onClick={() => navigate('/')}
        >
          🔙 New Analysis
        </button>
      </div>
    </div>
  );
}
