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

  const {
    task_type,
    metrics,
    diagnosis,
    suggestions,
    health_score,
    model_version,
    class_distribution,
    plain_language_summary,
    metrics_source,
  } = result;

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

      setAnalysisResult(prev => ({
        ...prev,
        task_type: res.task_type || prev?.task_type,
        model_version: res.new_model_version,
        metrics: res.new_metrics,
        health_score: res.new_health_score,
        plain_language_summary: res.plain_language_summary || prev?.plain_language_summary,
        metrics_source: res.metrics_source || prev?.metrics_source,
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

  const pct = (v) => v != null ? (v * 100).toFixed(1) + '%' : 'N/A';
  const val = (v) => v != null ? v : 'N/A';
  const isRegression = task_type === 'regression';

  const summaryMetrics = isRegression
    ? [
      { label: 'R2 Score', value: metrics.r2_score, suffix: '' },
      { label: 'RMSE', value: metrics.rmse, suffix: '' },
      { label: 'MAE', value: metrics.mae, suffix: '' },
      { label: 'Explained Var', value: metrics.explained_variance, suffix: '' },
    ]
    : [
      { label: 'Accuracy', value: metrics.accuracy, suffix: '' },
      { label: 'Precision', value: metrics.precision, suffix: '' },
      { label: 'Recall', value: metrics.recall, suffix: '' },
      { label: 'F1 Score', value: metrics.f1_score, suffix: '' },
    ];

  return (
    <div className="page-container" id="results-page">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>📋 Analysis Results</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Task Type: <strong>{isRegression ? 'Regression' : 'Classification'}</strong></p>
        <p>Model Version: <strong style={{ color: 'var(--accent-purple)' }}>{model_version}</strong></p>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
        {summaryMetrics.map((m, idx) => (
          <MetricsCard key={m.label} label={m.label} value={m.value} suffix={m.suffix} delay={idx * 0.1} />
        ))}
      </div>

      {/* Health Gauge + Confusion Matrix */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
        <HealthGauge
          score={health_score.score}
          status={health_score.status}
          breakdown={health_score.breakdown}
        />
        <div className="glass-card">
          {!isRegression && <ConfusionMatrix matrix={metrics.confusion_matrix} />}
          {isRegression && (
            <div style={{ textAlign: 'center', padding: 'var(--space-lg)' }}>
              <h3 style={{ marginBottom: 'var(--space-md)' }}>📉 Regression Error Summary</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                RMSE: <strong>{val(metrics.rmse)}</strong> · MAE: <strong>{val(metrics.mae)}</strong> · R2: <strong>{val(metrics.r2_score)}</strong>
              </p>
            </div>
          )}
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

      {/* Plain-language explanation */}
      {plain_language_summary && (
        <motion.div
          className="glass-card"
          style={{ marginBottom: 'var(--space-xl)' }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          id="plain-language-summary"
        >
          <h3 style={{ marginBottom: 'var(--space-sm)' }}>🧠 Easy Explanation (AI Assisted)</h3>
          <p style={{
            color: 'var(--text-secondary)',
            lineHeight: 1.8,
            whiteSpace: 'pre-line',
            marginBottom: 'var(--space-md)',
          }}>
            {plain_language_summary}
          </p>
          <div style={{
            fontSize: '0.82rem',
            color: 'var(--text-muted)',
            borderTop: '1px solid rgba(255,255,255,0.08)',
            paddingTop: 'var(--space-sm)',
          }}>
            Metrics Source: {metrics_source === 'model_evaluation_code'
              ? 'Computed by backend evaluation code (scikit-learn), not generated by Gemini.'
              : metrics_source || 'Computed by backend evaluation code.'}
          </div>
        </motion.div>
      )}

      {/* ───── Detailed Metrics Summary ───── */}
      <motion.div
        className="glass-card"
        style={{ marginBottom: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        id="detailed-metrics"
      >
        <h3 style={{ marginBottom: 'var(--space-md)' }}>📊 Detailed Metrics Summary</h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-md)' }}>
          {isRegression ? (
            <>
              <div className="metric-group">
                <h4 style={{ color: 'var(--accent-cyan)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Regression Metrics
                </h4>
                <MetricRow label="R2 Score" value={val(metrics.r2_score)} />
                <MetricRow label="Train R2" value={val(metrics.train_accuracy)} />
                <MetricRow label="Explained Variance" value={val(metrics.explained_variance)} />
              </div>

              <div className="metric-group">
                <h4 style={{ color: 'var(--accent-purple)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Error Metrics
                </h4>
                <MetricRow label="MAE" value={val(metrics.mae)} />
                <MetricRow label="MSE" value={val(metrics.mse)} />
                <MetricRow label="RMSE" value={val(metrics.rmse)} />
              </div>

              <div className="metric-group">
                <h4 style={{ color: 'var(--accent-green)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Dataset Info
                </h4>
                <MetricRow label="Test Samples" value={val(metrics.total_test_samples)} />
                <MetricRow label="Features" value={val(metrics.n_features)} />
              </div>
            </>
          ) : (
            <>
          {/* Core Metrics */}
          <div className="metric-group">
            <h4 style={{ color: 'var(--accent-cyan)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Core Metrics
            </h4>
            <MetricRow label="Test Accuracy" value={pct(metrics.accuracy)} />
            <MetricRow label="Train Accuracy" value={pct(metrics.train_accuracy)} />
            <MetricRow label="Error Rate" value={pct(metrics.error_rate)} />
            <MetricRow label="Misclassified" value={`${val(metrics.misclassified)} / ${val(metrics.total_test_samples)}`} />
          </div>

          {/* Weighted Metrics */}
          <div className="metric-group">
            <h4 style={{ color: 'var(--accent-purple)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Weighted Average
            </h4>
            <MetricRow label="Precision" value={pct(metrics.precision)} />
            <MetricRow label="Recall" value={pct(metrics.recall)} />
            <MetricRow label="F1 Score" value={pct(metrics.f1_score)} />
          </div>

          {/* Macro Metrics */}
          <div className="metric-group">
            <h4 style={{ color: 'var(--accent-green)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Macro Average
            </h4>
            <MetricRow label="Precision" value={pct(metrics.macro_precision)} />
            <MetricRow label="Recall" value={pct(metrics.macro_recall)} />
            <MetricRow label="F1 Score" value={pct(metrics.macro_f1)} />
          </div>

          {/* Dataset Info */}
          <div className="metric-group">
            <h4 style={{ color: 'var(--accent-orange)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Dataset Info
            </h4>
            <MetricRow label="Classes" value={val(metrics.n_classes)} />
            <MetricRow label="Test Samples" value={val(metrics.total_test_samples)} />
            <MetricRow label="Generalization Gap"
              value={metrics.train_accuracy != null && metrics.accuracy != null
                ? pct(Math.abs(metrics.train_accuracy - metrics.accuracy))
                : 'N/A'
              }
            />
          </div>
            </>
          )}
        </div>

        {/* Per-Class Breakdown */}
        {!isRegression && metrics.per_class && Object.keys(metrics.per_class).length > 0 && (
          <div style={{ marginTop: 'var(--space-lg)' }}>
            <h4 style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-sm)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Per-Class Breakdown
            </h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem',
              }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <th style={thStyle}>Class</th>
                    <th style={thStyle}>Precision</th>
                    <th style={thStyle}>Recall</th>
                    <th style={thStyle}>F1 Score</th>
                    <th style={thStyle}>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.per_class).map(([cls, m]) => (
                    <tr key={cls} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={tdStyle}>{cls}</td>
                      <td style={tdStyle}>{pct(m.precision)}</td>
                      <td style={tdStyle}>{pct(m.recall)}</td>
                      <td style={tdStyle}>{pct(m.f1_score)}</td>
                      <td style={tdStyle}>{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </motion.div>

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
            {(isRegression ? ['r2_score', 'rmse', 'mae'] : ['accuracy', 'precision', 'recall', 'f1_score']).map((key) => {
              const newVal = retrainResult.new_metrics[key] || 0;
              const oldVal = retrainResult.old_metrics[key] || 0;
              const delta = newVal - oldVal;
              const betterIfHigher = !isRegression || key === 'r2_score';
              const isPositive = betterIfHigher ? delta >= 0 : delta <= 0;
              const deltaLabel = isRegression
                ? `${delta >= 0 ? '+' : ''}${Number(delta).toFixed(4)}`
                : `${(Math.abs(delta) * 100).toFixed(1)}%`;
              return (
                <div key={key} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    {key.replace('_', ' ')}
                  </div>
                  <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                    {isRegression ? Number(newVal).toFixed(4) : `${(newVal * 100).toFixed(1)}%`}
                  </div>
                  <div className={isPositive ? 'improvement-positive' : 'improvement-negative'}>
                    {isPositive ? '▲' : '▼'} {deltaLabel}
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

/* ── Helper Components ── */

function MetricRow({ label, value }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.35rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)',
      fontSize: '0.88rem',
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
}

const thStyle = {
  textAlign: 'left', padding: '0.5rem 0.75rem',
  color: 'var(--text-muted)', fontWeight: 500,
};

const tdStyle = {
  padding: '0.5rem 0.75rem', color: 'var(--text-primary)',
  fontFamily: 'monospace',
};
