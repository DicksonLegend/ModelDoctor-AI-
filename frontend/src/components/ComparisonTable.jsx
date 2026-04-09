import React from 'react';
import { motion } from 'framer-motion';
import { getDownloadUrl } from '../api/client';

export default function ComparisonTable({ models = [] }) {
  if (!models.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📊</div>
        <p>No model versions to compare yet.<br />Analyze a model to get started.</p>
      </div>
    );
  }

  const sorted = [...models].sort((a, b) => {
    const vA = parseInt(a.version?.replace('v', '')) || 0;
    const vB = parseInt(b.version?.replace('v', '')) || 0;
    return vA - vB;
  });

  const latest = sorted[sorted.length - 1] || {};
  const isRegression = latest.task_type === 'regression' || latest.r2_score != null;

  const fmtPct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');
  const fmtNum = (v) => (typeof v === 'number' ? Number(v).toFixed(4) : '—');

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      id="comparison-table"
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Version</th>
              {isRegression ? (
                <>
                  <th>R2 Score</th>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>Explained Var</th>
                </>
              ) : (
                <>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                </>
              )}
              <th>Health</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <motion.tr
                key={m.version}
                className={m.is_best ? 'best-row' : ''}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
              >
                <td>
                  <strong>{m.version}</strong>
                  {m.is_best && <span className="best-badge" style={{ marginLeft: '0.5rem' }}>🏆 Best</span>}
                </td>
                {isRegression ? (
                  <>
                    <td>{fmtPct(m.r2_score)}</td>
                    <td>{fmtNum(m.rmse)}</td>
                    <td>{fmtNum(m.mae)}</td>
                    <td>{fmtPct(m.explained_variance)}</td>
                  </>
                ) : (
                  <>
                    <td>{fmtPct(m.accuracy)}</td>
                    <td>{fmtPct(m.precision)}</td>
                    <td>{fmtPct(m.recall)}</td>
                    <td>{fmtPct(m.f1_score)}</td>
                  </>
                )}
                <td>
                  <span style={{
                    fontWeight: 700,
                    color: m.health_score >= 70 ? 'var(--accent-green)' : m.health_score >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)',
                  }}>
                    {m.health_score}
                  </span>
                </td>
                <td style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  {m.created_at ? new Date(m.created_at).toLocaleDateString() : '—'}
                </td>
                <td>
                  <a
                    href={getDownloadUrl(m.version)}
                    className="btn btn-secondary"
                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                    download
                  >
                    ⬇ Download
                  </a>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
