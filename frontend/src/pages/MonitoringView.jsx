import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import Loader from '../components/Loader';
import { getModels } from '../api/client';

export default function MonitoringView() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const data = await getModels();
      setModels(data.models || []);
    } catch {
      // silent fail
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <Loader text="Loading monitoring data..." />;

  // Trend data — model versions over time
  const trendData = [...models]
    .sort((a, b) => {
      const vA = parseInt(a.version?.replace('v', '')) || 0;
      const vB = parseInt(b.version?.replace('v', '')) || 0;
      return vA - vB;
    })
    .map(m => ({
      version: m.version,
      Accuracy: +(m.accuracy * 100).toFixed(1),
      Health: m.health_score,
      F1: +(m.f1_score * 100).toFixed(1),
    }));

  return (
    <div className="page-container" id="monitoring-page">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>📡 Monitoring Dashboard</h1>
        <p>Track model performance across versions and over time.</p>
      </motion.div>

      {/* Summary Cards */}
      <div className="grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
        <motion.div
          className="glass-card metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="metric-value">{models.length}</div>
          <div className="metric-label">Total Models</div>
        </motion.div>

        <motion.div
          className="glass-card metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="metric-value" style={{ fontSize: '2rem' }}>
            {models.length > 0
              ? `${(Math.max(...models.map(m => m.accuracy)) * 100).toFixed(1)}%`
              : '—'}
          </div>
          <div className="metric-label">Best Accuracy</div>
        </motion.div>

        <motion.div
          className="glass-card metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="metric-value" style={{ fontSize: '2rem' }}>
            {models.length > 0
              ? Math.max(...models.map(m => m.health_score))
              : '—'}
          </div>
          <div className="metric-label">Best Health Score</div>
        </motion.div>
      </div>

      {/* Performance Trend Chart */}
      {trendData.length > 0 ? (
        <motion.div
          className="glass-card"
          style={{ marginBottom: 'var(--space-xl)' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 style={{ marginBottom: 'var(--space-lg)' }}>📈 Performance Trend</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="version" stroke="var(--text-muted)" fontSize={12} />
              <YAxis stroke="var(--text-muted)" fontSize={12} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-glass)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend />
              <Line
                type="monotone" dataKey="Accuracy" stroke="#8b5cf6"
                strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 7 }}
              />
              <Line
                type="monotone" dataKey="F1" stroke="#06b6d4"
                strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 7 }}
              />
              <Line
                type="monotone" dataKey="Health" stroke="#10b981"
                strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      ) : (
        <div className="glass-card empty-state">
          <div className="empty-icon">📡</div>
          <p>No monitoring data yet. Analyze and retrain models to see trends.</p>
        </div>
      )}

      {/* Model History Log */}
      {models.length > 0 && (
        <motion.div
          className="glass-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3 style={{ marginBottom: 'var(--space-lg)' }}>🕐 Model History</h3>
          {[...models].reverse().map((m, i) => (
            <motion.div
              key={m.version}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: 'var(--space-md)',
                borderBottom: '1px solid var(--border-glass)',
                fontSize: '0.9rem',
              }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.05 * i }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
                <strong style={{ color: 'var(--accent-purple)' }}>{m.version}</strong>
                {m.is_best && <span className="best-badge">🏆 Best</span>}
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-xl)', color: 'var(--text-secondary)' }}>
                <span>Acc: <strong style={{ color: 'var(--text-primary)' }}>{(m.accuracy * 100).toFixed(1)}%</strong></span>
                <span>Health: <strong style={{
                  color: m.health_score >= 70 ? 'var(--accent-green)' : 'var(--accent-orange)',
                }}>{m.health_score}</strong></span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  {m.created_at ? new Date(m.created_at).toLocaleString() : ''}
                </span>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
