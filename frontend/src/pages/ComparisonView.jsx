import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import ComparisonTable from '../components/ComparisonTable';
import Loader from '../components/Loader';
import { getModels, getDownloadUrl } from '../api/client';

export default function ComparisonView() {
  const [models, setModels] = useState([]);
  const [bestVersion, setBestVersion] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const data = await getModels();
      setModels(data.models || []);
      setBestVersion(data.best_version);
    } catch (err) {
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <Loader text="Loading model versions..." />;

  // Prepare chart data
  const barData = models.map(m => ({
    version: m.version,
    Accuracy: +(m.accuracy * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    F1: +(m.f1_score * 100).toFixed(1),
    Health: m.health_score,
  }));

  // Radar data for the latest 2 models
  const latestModels = [...models].sort((a, b) => {
    const vA = parseInt(a.version?.replace('v', '')) || 0;
    const vB = parseInt(b.version?.replace('v', '')) || 0;
    return vB - vA;
  }).slice(0, 2);

  const radarData = ['accuracy', 'precision', 'recall', 'f1_score'].map(key => {
    const item = { metric: key.replace('_', ' ').toUpperCase() };
    latestModels.forEach((m, i) => {
      item[m.version] = +(m[key] * 100).toFixed(1);
    });
    return item;
  });

  return (
    <div className="page-container" id="comparison-page">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>📊 Model Comparison</h1>
        <p>
          Compare all model versions side by side.
          {bestVersion && (
            <> Best model: <strong style={{ color: 'var(--accent-green)' }}>{bestVersion}</strong></>
          )}
        </p>
      </motion.div>

      {error && (
        <div style={{
          padding: 'var(--space-md)',
          background: 'rgba(239, 68, 68, 0.1)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--accent-red)',
          marginBottom: 'var(--space-lg)',
        }}>
          {error}
        </div>
      )}

      {/* Best Model Download */}
      {bestVersion && (
        <motion.div
          className="glass-card pulse-glow"
          style={{
            marginBottom: 'var(--space-xl)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 'var(--space-md)',
          }}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          id="best-model-download"
        >
          <div>
            <h3>🏆 Best Model: {bestVersion}</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Automatically selected based on highest health score
            </p>
          </div>
          <a
            href={getDownloadUrl(bestVersion)}
            className="btn btn-success"
            download
          >
            ⬇️ Download Best Model (.pkl)
          </a>
        </motion.div>
      )}

      {/* Comparison Table */}
      <div className="glass-card" style={{ marginBottom: 'var(--space-xl)' }}>
        <h3 style={{ marginBottom: 'var(--space-lg)' }}>📋 All Versions</h3>
        <ComparisonTable models={models.map(m => ({ ...m, is_best: m.version === bestVersion }))} />
      </div>

      {/* Charts */}
      {models.length > 0 && (
        <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
          {/* Bar Chart */}
          <motion.div
            className="glass-card"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 style={{ marginBottom: 'var(--space-lg)' }}>📈 Performance Overview</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData}>
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
                <Bar dataKey="Accuracy" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Health" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Radar Chart */}
          {latestModels.length >= 2 && (
            <motion.div
              className="glass-card"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h3 style={{ marginBottom: 'var(--space-lg)' }}>🎯 Latest 2 Versions</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="metric" stroke="var(--text-muted)" fontSize={11} />
                  <PolarRadiusAxis domain={[0, 100]} stroke="var(--text-muted)" fontSize={10} />
                  <Radar
                    name={latestModels[0]?.version}
                    dataKey={latestModels[0]?.version}
                    stroke="#8b5cf6"
                    fill="#8b5cf6"
                    fillOpacity={0.2}
                  />
                  <Radar
                    name={latestModels[1]?.version}
                    dataKey={latestModels[1]?.version}
                    stroke="#06b6d4"
                    fill="#06b6d4"
                    fillOpacity={0.2}
                  />
                  <Legend />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border-glass)',
                      borderRadius: '8px',
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
