import React from 'react';
import { motion } from 'framer-motion';

export default function DiagnosisPanel({ diagnosis = [] }) {
  if (!diagnosis.length) return null;

  return (
    <div id="diagnosis-panel">
      <h3 style={{ marginBottom: 'var(--space-md)' }}>🔍 Diagnosis Results</h3>
      {diagnosis.map((item, i) => (
        <motion.div
          key={i}
          className={`diagnosis-item severity-${item.severity}`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: i * 0.1 }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <strong>{item.problem}</strong>
              <span className={`severity-badge ${item.severity}`}>
                {item.severity}
              </span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              {item.reason}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
