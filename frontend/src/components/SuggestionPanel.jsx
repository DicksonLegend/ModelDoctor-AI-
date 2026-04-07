import React from 'react';
import { motion } from 'framer-motion';

export default function SuggestionPanel({ suggestions = [] }) {
  if (!suggestions.length) return null;

  return (
    <div id="suggestion-panel">
      <h3 style={{ marginBottom: 'var(--space-md)' }}>💡 Improvement Suggestions</h3>
      {suggestions.map((s, i) => (
        <motion.div
          key={i}
          className="suggestion-card"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: i * 0.1 }}
        >
          <div className="suggestion-issue">⚠️ {s.issue}</div>
          <div className="suggestion-action">→ {s.action}</div>
          <div className="suggestion-explanation">{s.explanation}</div>
        </motion.div>
      ))}
    </div>
  );
}
