import React from 'react';
import { motion } from 'framer-motion';

export default function MetricsCard({ label, value, suffix = '', delay = 0 }) {
  const displayValue = typeof value === 'number'
    ? (value < 1 ? (value * 100).toFixed(1) : value.toFixed(1))
    : value;

  const displaySuffix = typeof value === 'number' && value < 1 ? '%' : suffix;

  return (
    <motion.div
      className="glass-card metric-card"
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, delay }}
      id={`metric-${label?.toLowerCase().replace(/\s+/g, '-')}`}
    >
      <div className="metric-value">
        {displayValue}{displaySuffix}
      </div>
      <div className="metric-label">{label}</div>
    </motion.div>
  );
}
