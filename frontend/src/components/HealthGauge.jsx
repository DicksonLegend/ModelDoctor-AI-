import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function HealthGauge({ score = 0, status = 'Unknown', breakdown = {} }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    let current = 0;
    const interval = setInterval(() => {
      current += 1;
      if (current >= score) {
        current = score;
        clearInterval(interval);
      }
      setAnimatedScore(current);
    }, 15);
    return () => clearInterval(interval);
  }, [score]);

  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (animatedScore / 100) * circumference;

  const getColor = () => {
    if (score >= 85) return 'var(--accent-green)';
    if (score >= 70) return 'var(--accent-cyan)';
    if (score >= 50) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  const getStatusClass = () => {
    if (score >= 85) return 'status-excellent';
    if (score >= 70) return 'status-good';
    if (score >= 50) return 'status-needs-tuning';
    return 'status-poor';
  };

  return (
    <motion.div
      className="glass-card health-gauge"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      id="health-gauge"
    >
      <h3 style={{ marginBottom: 'var(--space-md)' }}>⭐ Model Health Score</h3>

      <div className="gauge-circle">
        <svg width="180" height="180">
          {/* Background track */}
          <circle
            cx="90" cy="90" r={radius}
            fill="none"
            stroke="var(--border-glass)"
            strokeWidth="10"
          />
          {/* Score arc */}
          <circle
            cx="90" cy="90" r={radius}
            fill="none"
            stroke={getColor()}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
        </svg>
        <div className="gauge-score">
          <div className="score-value" style={{ color: getColor() }}>
            {animatedScore}
          </div>
          <div className="score-label">/ 100</div>
        </div>
      </div>

      <div className={`gauge-status ${getStatusClass()}`}>
        {status}
      </div>

      {/* Breakdown */}
      {Object.keys(breakdown).length > 0 && (
        <div style={{ width: '100%', marginTop: 'var(--space-lg)' }}>
          {Object.entries(breakdown).map(([key, val]) => (
            <div key={key} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '0.4rem 0', borderBottom: '1px solid var(--border-glass)',
              fontSize: '0.85rem',
            }}>
              <span style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                {key.replace(/_/g, ' ')}
              </span>
              <span style={{ fontWeight: 600, color: val >= 70 ? 'var(--accent-green)' : val >= 50 ? 'var(--accent-orange)' : 'var(--accent-red)' }}>
                {val}/100
              </span>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
