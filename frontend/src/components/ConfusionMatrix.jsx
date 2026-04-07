import React from 'react';

export default function ConfusionMatrix({ matrix = [] }) {
  if (!matrix.length) return null;

  const maxVal = Math.max(...matrix.flat(), 1);

  const getColor = (val) => {
    const intensity = val / maxVal;
    if (intensity > 0.7) return `rgba(139, 92, 246, ${0.3 + intensity * 0.5})`;
    if (intensity > 0.3) return `rgba(6, 182, 212, ${0.2 + intensity * 0.4})`;
    return `rgba(255, 255, 255, ${0.03 + intensity * 0.1})`;
  };

  return (
    <div id="confusion-matrix" style={{ textAlign: 'center' }}>
      <h3 style={{ marginBottom: 'var(--space-md)' }}>📊 Confusion Matrix</h3>
      <div
        className="confusion-matrix"
        style={{ gridTemplateColumns: `repeat(${matrix[0]?.length || 2}, 1fr)` }}
      >
        {matrix.map((row, i) =>
          row.map((val, j) => (
            <div
              key={`${i}-${j}`}
              className="confusion-cell"
              style={{ backgroundColor: getColor(val) }}
              title={`Actual: ${i}, Predicted: ${j}`}
            >
              {val}
            </div>
          ))
        )}
      </div>
      <div style={{ marginTop: 'var(--space-sm)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        Rows: Actual · Columns: Predicted
      </div>
    </div>
  );
}
