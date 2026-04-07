import React from 'react';

export default function Loader({ text = 'Processing...' }) {
  return (
    <div className="loader-container" id="loader">
      <div className="loader-spinner"></div>
      <p className="loader-text">{text}</p>
    </div>
  );
}
