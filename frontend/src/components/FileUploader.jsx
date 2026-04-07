import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';

export default function FileUploader({ label, accept, file, onFileSelect, icon = '📁', hint }) {
  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) {
      onFileSelect(accepted[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept || {},
    multiple: false,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
        id={`upload-${label?.toLowerCase().replace(/\s+/g, '-') || 'file'}`}
      >
        <input {...getInputProps()} />
        <span className="upload-icon">{icon}</span>
        <h3>{label || 'Upload File'}</h3>
        <p>{hint || 'Drag & drop or click to browse'}</p>

        <AnimatePresence>
          {file && (
            <motion.div
              className="file-info"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              ✅ {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
