import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 min for large-dataset retraining
});

export async function analyzeModel(formData) {
  const response = await api.post('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function retrainModel(formData) {
  const response = await api.post('/retrain', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function predictModel(data) {
  const response = await api.post('/predict', data);
  return response.data;
}

export async function getModels() {
  const response = await api.get('/models');
  return response.data;
}

export function getDownloadUrl(version) {
  return `${API_BASE}/download_model?version=${version}`;
}

export default api;
