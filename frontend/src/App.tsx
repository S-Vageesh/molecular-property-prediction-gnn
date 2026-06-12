import { useState } from 'react';
import axios from 'axios';
import './App.css';
import Header from './components/Header';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ModelInfo from './components/ModelInfo';

const API_BASE_URL = 'http://localhost:5000';

export interface PredictionData {
  smiles: string;
  solubility: number | null;
  imagePath: string | null;
  explanationImage: string | null;
}

function App() {
  const [smiles, setSmiles] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PredictionData>({
    smiles: '',
    solubility: null,
    imagePath: null,
    explanationImage: null,
  });

  const handlePredict = async () => {
    if (!smiles) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, { smiles });
      setData({
        ...data,
        smiles: response.data.smiles,
        solubility: response.data.predicted_solubility,
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!smiles) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, { smiles });
      setData({
        smiles: response.data.smiles,
        solubility: response.data.predicted_solubility,
        imagePath: `${API_BASE_URL}/${response.data.image_path}`,
        explanationImage: null,
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!smiles) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/explain`, { smiles });
      setData({
        smiles: response.data.smiles,
        solubility: response.data.prediction,
        imagePath: data.imagePath, // Keep existing if any
        explanationImage: `${API_BASE_URL}/${response.data.explanation_image}`,
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Explanation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main className="dashboard">
        <div className="column left">
          <InputSection 
            smiles={smiles} 
            setSmiles={setSmiles} 
            onPredict={handlePredict}
            onAnalyze={handleAnalyze}
            onExplain={handleExplain}
            loading={loading}
          />
          {error && <div className="error-message">{error}</div>}
        </div>
        <div className="column right">
          <ResultsSection data={data} loading={loading} />
          <ModelInfo />
        </div>
      </main>
    </div>
  );
}

export default App;
