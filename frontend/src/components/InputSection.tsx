interface InputSectionProps {
  smiles: string;
  setSmiles: (val: string) => void;
  onPredict: () => void;
  onAnalyze: () => void;
  onExplain: () => void;
  loading: boolean;
}

export default function InputSection({ 
  smiles, 
  setSmiles, 
  onPredict, 
  onAnalyze, 
  onExplain, 
  loading 
}: InputSectionProps) {
  return (
    <section className="card">
      <h2>Analyze Molecule</h2>
      <div className="input-group">
        <label className="info-label">SMILES String</label>
        <input 
          type="text" 
          placeholder="e.g. CCO, c1ccccc1" 
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          disabled={loading}
        />
        <div className="button-group">
          <button 
            className="btn-primary" 
            onClick={onPredict}
            disabled={loading || !smiles}
          >
            {loading ? <span className="loading-spinner"></span> : 'Predict Solubility'}
          </button>
          <button 
            className="btn-secondary" 
            onClick={onAnalyze}
            disabled={loading || !smiles}
          >
            Visualize Structure
          </button>
          <button 
            className="btn-accent" 
            onClick={onExplain}
            disabled={loading || !smiles}
          >
            Explain with GNNExplainer
          </button>
        </div>
      </div>
    </section>
  );
}
