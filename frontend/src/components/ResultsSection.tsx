import { PredictionData } from '../App';

interface ResultsSectionProps {
  data: PredictionData;
  loading: boolean;
}

export default function ResultsSection({ data, loading }: ResultsSectionProps) {
  return (
    <section className="column">
      {/* Prediction Card */}
      <div className="card">
        <h2>Prediction Results</h2>
        <div className="prediction-display">
          <p className="info-label">Predicted Solubility</p>
          <div className="prediction-value">
            {data.solubility !== null ? data.solubility.toFixed(4) : '--'}
            <span className="unit"> log(mol/L)</span>
          </div>
        </div>
      </div>

      {/* Visualizations Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="card">
          <h2>Structure</h2>
          <div className="image-container">
            {data.imagePath ? (
              <img src={data.imagePath} alt="Molecule Structure" />
            ) : (
              <span className="placeholder-text">No structure generated</span>
            )}
          </div>
        </div>

        <div className="card">
          <h2>Explainer</h2>
          <div className="image-container">
            {data.explanationImage ? (
              <img src={data.explanationImage} alt="GNN Explanation" />
            ) : (
              <span className="placeholder-text">No explanation generated</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
