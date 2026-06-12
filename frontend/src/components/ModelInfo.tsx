export default function ModelInfo() {
  return (
    <section className="card">
      <h2>Model Specifications</h2>
      <div className="info-grid">
        <div className="info-item">
          <span className="info-label">Best Architecture</span>
          <span className="info-value">GIN (Graph Isomorphism)</span>
        </div>
        <div className="info-item">
          <span className="info-label">Dataset</span>
          <span className="info-value">ESOL (MoleculeNet)</span>
        </div>
        <div className="info-item">
          <span className="info-label">Test RMSE</span>
          <span className="info-value">0.8566</span>
        </div>
        <div className="info-item">
          <span className="info-label">Test MAE</span>
          <span className="info-value">0.6876</span>
        </div>
      </div>
    </section>
  );
}
