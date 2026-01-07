"use client";

import { useEffect, useState, useMemo } from 'react';

export default function Home() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [model, setModel] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    async function fetchData() {
      try {
        const [modelRes, dataRes] = await Promise.all([
          fetch('/model_params.json'),
          fetch('/diabetes_data.json')
        ]);
        const modelData = await modelRes.json();
        const dataset = await dataRes.json();

        setModel(modelData);
        setData(dataset);
        setLoading(false);
      } catch (e) {
        console.error("Failed to load data", e);
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const selectedPatient = data[selectedIdx];

  const prediction = useMemo(() => {
    if (!selectedPatient || !model) return 0;

    // Feature order strictly matches Python export
    // ["AGE", "GENDER", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
    const features = model.feature_names;
    const x = [1.0]; // Intercept

    features.forEach((f, i) => {
      const rawVal = selectedPatient[f];
      // Normalize: (val - mean) / range
      // Note: model.mean and model.range indices match features order
      const normalized = (rawVal - model.mean[i]) / model.range[i];
      x.push(normalized);
    });

    // Dot product with theta
    let y_pred = 0;
    x.forEach((val, i) => {
      y_pred += val * model.theta[i];
    });

    return y_pred;
  }, [selectedPatient, model]);

  if (loading) {
    return (
      <div className="container" style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center' }}>
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <main className="container">
      <header>
        <h1>Diabetes Progression Prediction</h1>
        <p className="subtitle">Multiple Linear Regression Model (Batch Gradient Descent)</p>
      </header>

      <div className="dashboard-grid">
        {/* Left Panel: Patient List */}
        <div className="glass-panel list-container">
          <div className="section-title">
            <span>📋</span> Dataset ({data.length} Patients)
          </div>
          <div className="patient-list">
            {data.map((p, idx) => (
              <div
                key={idx}
                className={`patient-item ${idx === selectedIdx ? 'active' : ''}`}
                onClick={() => setSelectedIdx(idx)}
              >
                <div className="patient-id">Patient #{idx + 1}</div>
                <div className="patient-meta">
                  <span>Age: {p.AGE}</span>
                  <span>BMI: {p.BMI}</span>
                  <span>Actual Y: {p.Y}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Panel: Analysis */}
        <div className="glass-panel analysis-panel">

          {/* Feature Metrics */}
          <div>
            <div className="section-title">📊 Clinical Features</div>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Age</div>
                <div className="metric-value">{selectedPatient?.AGE}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">BMI</div>
                <div className="metric-value">{selectedPatient?.BMI}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Blood Pressure</div>
                <div className="metric-value">{selectedPatient?.BP}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Cholesterol (S1)</div>
                <div className="metric-value">{selectedPatient?.S1}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">LDL (S2)</div>
                <div className="metric-value">{selectedPatient?.S2}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">HDL (S3)</div>
                <div className="metric-value">{selectedPatient?.S3}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Glucose (S6)</div>
                <div className="metric-value">{selectedPatient?.S6}</div>
              </div>
            </div>
          </div>

          {/* Prediction Result */}
          <div className="prediction-area">
            <div className="prediction-circle" style={{ animationDuration: '6s' }}>
            </div>
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -70%)', textAlign: 'center' }}>
              <div className="pred-val">{prediction.toFixed(1)}</div>
              <div className="pred-label">Predicted Progression</div>
            </div>

            <div className="comparison">
              <div className="comp-item">
                <div className="comp-label">ACTUAL VALUE</div>
                <div className="comp-val actual-val">{selectedPatient?.Y}</div>
              </div>
              <div className="comp-item">
                <div className="comp-label">ERROR (MSE Contrib.)</div>
                <div className="comp-val error-val">
                  {(prediction - selectedPatient?.Y).toFixed(1)}
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}
