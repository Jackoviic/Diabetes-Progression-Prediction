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
      // Normalize: (val - mean) / std
      // Note: model.mean and model.std indices match features order
      const normalized = (rawVal - model.mean[i]) / model.std[i];
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

      {/* 1. Process Pipeline Section */}
      <section style={{ marginBottom: '4rem' }}>
        <div className="section-title">🚀 How It Works: The ML Pipeline</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
          {[
            { title: '1. Load Data', desc: 'Load the 442-patient dataset. Features: Age, BMI, BP, & 6 Blood Serum levels.' },
            { title: '2. Preprocessing', desc: 'Split data: 80% Training, 20% Testing. Apply Z-score Standardization.' },
            { title: '3. Architecture', desc: 'Initialize Linear Regression model with zero-weights.' },
            { title: '4. Training', desc: 'Minimize MSE using Batch Gradient Descent (10k iters) with auto-convergence test.' },
            { title: '5. Evaluation', desc: 'Validate model on unseen Test Set to ensure generalization.' }
          ].map((step, i) => (
            <div key={i} className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem', opacity: 0.5 }}>0{i + 1}</div>
              <h3 style={{ marginBottom: '0.5rem', color: 'var(--accent-primary)' }}>{step.title}</h3>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 2. Model Performance Section */}
      <section style={{ marginBottom: '4rem' }}>
        <div className="section-title">📈 Model Performance</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>

          {/* Cost History */}
          <div className="glass-panel" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '2rem', padding: '2rem' }}>
            <div style={{ flex: '1 1 500px' }}>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                <img src="/results/cost_history.png" alt="Cost History" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
            </div>
            <div style={{ flex: '1 1 300px' }}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--accent-primary)', fontSize: '1.5rem' }}>Gradient Descent Convergence</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.6' }}>
                The cost function (MSE) decreases as the model learns. We use <strong>Batch Gradient Descent</strong> with an automatic convergence check.
              </p>
              <ul style={{ color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.95rem' }}>
                <li>⚡ <strong>Learning Rate (&alpha;):</strong> 0.01</li>
                <li>🔄 <strong>Max Iterations:</strong> 10,000</li>
                <li>🛑 <strong>Convergence Threshold (&epsilon;):</strong> 1e-3</li>
              </ul>
            </div>
          </div>

          {/* Train Performance */}
          <div className="glass-panel" style={{ display: 'flex', flexWrap: 'wrap-reverse', alignItems: 'center', gap: '2rem', padding: '2rem' }}>
            <div style={{ flex: '1 1 300px' }}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--success)', fontSize: '1.5rem' }}>Training Set Fit</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.6' }}>
                The model minimizes error on the <strong>80% Training Split</strong>. The scatter plot shows predicted vs. actual values.
              </p>
              <ul style={{ color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.95rem' }}>
                <li>🎯 <strong>Goal:</strong> Points should align with the dashed line (Perfect Prediction).</li>
                <li>📉 <strong>Optimization:</strong> Parameters (&theta;) are tuned here.</li>
              </ul>
            </div>
            <div style={{ flex: '1 1 500px' }}>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                <img src="/results/train_performance.png" alt="Train Performance" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
            </div>
          </div>

          {/* Test Performance */}
          <div className="glass-panel" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '2rem', padding: '2rem' }}>
            <div style={{ flex: '1 1 500px' }}>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                <img src="/results/test_performance.png" alt="Test Performance" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
            </div>
            <div style={{ flex: '1 1 300px' }}>
              <h3 style={{ marginBottom: '1rem', color: 'var(--accent-secondary)', fontSize: '1.5rem' }}>Test Set Generalization</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.6' }}>
                Performance on the <strong>20% Test Split</strong> (unseen data). This validates that the model hasn't just memorized the training data.
              </p>
              <ul style={{ color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.95rem' }}>
                <li>🧪 <strong>Validation:</strong> Crucial for real-world reliability.</li>
                <li>📊 <strong>Visual Check:</strong> Distribution should resemble the training plot.</li>
              </ul>
            </div>
          </div>

        </div>
      </section>

      {/* 3. Interactive Dashboard Section */}
      <section id="dashboard">
        <div className="section-title">🔮 Interactive Dashboard</div>
        <p style={{ marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
          Select a patient from the list to view their clinical details and the model's prediction.
        </p>

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
                <div className="metric-card"><div className="metric-label">Age</div><div className="metric-value">{selectedPatient?.AGE}</div></div>
                <div className="metric-card"><div className="metric-label">Gender</div><div className="metric-value">{selectedPatient?.GENDER === 1 ? 'F' : 'M'}</div></div>
                <div className="metric-card"><div className="metric-label">BMI</div><div className="metric-value">{selectedPatient?.BMI}</div></div>
                <div className="metric-card"><div className="metric-label">BP (Mean)</div><div className="metric-value">{selectedPatient?.BP}</div></div>
                <div className="metric-card"><div className="metric-label">TC (S1)</div><div className="metric-value">{selectedPatient?.S1}</div></div>
                <div className="metric-card"><div className="metric-label">LDL (S2)</div><div className="metric-value">{selectedPatient?.S2}</div></div>
                <div className="metric-card"><div className="metric-label">HDL (S3)</div><div className="metric-value">{selectedPatient?.S3}</div></div>
                <div className="metric-card"><div className="metric-label">TCH (S4)</div><div className="metric-value">{selectedPatient?.S4}</div></div>
                <div className="metric-card"><div className="metric-label">LTG (S5)</div><div className="metric-value">{selectedPatient?.S5}</div></div>
                <div className="metric-card"><div className="metric-label">GLU (S6)</div><div className="metric-value">{selectedPatient?.S6}</div></div>
              </div>
            </div>

            {/* Prediction Result */}
            <div className="prediction-area">
              {/* Gauge Container */}
              <div style={{ position: 'relative', width: '200px', height: '200px', marginBottom: '2rem' }}>
                <div className="prediction-circle" style={{ animationDuration: '6s', margin: 0 }}>
                </div>
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center', width: '100%' }}>
                  <div className="pred-val">{prediction.toFixed(1)}</div>
                  <div className="pred-label">Predicted Progression</div>
                </div>
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
      </section>
    </main>
  );
}
