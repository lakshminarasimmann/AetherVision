import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import './index.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://image-quality-defect-detection-760o.onrender.com';

function App() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null); // Array for batch, single object for single
  const [error, setError] = useState(null);

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleFiles = (selectedFiles) => {
    const validFiles = selectedFiles.filter(f => f.type.startsWith('image/'));
    if (validFiles.length === 0) {
      setError('Please select valid image files.');
      return;
    }
    setError(null);
    setFiles(validFiles);
    
    // Create previews
    const newPreviews = validFiles.map(f => URL.createObjectURL(f));
    setPreviews(newPreviews);
    setResults(null);
  };

  const analyzeImages = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    
    try {
      if (files.length === 1) {
        formData.append('file', files[0]);
        const response = await axios.post(`${API_BASE_URL}/api/v1/analyze`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults([response.data]); // Store as array for uniform rendering
      } else {
        files.forEach(f => formData.append('files', f));
        const response = await axios.post(`${API_BASE_URL}/api/v1/analyze-batch`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults(response.data.batch_results);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const getLabelColor = (label) => {
    if (label === 'ACCEPTABLE') return 'var(--success)';
    if (label === 'DEGRADED') return 'var(--warning)';
    return 'var(--danger)';
  };

  const getSeverityColor = (severity) => {
    if (severity === 'high') return 'var(--danger)';
    if (severity === 'medium') return 'var(--warning)';
    return 'var(--accent-cyan)';
  };

  const renderResultCard = (result, index) => {
    if (result.error) {
      return (
        <div key={index} className="glass-card" style={{ marginBottom: '20px', borderColor: 'var(--danger)' }}>
          <h3 style={{ color: 'var(--danger)' }}><AlertTriangle /> Error analyzing {result.filename}</h3>
          <p>{result.error}</p>
        </div>
      );
    }

    return (
      <div key={index} className="glass-card animate-fade-in" style={{ animationDelay: `${index * 0.1}s`, marginBottom: '32px' }}>
        <h2 style={{ marginTop: 0, display: 'flex', justifyContent: 'space-between' }}>
          <span>{result.filename || 'Analysis Result'}</span>
          <span style={{ fontSize: '1rem', color: getLabelColor(result.quality_label), background: 'rgba(0,0,0,0.3)', padding: '4px 12px', borderRadius: '16px' }}>
            {result.quality_label} (Score: {result.quality_score})
          </span>
        </h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
          
          {/* Images Section */}
          <div>
            <h4 style={{ color: 'var(--text-secondary)', marginTop: 0 }}>Quality Heatmap</h4>
            <div style={{ position: 'relative', width: '100%', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
               {result.heatmap ? (
                 <img src={result.heatmap} alt="Quality Heatmap" style={{ width: '100%', display: 'block' }} />
               ) : (
                 <img src={previews[index]} alt="Original" style={{ width: '100%', display: 'block' }} />
               )}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
              Red regions indicate areas of high degradation (e.g. blur or noise).
            </p>
          </div>

          {/* Details Section */}
          <div>
            <h4 style={{ color: 'var(--text-secondary)', marginTop: 0 }}>Detected Issues</h4>
            {result.issues && result.issues.length > 0 ? (
              <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 24px 0' }}>
                {result.issues.map((issue, idx) => (
                  <li key={idx} style={{ 
                    background: 'rgba(0,0,0,0.3)', 
                    padding: '12px', 
                    borderRadius: '8px', 
                    marginBottom: '8px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong style={{ textTransform: 'capitalize' }}>{issue.type}</strong>
                      <span style={{ 
                        fontSize: '0.8rem', 
                        padding: '2px 8px', 
                        borderRadius: '12px',
                        background: getSeverityColor(issue.severity),
                        color: 'black',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {issue.severity}
                      </span>
                    </div>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Confidence: {(issue.confidence * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--success)', marginBottom: '24px' }}>
                <CheckCircle /> No significant issues detected
              </div>
            )}

            <h4 style={{ color: 'var(--text-secondary)', marginTop: 0 }}>Interpretable Statistics</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '12px' }}>
              {Object.entries(result.stats || {}).map(([key, value]) => (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '4px' }}>
                    <span style={{ textTransform: 'capitalize' }}>{key.replace('_', ' ')}</span>
                    <span>{typeof value === 'number' ? value.toFixed(3) : value}</span>
                  </div>
                  <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ 
                      height: '100%', 
                      background: 'var(--accent-purple)', 
                      width: `${Math.min(100, Math.max(0, value * 100))}%` 
                    }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '40px 20px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }} className="animate-fade-in">
        <h1 style={{ fontSize: '3rem', margin: '0 0 16px 0' }} className="gradient-text">
          AI Image Vision
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto' }}>
          Upload one or more images to automatically evaluate visual quality, generate heatmaps, and uncover potential defects.
        </p>
      </header>

      <main style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
        
        {/* Upload Section */}
        <section className="glass-card animate-fade-in">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: 0 }}>
            <Upload /> Image Input (Batch Supported)
          </h2>
          
          <div 
            onDragOver={handleDragOver} 
            onDrop={handleDrop}
            style={{
              border: '2px dashed rgba(255, 255, 255, 0.2)',
              borderRadius: '12px',
              padding: '40px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '20px',
              backgroundColor: 'rgba(0,0,0,0.2)',
              transition: 'background-color 0.3s'
            }}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input 
              id="file-upload" 
              type="file" 
              accept="image/*" 
              multiple
              style={{ display: 'none' }} 
              onChange={handleFileChange}
            />
            {previews.length > 0 ? (
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center' }}>
                {previews.map((src, i) => (
                  <img key={i} src={src} alt="Preview" style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '8px' }} />
                ))}
              </div>
            ) : (
              <div>
                <Upload size={48} style={{ color: 'var(--text-secondary)', marginBottom: '16px' }} />
                <p>Drag and drop images here, or click to browse</p>
              </div>
            )}
          </div>
          
          {error && (
            <div style={{ color: 'var(--danger)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} /> {error}
            </div>
          )}
          
          <button 
            className="btn" 
            style={{ width: '100%' }} 
            onClick={analyzeImages}
            disabled={files.length === 0 || loading}
          >
            {loading ? 'Analyzing...' : `Analyze ${files.length} Image(s)`}
            {!loading && <Activity size={18} />}
          </button>
        </section>

        {/* Results Section */}
        {results && (
          <section>
            {results.map((result, index) => renderResultCard(result, index))}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
