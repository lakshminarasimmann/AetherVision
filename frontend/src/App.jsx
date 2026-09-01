import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Scan, 
  Upload, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  Layers, 
  Cpu, 
  Flame, 
  History as HistoryIcon, 
  ExternalLink, 
  Github, 
  Sparkles, 
  RefreshCw, 
  Check, 
  FileText,
  Sliders,
  ShieldCheck,
  Eye
} from 'lucide-react';
import './index.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://image-quality-defect-detection-760o.onrender.com';

export default function App() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('studio');

  // Check backend health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/v1/health`, { timeout: 4000 });
        if (res.data?.status === 'ok') {
          setApiStatus('online');
        } else {
          setApiStatus('offline');
        }
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/v1/history?limit=15`);
      setHistory(res.data || []);
      setShowHistory(true);
    } catch {
      setError('Could not fetch historical diagnostics from the database.');
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
    }
  };

  const processFiles = (selectedFiles) => {
    const valid = selectedFiles.filter(f => f.type.startsWith('image/'));
    if (valid.length === 0) {
      setError('Please upload valid image files (JPG, PNG, WebP).');
      return;
    }
    setError(null);
    setFiles(valid);
    setPreviews(valid.map(f => URL.createObjectURL(f)));
    setResults(null);
  };

  // Demo sample generator for quick user testing
  const loadDemoImage = async (type = 'clean') => {
    setError(null);
    setLoading(true);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');

      if (type === 'clean') {
        // High contrast clean geometric graphic
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(256, 256, 140, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 36px sans-serif';
        ctx.fillText('AetherVision HD', 120, 270);
      } else {
        // Blurry low-contrast artifact image
        ctx.fillStyle = '#222222';
        ctx.fillRect(0, 0, 512, 512);
        ctx.filter = 'blur(14px)';
        ctx.fillStyle = '#555555';
        ctx.beginPath();
        ctx.arc(256, 256, 120, 0, Math.PI * 2);
        ctx.fill();
      }

      canvas.toBlob((blob) => {
        const demoFile = new File([blob], `demo_${type}_sample.jpg`, { type: 'image/jpeg' });
        processFiles([demoFile]);
        setLoading(false);
      }, 'image/jpeg');
    } catch {
      setError('Could not initialize demo sample image.');
      setLoading(false);
    }
  };

  const analyzeImages = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();

    try {
      if (files.length === 1) {
        formData.append('file', files[0]);
        const res = await axios.post(`${API_BASE_URL}/api/v1/analyze`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults([{ ...res.data, filename: files[0].name }]);
      } else {
        files.forEach(f => formData.append('files', f));
        const res = await axios.post(`${API_BASE_URL}/api/v1/analyze-batch`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResults(res.data.batch_results || []);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during image quality diagnostics.');
    } finally {
      setLoading(false);
    }
  };

  const getLabelTheme = (label) => {
    if (label === 'ACCEPTABLE') return { color: '#10b981', class: 'emerald', text: 'Acceptable Quality' };
    if (label === 'DEGRADED') return { color: '#f59e0b', class: 'amber', text: 'Degraded Quality' };
    return { color: '#f43f5e', class: 'rose', text: 'Defective Image' };
  };

  const getSeverityBadge = (sev) => {
    if (sev === 'high') return <span className="pill-badge rose">HIGH SEVERITY</span>;
    if (sev === 'medium') return <span className="pill-badge amber">MEDIUM SEVERITY</span>;
    return <span className="pill-badge emerald">LOW SEVERITY</span>;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* ────────────────── TOP NAVIGATION ────────────────── */}
      <header style={{
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(8, 9, 13, 0.85)',
        backdropFilter: 'blur(20px)',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{
          maxWidth: '1280px',
          margin: '0 auto',
          padding: '16px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          {/* Logo & Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #10b981 0%, #6366f1 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 20px rgba(16, 185, 129, 0.35)'
            }}>
              <Scan size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                  AetherVision
                </span>
                <span style={{ 
                  background: 'rgba(16, 185, 129, 0.15)', 
                  color: '#10b981', 
                  fontSize: '0.7rem', 
                  fontWeight: 700, 
                  padding: '2px 6px', 
                  borderRadius: '6px',
                  border: '1px solid rgba(16, 185, 129, 0.3)'
                }}>
                  AI 2.0
                </span>
              </div>
              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Image Quality & Defect Diagnostics</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <button 
              className="btn-secondary" 
              onClick={() => { setActiveTab('studio'); setShowHistory(false); }}
              style={{ borderColor: activeTab === 'studio' && !showHistory ? 'var(--accent-emerald)' : undefined }}
            >
              <Cpu size={16} /> Studio
            </button>
            <button 
              className="btn-secondary" 
              onClick={() => { setActiveTab('workflow'); setShowHistory(false); }}
              style={{ borderColor: activeTab === 'workflow' ? 'var(--accent-emerald)' : undefined }}
            >
              <Layers size={16} /> Architecture
            </button>
            <button 
              className="btn-secondary" 
              onClick={fetchHistory}
              style={{ borderColor: showHistory ? 'var(--accent-emerald)' : undefined }}
            >
              <HistoryIcon size={16} /> History
            </button>
            <a 
              href={`${API_BASE_URL}/docs`} 
              target="_blank" 
              rel="noreferrer" 
              className="btn-secondary" 
              style={{ textDecoration: 'none' }}
            >
              <FileText size={16} /> API Docs <ExternalLink size={12} />
            </a>
            <a 
              href="https://github.com/lakshminarasimmann/mage-quality-defect-detection" 
              target="_blank" 
              rel="noreferrer" 
              className="btn-secondary"
              style={{ textDecoration: 'none' }}
            >
              <Github size={16} /> GitHub
            </a>

            {/* Live Status indicator */}
            <div className={`pill-badge ${apiStatus === 'online' ? 'emerald' : 'rose'}`} style={{ fontSize: '0.75rem' }}>
              <div style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                background: apiStatus === 'online' ? '#10b981' : '#f43f5e',
                boxShadow: apiStatus === 'online' ? '0 0 8px #10b981' : '0 0 8px #f43f5e'
              }} />
              {apiStatus === 'online' ? 'API Online' : 'API Connecting'}
            </div>
          </nav>
        </div>
      </header>

      {/* ────────────────── MAIN CONTENT ────────────────── */}
      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '48px 24px', flex: 1, width: '100%' }}>

        {/* ── HERO BANNER ── */}
        <section style={{ textAlign: 'center', marginBottom: '56px' }} className="animate-fade-in">
          <div style={{ display: 'inline-flex', marginBottom: '16px' }}>
            <span className="pill-badge emerald">
              <Sparkles size={14} /> Powered by Hybrid OpenCV & PyTorch Architecture
            </span>
          </div>
          <h1 className="gradient-title" style={{ fontSize: 'clamp(2.4rem, 5vw, 3.8rem)', lineHeight: 1.15, marginBottom: '18px' }}>
            Precision Image Quality &<br />Defect Localization Engine
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.12rem', maxWidth: '720px', margin: '0 auto 28px auto' }}>
            Autonomous multi-factor quality scoring, sub-pixel defect classification, and 16x16 localized spatial degradation heatmaps—engineered for mission-critical visual pipelines.
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', flexWrap: 'wrap' }}>
            <button className="btn-primary" onClick={() => { setActiveTab('studio'); setShowHistory(false); }}>
              <Scan size={18} /> Open Analyzer Studio
            </button>
            <button className="btn-secondary" onClick={() => loadDemoImage('clean')}>
              <CheckCircle2 size={16} color="#10b981" /> Load Demo Clean Image
            </button>
            <button className="btn-secondary" onClick={() => loadDemoImage('blur')}>
              <AlertTriangle size={16} color="#f59e0b" /> Load Demo Defective Image
            </button>
          </div>
        </section>

        {/* ── CONDITIONAL SECTION: HISTORY VIEW ── */}
        {showHistory ? (
          <section className="glass-panel animate-fade-in" style={{ marginBottom: '40px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <div>
                <h2>Historical Analysis Records</h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Recent diagnostic evaluations stored in the transactional SQLite database.</p>
              </div>
              <button className="btn-secondary" onClick={fetchHistory}><RefreshCw size={15} /> Refresh</button>
            </div>

            {history.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px' }}>No previous analyses recorded in database yet.</p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.92rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '12px' }}>ID</th>
                      <th style={{ padding: '12px' }}>Filename</th>
                      <th style={{ padding: '12px' }}>Quality Score</th>
                      <th style={{ padding: '12px' }}>Diagnostic Label</th>
                      <th style={{ padding: '12px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h) => {
                      const theme = getLabelTheme(h.quality_label);
                      return (
                        <tr key={h.id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                          <td style={{ padding: '12px', color: 'var(--text-muted)' }}>#{h.id}</td>
                          <td style={{ padding: '12px', fontWeight: 600 }}>{h.filename}</td>
                          <td style={{ padding: '12px', color: theme.color, fontWeight: 700 }}>{h.quality_score}/100</td>
                          <td style={{ padding: '12px' }}>
                            <span className={`pill-badge ${theme.class}`}>{h.quality_label}</span>
                          </td>
                          <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                            {new Date(h.timestamp).toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        ) : null}

        {/* ── CONDITIONAL SECTION: ARCHITECTURE SHOWCASE ── */}
        {activeTab === 'workflow' && !showHistory ? (
          <section className="animate-fade-in" style={{ marginBottom: '40px' }}>
            <div className="glass-panel" style={{ marginBottom: '32px' }}>
              <h2 style={{ marginBottom: '12px' }}>Theoretical Processing Workflow</h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                How AetherVision combines deterministic feature engineering with PyTorch multi-task neural regression:
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
                <div style={{ background: 'var(--bg-input)', padding: '20px', borderRadius: '14px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ color: 'var(--accent-emerald)', marginBottom: '10px' }}><Cpu size={28} /></div>
                  <h4 style={{ marginBottom: '6px' }}>1. CV Extraction</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Extracts 6 statistical features: Laplacian variance (sharpness), HSV luminance, contrast, high-frequency noise difference, saturation, and histogram clipping.
                  </p>
                </div>

                <div style={{ background: 'var(--bg-input)', padding: '20px', borderRadius: '14px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ color: 'var(--accent-cyan)', marginBottom: '10px' }}><Flame size={28} /></div>
                  <h4 style={{ marginBottom: '6px' }}>2. Thermal Heatmapping</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Subdivides image into a 16x16 grid of 32x32 patches. Calculates localized edge response and applies inverted JET colormap alpha blending.
                  </p>
                </div>

                <div style={{ background: 'var(--bg-input)', padding: '20px', borderRadius: '14px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ color: 'var(--accent-violet)', marginBottom: '10px' }}><Layers size={28} /></div>
                  <h4 style={{ marginBottom: '6px' }}>3. PyTorch Neural MLP</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Multi-task neural network with dual output heads: continuous overall quality score regression + 5-channel multi-label defect probabilities.
                  </p>
                </div>

                <div style={{ background: 'var(--bg-input)', padding: '20px', borderRadius: '14px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ color: 'var(--accent-amber)', marginBottom: '10px' }}><ShieldCheck size={28} /></div>
                  <h4 style={{ marginBottom: '6px' }}>4. ACID Persistence</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Transactional logging with SQLAlchemy SQLite ORM capturing full JSON statistics, defect tags, and audit timestamps.
                  </p>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {/* ── STUDIO WORKSPACE (DROPZONE & ANALYZER) ── */}
        {activeTab === 'studio' && !showHistory ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            
            {/* Upload Zone Panel */}
            <section className="glass-panel animate-fade-in">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <h2 style={{ fontSize: '1.4rem' }}>Image Input Studio</h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>Upload single images or multi-file batches for concurrent quality analysis.</p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn-secondary" onClick={() => loadDemoImage('clean')}>Demo: Sharp</button>
                  <button className="btn-secondary" onClick={() => loadDemoImage('blur')}>Demo: Defective</button>
                </div>
              </div>

              {/* Drag and Drop Container */}
              <div 
                className="dropzone-container"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input-element').click()}
              >
                <input 
                  id="file-input-element" 
                  type="file" 
                  accept="image/*" 
                  multiple 
                  style={{ display: 'none' }} 
                  onChange={handleFileInput}
                />

                {previews.length > 0 ? (
                  <div>
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '16px' }}>
                      {previews.map((src, i) => (
                        <div key={i} style={{ position: 'relative', width: '90px', height: '90px', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                          <img src={src} alt="Upload Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        </div>
                      ))}
                    </div>
                    <p style={{ fontWeight: 600, color: 'var(--accent-emerald)' }}>
                      {files.length} Image(s) Selected — Click or Drag to replace
                    </p>
                  </div>
                ) : (
                  <div>
                    <div style={{
                      width: '64px',
                      height: '64px',
                      borderRadius: '16px',
                      background: 'rgba(16, 185, 129, 0.1)',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 16px auto',
                      color: 'var(--accent-emerald)'
                    }}>
                      <Upload size={28} />
                    </div>
                    <h3 style={{ fontSize: '1.15rem', marginBottom: '6px' }}>Drag & Drop Image Files Here</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>Supports JPG, PNG, WebP format with multi-file batch triage</p>
                  </div>
                )}
              </div>

              {/* Error Alert */}
              {error && (
                <div style={{
                  marginTop: '18px',
                  padding: '12px 16px',
                  borderRadius: '10px',
                  background: 'rgba(244, 63, 94, 0.1)',
                  border: '1px solid rgba(244, 63, 94, 0.3)',
                  color: '#fb7185',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  fontSize: '0.9rem'
                }}>
                  <AlertTriangle size={18} /> {error}
                </div>
              )}

              {/* Action Trigger Button */}
              <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end' }}>
                <button 
                  className="btn-primary" 
                  onClick={analyzeImages} 
                  disabled={files.length === 0 || loading}
                  style={{ minWidth: '220px' }}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={18} className="spin-animation" /> Running AI Diagnostics...
                    </>
                  ) : (
                    <>
                      <Activity size={18} /> Analyze {files.length > 0 ? `${files.length} Image(s)` : ''}
                    </>
                  )}
                </button>
              </div>
            </section>

            {/* ── RESULTS DASHBOARD ── */}
            {results && results.map((result, idx) => {
              const theme = getLabelTheme(result.quality_label);
              return (
                <section key={idx} className="glass-panel animate-fade-in" style={{ borderColor: theme.color + '40' }}>
                  
                  {/* Result Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '18px', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
                    <div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Analysis Diagnostics</span>
                      <h2 style={{ fontSize: '1.5rem', marginTop: '2px' }}>{result.filename || 'Processed Image'}</h2>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className={`pill-badge ${theme.class}`} style={{ fontSize: '0.95rem', padding: '8px 18px' }}>
                        {theme.text}
                      </span>
                    </div>
                  </div>

                  {/* Main Grid: Gauge + Heatmap + Diagnostics */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '28px', marginBottom: '32px' }}>
                    
                    {/* Column 1: Overall Quality Circular Score */}
                    <div style={{ background: 'var(--bg-input)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                      <h4 style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>Overall Quality Score</h4>
                      
                      <svg viewBox="0 0 36 36" className="circular-chart">
                        <path 
                          className="circle-bg" 
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                        />
                        <path 
                          className="circle" 
                          stroke={theme.color}
                          strokeDasharray={`${result.quality_score}, 100`} 
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" 
                        />
                        <text x="18" y="20.5" className="percentage-text">{result.quality_score}</text>
                      </svg>

                      <div style={{ marginTop: '16px' }}>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Status Evaluation: </span>
                        <strong style={{ color: theme.color }}>{result.quality_label}</strong>
                      </div>
                    </div>

                    {/* Column 2: Dual Heatmap & Original Visualizer */}
                    <div style={{ background: 'var(--bg-input)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <h4 style={{ color: 'var(--text-secondary)' }}>Spatial Degradation Heatmap</h4>
                        <span className="pill-badge amber" style={{ fontSize: '0.7rem' }}>16x16 Grid</span>
                      </div>

                      <div style={{ width: '100%', height: '210px', borderRadius: '10px', overflow: 'hidden', background: '#000', position: 'relative' }}>
                        {result.heatmap ? (
                          <img src={result.heatmap} alt="Degradation Heatmap" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : previews[idx] ? (
                          <img src={previews[idx]} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : null}
                      </div>

                      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '10px' }}>
                        Thermal mapping: <span style={{ color: '#f43f5e', fontWeight: 600 }}>Red zones</span> identify severe edge attenuation, defocus blur, or noise artifacts.
                      </p>
                    </div>

                    {/* Column 3: Detected Defects & Severities */}
                    <div style={{ background: 'var(--bg-input)', padding: '24px', borderRadius: '16px', border: '1px solid var(--border-subtle)' }}>
                      <h4 style={{ color: 'var(--text-secondary)', marginBottom: '14px' }}>Detected Photographic Issues</h4>
                      
                      {result.issues && result.issues.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                          {result.issues.map((iss, i) => (
                            <div key={i} style={{ 
                              background: 'rgba(255, 255, 255, 0.03)', 
                              padding: '12px 14px', 
                              borderRadius: '10px',
                              border: '1px solid var(--border-subtle)',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center'
                            }}>
                              <div>
                                <strong style={{ textTransform: 'capitalize', fontSize: '0.95rem' }}>{iss.type}</strong>
                                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                  Confidence: {(iss.confidence * 100).toFixed(0)}%
                                </div>
                              </div>
                              {getSeverityBadge(iss.severity)}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ padding: '30px 10px', textAlign: 'center', color: '#10b981' }}>
                          <CheckCircle2 size={36} style={{ margin: '0 auto 10px auto' }} />
                          <p style={{ fontWeight: 600 }}>No critical defects detected</p>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Image meets standard clarity thresholds</span>
                        </div>
                      )}
                    </div>

                  </div>

                  {/* Feature Statistics Metrics Breakdown Bar Grid */}
                  <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '24px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                      <Sliders size={18} color="var(--accent-emerald)" />
                      <h3>Interpretable Computer Vision Metrics</h3>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
                      {Object.entries(result.stats || {}).map(([key, val]) => {
                        const numVal = typeof val === 'number' ? val : 0;
                        const percentage = Math.min(100, Math.max(0, numVal * 100));
                        return (
                          <div key={key} style={{ background: 'var(--bg-input)', padding: '14px 16px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.88rem' }}>
                              <span style={{ textTransform: 'capitalize', color: 'var(--text-secondary)' }}>
                                {key.replace('_', ' ')}
                              </span>
                              <strong style={{ fontFamily: 'monospace' }}>{numVal.toFixed(3)}</strong>
                            </div>
                            <div style={{ height: '6px', width: '100%', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                              <div style={{ 
                                height: '100%', 
                                width: `${percentage}%`, 
                                background: 'linear-gradient(90deg, #10b981, #6366f1)', 
                                borderRadius: '3px',
                                transition: 'width 0.8s ease'
                              }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                </section>
              );
            })}
          </div>
        ) : null}

      </main>

      {/* ────────────────── FOOTER ────────────────── */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        background: 'rgba(8, 9, 13, 0.95)',
        padding: '32px 24px',
        textAlign: 'center',
        marginTop: 'auto'
      }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
            AetherVision AI Platform — Autonomous Image Quality & Defect Localization
          </p>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Developed by{' '}
            <a 
              href="https://lakshminarasimman.vercel.app/" 
              target="_blank" 
              rel="noreferrer" 
              style={{ color: 'var(--accent-emerald)', textDecoration: 'none', fontWeight: 600 }}
            >
              lakshminarasimmann
            </a>
          </p>
        </div>
      </footer>

    </div>
  );
}
