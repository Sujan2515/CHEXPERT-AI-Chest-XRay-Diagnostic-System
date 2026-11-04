
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Globe2,
  HeartPulse,
  Menu,
  ShieldCheck,
  UploadCloud,
  X,
  Zap
} from "lucide-react";

import { useState } from "react";
import "./App.css";

// Local backend URL for development.
// We will change this to the deployed backend URL later.
const API_URL = "http://127.0.0.1:8000";

function App() {
  const [mobileMenu, setMobileMenu] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // --------------------------------------------------
  // HANDLE IMAGE UPLOAD
  // --------------------------------------------------

  const handleFile = (event) => {
    const file = event.target.files?.[0];

    setPrediction(null);
    setError("");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      setSelectedFile(null);
      setError("Please select a valid JPG or PNG image.");
      event.target.value = "";
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setSelectedFile(null);
      setError("Image size must be below 10 MB.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
  };

  // --------------------------------------------------
  // SEND IMAGE TO FASTAPI BACKEND
  // --------------------------------------------------

  const handlePrediction = async () => {
    if (!selectedFile) {
      setError("Please select a chest X-ray image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPrediction(data);

    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Please make sure your FastAPI server is running."
          : err.message
      );

    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // RESULT CARD COMPONENT
  // --------------------------------------------------

  const ResultCard = ({ title, result }) => (
    <article className="feature-card" style={{ marginBottom: "18px" }}>
      <h3>{title}</h3>

      <p>
        Prediction: <strong>{result.prediction}</strong>
      </p>

      <p>
        Confidence: <strong>{result.confidence}%</strong>
      </p>

      <div style={{ marginTop: "15px" }}>
        <strong>Class probabilities</strong>

        {Object.entries(result.probabilities).map(([label, value]) => (
          <div key={label} style={{ marginTop: "12px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "5px"
              }}
            >
              <span>{label}</span>
              <strong>{value}%</strong>
            </div>

            <div
              style={{
                width: "100%",
                height: "8px",
                background: "#263449",
                borderRadius: "10px",
                overflow: "hidden"
              }}
            >
              <div
                style={{
                  width: `${value}%`,
                  height: "100%",
                  background: "#55d6be",
                  borderRadius: "10px"
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </article>
  );

  return (
    <div className="app">

      {/* NAVIGATION */}

      <nav className="navbar">
        <a href="#home" className="brand">
          <span className="brand-icon">
            <HeartPulse size={25} />
          </span>

          <span>
            CHEXPERT<span className="brand-dot">.</span>
          </span>
        </a>

        <button
          className="menu-toggle"
          onClick={() => setMobileMenu(!mobileMenu)}
        >
          {mobileMenu ? <X /> : <Menu />}
        </button>

        <div className={`nav-links ${mobileMenu ? "show-menu" : ""}`}>
          <a href="#home" onClick={() => setMobileMenu(false)}>
            Home
          </a>

          <a href="#technology" onClick={() => setMobileMenu(false)}>
            Technology
          </a>

          <a href="#research" onClick={() => setMobileMenu(false)}>
            Research
          </a>

          <a href="#about" onClick={() => setMobileMenu(false)}>
            About
          </a>

          <a
            href="#demo"
            className="nav-button"
            onClick={() => setMobileMenu(false)}
          >
            Explore Demo <ArrowRight size={16} />
          </a>
        </div>
      </nav>

      <main>

        {/* HERO SECTION */}

        <section className="hero" id="home">
          <div className="hero-content">

            <div className="eyebrow">
              <span className="pulse-dot"></span>
              AI-POWERED CHEST X-RAY RESEARCH
            </div>

            <h1>
              Bringing intelligence
              <br />
              to <span>chest imaging.</span>
            </h1>

            <p className="hero-description">
              Exploring AI-powered chest X-ray analysis through deep learning,
              explainable artificial intelligence and intelligent report generation.
            </p>

            <div className="hero-actions">
              <a href="#demo" className="primary-button">
                Explore AI Demo <ArrowRight size={18} />
              </a>

              <a href="#technology" className="secondary-button">
                Discover Technology
              </a>
            </div>

            <div className="hero-note">
              <ShieldCheck size={17} />
              Research prototype · Not for clinical diagnosis
            </div>

          </div>

          <div className="hero-visual">
            <div className="visual-glow"></div>

            <div className="scan-card">

              <div className="scan-top">
                <span>
                  <Activity size={16} /> CHEST IMAGING
                </span>

                <span className="live-label">
                  AI RESEARCH
                </span>
              </div>

              <div className="xray-placeholder">
                <HeartPulse size={100} strokeWidth={0.8} />
                <span>CHEST X-RAY</span>
              </div>

              <div className="scan-bottom">
                <span>Deep learning analysis</span>

                <span>
                  <CheckCircle2 size={15} /> Prototype
                </span>
              </div>

            </div>

            <div className="floating-tag tag-one">
              <BrainCircuit size={19} /> DenseNet-121
            </div>

            <div className="floating-tag tag-two">
              <Zap size={19} /> Explainable AI
            </div>
          </div>
        </section>

        {/* STATS SECTION */}

        <section className="stats-strip">
          <div>
            <strong>DenseNet-121</strong>
            <span>Deep learning backbone</span>
          </div>

          <div>
            <strong>2 Models</strong>
            <span>Pneumonia and TB</span>
          </div>

          <div>
            <strong>224 × 224</strong>
            <span>Input image resolution</span>
          </div>

          <div>
            <strong>Research</strong>
            <span>Experimental prototype</span>
          </div>
        </section>

        {/* TECHNOLOGY SECTION */}

        <section className="section" id="technology">

          <div className="section-heading">
            <span className="section-label">
              OUR TECHNOLOGY
            </span>

            <h2>Intelligence behind the image.</h2>

            <p>
              A research-oriented pipeline combining computer vision and
              artificial intelligence for chest X-ray analysis.
            </p>
          </div>

          <div className="feature-grid">

            <article className="feature-card">
              <div className="feature-icon">
                <BrainCircuit />
              </div>

              <h3>Deep Learning</h3>

              <p>
                DenseNet-121 transfer learning models for experimental
                pneumonia and tuberculosis classification.
              </p>
            </article>

            <article className="feature-card">
              <div className="feature-icon">
                <Activity />
              </div>

              <h3>Explainable AI</h3>

              <p>
                Planned visual explanations to help explore which image
                regions influence model predictions.
              </p>
            </article>

            <article className="feature-card">
              <div className="feature-icon">
                <FileText />
              </div>

              <h3>AI Reports</h3>

              <p>
                A planned report-generation workflow to present model
                findings in a clear and accessible format.
              </p>
            </article>

            <article className="feature-card">
              <div className="feature-icon">
                <Globe2 />
              </div>

              <h3>Multilingual Access</h3>

              <p>
                A planned multilingual interface for improving
                accessibility in underserved communities.
              </p>
            </article>

          </div>
        </section>

        {/* LIVE AI DEMONSTRATION */}

        <section className="demo-section" id="demo">

          <div className="section-heading">
            <span className="section-label">
              AI-POWERED DEMONSTRATION
            </span>

            <h2>Explore chest X-ray analysis.</h2>

            <p>
              Upload a chest X-ray image to receive experimental
              predictions from our DenseNet-121 models.
            </p>
          </div>

          <div className="upload-card">

            <div className="upload-icon">
              <UploadCloud size={34} />
            </div>

            <h3>Upload a chest X-ray image</h3>

            <p>
              Supported formats: JPG, JPEG and PNG · Maximum 10 MB
            </p>

            <label className="upload-button">
              Choose an image

              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={handleFile}
              />
            </label>

            {selectedFile && (
              <div className="file-selected">
                <CheckCircle2 size={18} />
                {selectedFile.name}
              </div>
            )}

            {selectedFile && (
              <button
                className="primary-button"
                onClick={handlePrediction}
                disabled={loading}
                style={{
                  marginTop: "20px",
                  border: "none",
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.7 : 1
                }}
              >
                {loading ? "Analyzing image..." : "Analyze X-ray"}

                {!loading && <ArrowRight size={18} />}
              </button>
            )}

            {loading && (
              <p style={{ marginTop: "20px" }}>
                Please wait while both AI models analyse your image...
              </p>
            )}

            {error && (
              <p
                role="alert"
                style={{
                  color: "#ff7777",
                  marginTop: "18px"
                }}
              >
                {error}
              </p>
            )}

            {/* PREDICTION RESULTS */}

            {prediction && (
              <div
                style={{
                  marginTop: "30px",
                  textAlign: "left",
                  width: "100%"
                }}
              >
                <h3>Analysis Results</h3>

                <p
                  style={{
                    color: "#55d6be",
                    marginBottom: "20px"
                  }}
                >
                  Both models completed their predictions successfully.
                </p>

                <ResultCard
                  title="Pneumonia Analysis"
                  result={prediction.results.pneumonia}
                />

                <ResultCard
                  title="Tuberculosis Analysis"
                  result={prediction.results.tuberculosis}
                />

                <p
                  style={{
                    fontSize: "13px",
                    color: "#aab5c5",
                    lineHeight: "1.6"
                  }}
                >
                  {prediction.notice}
                </p>

              </div>
            )}

            <div className="upload-disclaimer">
              <ShieldCheck size={16} />

              Experimental research predictions only.
              Not intended for medical diagnosis or treatment.
              Use permitted research images only; avoid real patient data.
            </div>

          </div>
        </section>

        {/* RESEARCH SECTION */}

        <section className="research-section" id="research">

          <div>
            <span className="section-label">
              RESEARCH & DEVELOPMENT
            </span>

            <h2>Build. Explain. Trust.</h2>

            <p>
              Our research explores how deep learning models can be combined
              with explainability techniques and accessible reporting to
              support the study of chest X-ray analysis.
            </p>

            <a href="#about" className="text-link">
              Learn about the project <ArrowRight size={17} />
            </a>
          </div>

          <div className="research-visual">
            <div className="research-circle">
              <BrainCircuit size={55} />
            </div>

            <span>AI RESEARCH FRAMEWORK</span>
          </div>

        </section>

      </main>

      {/* FOOTER */}

      <footer id="about">

        <div className="footer-brand">

          <a href="#home" className="brand">
            <span className="brand-icon">
              <HeartPulse size={23} />
            </span>

            CHEXPERT<span className="brand-dot">.</span>
          </a>

          <p>AI-powered chest X-ray research prototype.</p>

        </div>

        <div className="footer-bottom">
          <span>
            © {new Date().getFullYear()} CHEXPERT. Academic project.
          </span>

          <span>
            Not intended for medical diagnosis or treatment.
          </span>
        </div>

      </footer>

    </div>
  );
}

export default App;
