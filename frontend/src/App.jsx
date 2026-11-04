
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

function App() {
  const [mobileMenu, setMobileMenu] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFile = (event) => {
    const file = event.target.files?.[0];

    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
    } else {
      setSelectedFile(null);
      alert("Please select a valid image file.");
    }
  };

  return (
    <div className="app">
      <nav className="navbar">
        <a href="#home" className="brand">
          <span className="brand-icon"><HeartPulse size={25} /></span>
          <span>CHEXPERT<span className="brand-dot">.</span></span>
        </a>

        <button className="menu-toggle" onClick={() => setMobileMenu(!mobileMenu)}>
          {mobileMenu ? <X /> : <Menu />}
        </button>

        <div className={`nav-links ${mobileMenu ? "show-menu" : ""}`}>
          <a href="#home" onClick={() => setMobileMenu(false)}>Home</a>
          <a href="#technology" onClick={() => setMobileMenu(false)}>Technology</a>
          <a href="#research" onClick={() => setMobileMenu(false)}>Research</a>
          <a href="#about" onClick={() => setMobileMenu(false)}>About</a>
          <a href="#demo" className="nav-button" onClick={() => setMobileMenu(false)}>
            Explore Demo <ArrowRight size={16} />
          </a>
        </div>
      </nav>

      <main>
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
                <span><Activity size={16} /> CHEST IMAGING</span>
                <span className="live-label">AI RESEARCH</span>
              </div>

              <div className="xray-placeholder">
                <HeartPulse size={100} strokeWidth={0.8} />
                <span>CHEST X-RAY</span>
              </div>

              <div className="scan-bottom">
                <span>Deep learning analysis</span>
                <span><CheckCircle2 size={15} /> Prototype</span>
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

        <section className="stats-strip">
          <div><strong>DenseNet-121</strong><span>Deep learning backbone</span></div>
          <div><strong>2 Models</strong><span>Pneumonia and TB</span></div>
          <div><strong>224 × 224</strong><span>Input image resolution</span></div>
          <div><strong>Research</strong><span>Experimental prototype</span></div>
        </section>

        <section className="section" id="technology">
          <div className="section-heading">
            <span className="section-label">OUR TECHNOLOGY</span>
            <h2>Intelligence behind the image.</h2>
            <p>
              A research-oriented pipeline combining computer vision and
              artificial intelligence for chest X-ray analysis.
            </p>
          </div>

          <div className="feature-grid">
            <article className="feature-card">
              <div className="feature-icon"><BrainCircuit /></div>
              <h3>Deep Learning</h3>
              <p>DenseNet-121 transfer learning models for experimental pneumonia and tuberculosis classification.</p>
            </article>

            <article className="feature-card">
              <div className="feature-icon"><Activity /></div>
              <h3>Explainable AI</h3>
              <p>Planned visual explanations to help explore which image regions influence model predictions.</p>
            </article>

            <article className="feature-card">
              <div className="feature-icon"><FileText /></div>
              <h3>AI Reports</h3>
              <p>A planned report-generation workflow to present model findings in a clear and accessible format.</p>
            </article>

            <article className="feature-card">
              <div className="feature-icon"><Globe2 /></div>
              <h3>Multilingual Access</h3>
              <p>A planned multilingual interface for improving accessibility in underserved communities.</p>
            </article>
          </div>
        </section>

        <section className="demo-section" id="demo">
          <div className="section-heading">
            <span className="section-label">EXPERIMENTAL DEMONSTRATION</span>
            <h2>Explore chest X-ray analysis.</h2>
            <p>
              Select an image to preview the research demonstration interface.
              Model inference will be connected in a later deployment step.
            </p>
          </div>

          <div className="upload-card">
            <div className="upload-icon"><UploadCloud size={34} /></div>
            <h3>Upload a chest X-ray image</h3>
            <p>Supported formats: JPG, JPEG and PNG</p>

            <label className="upload-button">
              Choose an image
              <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleFile} />
            </label>

            {selectedFile && (
              <div className="file-selected">
                <CheckCircle2 size={18} />
                {selectedFile.name}
              </div>
            )}

            <div className="upload-disclaimer">
              <ShieldCheck size={16} />
              This is a frontend preview. No AI prediction is performed yet.
              Do not upload real patient images.
            </div>
          </div>
        </section>

        <section className="research-section" id="research">
          <div>
            <span className="section-label">RESEARCH & DEVELOPMENT</span>
            <h2>Build. Explain. Trust.</h2>
            <p>
              Our research explores how deep learning models can be combined
              with explainability techniques and accessible reporting to
              support the study of chest X-ray analysis.
            </p>
            <a href="#about" className="text-link">Learn about the project <ArrowRight size={17} /></a>
          </div>
          <div className="research-visual">
            <div className="research-circle"><BrainCircuit size={55} /></div>
            <span>AI RESEARCH FRAMEWORK</span>
          </div>
        </section>
      </main>

      <footer id="about">
        <div className="footer-brand">
          <a href="#home" className="brand">
            <span className="brand-icon"><HeartPulse size={23} /></span>
            CHEXPERT<span className="brand-dot">.</span>
          </a>
          <p>AI-powered chest X-ray research prototype.</p>
        </div>

        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} CHEXPERT. Academic project.</span>
          <span>Not intended for medical diagnosis or treatment.</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
