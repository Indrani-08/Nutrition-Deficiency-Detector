function HowItWorksSection() {
  return (
    <section className="how-it-works">
      <h2>How It Works</h2>

      <p className="section-description">
        Our AI-powered system analyzes fingernail images to identify
        possible nutritional deficiencies using deep learning techniques.
        Follow these simple steps to receive your health assessment.
      </p>

      <div className="steps">

        <div className="step-card">
          <div className="step-number">1</div>
          <h3>Upload Nail Image</h3>
          <p>
            Upload a clear image of your fingernail for AI analysis.
          </p>
        </div>

        <div className="step-card">
          <div className="step-number">2</div>
          <h3>Image Processing</h3>
          <p>
            The uploaded image is processed and prepared for prediction.
          </p>
        </div>

        <div className="step-card">
          <div className="step-number">3</div>
          <h3>AI Analysis</h3>
          <p>
            Our trained TensorFlow model predicts possible nutritional deficiencies.
          </p>
        </div>

        <div className="step-card">
          <div className="step-number">4</div>
          <h3>Health Report</h3>
          <p>
            Receive confidence scores, dietary suggestions, and health recommendations.
          </p>
        </div>

      </div>
    </section>
  );
}

export default HowItWorksSection;