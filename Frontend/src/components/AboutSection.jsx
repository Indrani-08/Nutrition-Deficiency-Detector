function AboutSection() {
  return (
    <section className="about">

      <h2>About the Project</h2>

      <p className="about-description">
        Nail Nutrition is an AI-powered healthcare application that analyzes
        fingernail images to identify possible nutritional deficiencies.
        The system uses Deep Learning techniques to assist in early nutritional
        screening by providing health insights and personalized recommendations.
      </p>

      <div className="about-grid">

        <div className="about-card">
          <h3>Project Objective</h3>

          <p>
            To develop an AI-based solution that detects nutritional
            deficiencies through fingernail image analysis and helps users
            understand their health condition at an early stage.
          </p>
        </div>

        <div className="about-card">
          <h3>Technology Stack</h3>

          <ul>
            <li>React.js</li>
            <li>Flask</li>
            <li>TensorFlow / Keras</li>
            <li>Python</li>
            <li>HTML5 & CSS3</li>
          </ul>
        </div>

        <div className="about-card">
          <h3>Key Features</h3>

          <ul>
            <li>AI-based Nail Image Analysis</li>
            <li>Confidence Score Prediction</li>
            <li>Dietary Recommendations</li>
            <li>Lifestyle Suggestions</li>
            <li>Medical Guidance</li>
          </ul>
        </div>

      </div>

    </section>
  );
}

export default AboutSection;