import { saveReport } from "../services/api";

function ResultCard({ result }) {
  if (!result) return null;

  const riskClass =
    result.risk_level.toLowerCase() === "high"
      ? "high"
      : result.risk_level.toLowerCase() === "moderate"
      ? "moderate"
      : "low";

  const handleSave = async () => {
    const user = JSON.parse(localStorage.getItem("user"));

    if (!user) {
      alert("Please login first.");
      return;
    }

    const response = await saveReport({
      user_id: user.id,
      prediction: result.prediction,
      confidence: result.confidence,
      risk_level: result.risk_level,
      description: result.description,
    });

    if (response.success) {
      alert("Report saved successfully!");
    } else {
      alert("Failed to save report.");
    }
  };

  return (
    <div className="result-card">

      <div className="result-header">
        <h2>Analysis Report</h2>
        <h3>{result.prediction}</h3>
      </div>

      <div className="top-cards">

        <div className="info-box">
          <h4>Confidence Score</h4>

          <div className="progress-bar">
            <div
              className="progress"
              style={{ width: `${result.confidence}%` }}
            ></div>
          </div>

          <p>{result.confidence}%</p>
        </div>

        <div className="info-box">
          <h4>Risk Level</h4>

          <span className={`risk-badge ${riskClass}`}>
            {result.risk_level}
          </span>
        </div>

      </div>

      <div className="section">
        <h4>Clinical Observation</h4>
        <p>{result.description}</p>
      </div>

      <div className="grid">

        <div className="card">
          <h4>Symptoms</h4>

          <ul>
            {result.symptoms.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>

        </div>

        <div className="card">
          <h4>Recommended Foods</h4>

          <ul>
            {result.foods.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>

        </div>

        <div className="card">
          <h4>Home Remedies</h4>

          <ul>
            {result.home_remedies.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>

        </div>

        <div className="card">
          <h4>Lifestyle Recommendations</h4>

          <ul>
            {result.lifestyle.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>

        </div>

      </div>

      <div className="section doctor">
        <h4>Medical Recommendation</h4>
        <p>{result.doctor_advice}</p>
      </div>

      <div style={{ textAlign: "center", marginTop: "30px" }}>
        <button
          type="button"
          onClick={handleSave}
          className="save-report-btn"
        >
          💾 Save Report
        </button>
      </div>

      <div className="section disclaimer">
        <h4>Medical Disclaimer</h4>
        <p>{result.disclaimer}</p>
      </div>

    </div>
  );
}

export default ResultCard;