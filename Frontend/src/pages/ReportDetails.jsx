import { useLocation } from "react-router-dom";
import Navbar from "../components/Navbar";

function ReportDetails() {

  const location = useLocation();

  const report = location.state?.report;

  if (!report) {
    return <h2>No Report Found</h2>;
  }

  return (
    <>
      <Navbar />

      <section className="auth-container">

        <div className="auth-card">

          <h2>AI Report Details</h2>

        <div className="report-details-card">

  <div className="detail-box">
    <h4>🩺 Prediction</h4>
    <p>{report.prediction}</p>
  </div>

  <div className="detail-box">
    <h4>📊 Confidence</h4>
    <p>{report.confidence}%</p>
  </div>

  <div className="detail-box">
    <h4>⚠ Risk Level</h4>
    <p>{report.risk_level}</p>
  </div>

  <div className="detail-box">
    <h4>📝 Description</h4>
    <p>{report.description}</p>
  </div>

  <div className="detail-box">
    <h4>📅 Generated On</h4>
    <p>{report.created_at}</p>
  </div>

  <button
    className="back-btn"
    onClick={() => window.history.back()}
  >
    ⬅ Back to Reports
  </button>

</div>

        </div>

      </section>
    </>
  );
}

export default ReportDetails;