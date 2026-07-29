import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import { getReports, deleteReport } from "../services/api";
import { useNavigate } from "react-router-dom";

function Reports() {

  const user = JSON.parse(localStorage.getItem("user"));

  const [reports, setReports] = useState([]);

  const navigate = useNavigate();

  // Load Reports
  useEffect(() => {

    async function loadReports() {

      const data = await getReports(user.id);

      setReports(data);

    }

    loadReports();

  }, [user.id]);

  // Delete Report
  const handleDelete = async (reportId) => {

    const confirmDelete = window.confirm(
      "Are you sure you want to delete this report?"
    );

    if (!confirmDelete) return;

    const result = await deleteReport(reportId);

    if (result.success) {

      alert("Report deleted successfully!");

      setReports((prevReports) =>
        prevReports.filter((report) => report.id !== reportId)
      );

    } else {

      alert(result.message);

    }

  };

  return (
    <>
      <Navbar />

      <section className="auth-container">

        <div className="auth-card">

          <h2>My Reports</h2>

          {reports.length === 0 ? (

            <p>No reports available.</p>

          ) : (

            reports.map((report) => (

              <div
                className="report-card"
                key={report.id}
              >

                <h3>{report.prediction}</h3>

                <p>
                  <strong>Confidence:</strong>{" "}
                  {report.confidence}%
                </p>

                <p>
                  <strong>Risk Level:</strong>{" "}
                  <span
                    className={
                      report.risk_level === "High"
                        ? "risk-high"
                        : report.risk_level === "Moderate"
                        ? "risk-moderate"
                        : "risk-low"
                    }
                  >
                    {report.risk_level}
                  </span>
                </p>

                <p>{report.description}</p>

                <small>{report.created_at}</small>

                <div className="report-actions">

                  <button
                    className="view-btn"
                    onClick={() =>
                      navigate("/report-details", {
                        state: { report },
                      })
                    }
                  >
                    👁 View Details
                  </button>

                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(report.id)}
                  >
                    🗑 Delete
                  </button>

                </div>

              </div>

            ))

          )}

        </div>

      </section>

    </>
  );
}

export default Reports;