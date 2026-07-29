import Navbar from "../components/Navbar";

function MyHealth() {
  return (
    <>
      <Navbar />

      <section className="my-health">

        <h2>My Health</h2>

        <p className="health-description">
          Manage your health profile, access previous AI analyses,
          and explore upcoming features for personalized health tracking.
        </p>

        <div className="health-grid">

          <div className="health-card">
            <h3>👤 User Information</h3>

            <p><strong>Name:</strong> ____________</p>
            <p><strong>Email:</strong> ____________</p>
            <p><strong>Age:</strong> ____________</p>
            <p><strong>Gender:</strong> ____________</p>
          </div>

          <div className="health-card">
            <h3>📊 Previous Analyses</h3>

            <p>No previous analyses available.</p>
          </div>

          <div className="health-card">
            <h3>📁 Saved Reports</h3>

            <p>No saved reports found.</p>
          </div>

          <div className="health-card">
            <h3>🔐 Account</h3>

            <button className="health-btn">Login</button>

            <button className="health-btn signup">
              Sign Up
            </button>
          </div>

          <div className="health-card future">
            <h3>🚀 Upcoming Features</h3>

            <ul>
              <li>Save AI Reports</li>
              <li>Download PDF Reports</li>
              <li>Nutrition Progress Tracking</li>
              <li>Personalized Health Dashboard</li>
              <li>Cloud Synchronization</li>
            </ul>

          </div>

        </div>

      </section>

    </>
  );
}

export default MyHealth;