import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Navbar from "../components/Navbar";
import { verifyOtp } from "../services/api";

function VerifyOtp() {
  const navigate = useNavigate();
  const location = useLocation();

  const email = location.state?.email;

  const [otp, setOtp] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    const data = await verifyOtp({
      email,
      otp,
    });

    if (data.success) {
      alert("OTP verified successfully!");

      navigate("/reset-password", {
        state: { email },
      });
    } else {
      alert(data.message);
    }
  };

  return (
    <>
      <Navbar />

      <section className="auth-container">
        <div className="auth-card">

          <h2>Verify OTP</h2>

          <p className="auth-subtitle">
            Enter the OTP sent to your email.
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>

            <input
              type="text"
              placeholder="Enter OTP"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              required
            />

            <button type="submit">
              Verify OTP
            </button>

          </form>

        </div>
      </section>
    </>
  );
}

export default VerifyOtp;