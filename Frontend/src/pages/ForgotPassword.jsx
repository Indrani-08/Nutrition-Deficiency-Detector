import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { forgotPassword } from "../services/api";

function ForgotPassword() {

  const navigate = useNavigate();

  const [email, setEmail] = useState("");

  const handleSubmit = async (e) => {
  e.preventDefault();

  const data = await forgotPassword(email);

  if (data.success) {
    alert("OTP sent successfully!");

    navigate("/verify-otp", {
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

          <h2>Forgot Password</h2>

          <p className="auth-subtitle">
            Enter your registered email address.
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>

            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e)=>setEmail(e.target.value)}
              required
            />

            <button type="submit">
              Send OTP
            </button>

          </form>

        </div>

      </section>
    </>
  );
}

export default ForgotPassword;