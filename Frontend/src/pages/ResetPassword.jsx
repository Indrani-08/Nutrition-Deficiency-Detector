import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Navbar from "../components/Navbar";
import { resetPassword } from "../services/api";

function ResetPassword() {
  const navigate = useNavigate();
  const location = useLocation();

  const email = location.state?.email;

  const [newPassword, setNewPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    const data = await resetPassword({
      email,
      new_password: newPassword,
    });

    if (data.success) {
      alert("Password reset successfully!");
      navigate("/login");
    } else {
      alert(data.message);
    }
  };

  return (
    <>
      <Navbar />

      <section className="auth-container">
        <div className="auth-card">

          <h2>Reset Password</h2>

          <p className="auth-subtitle">
            Enter your new password.
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>

            <input
              type="password"
              placeholder="New Password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
            />

            <button type="submit">
              Reset Password
            </button>

          </form>

        </div>
      </section>
    </>
  );
}

export default ResetPassword;