import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import Navbar from "../components/Navbar";
import { loginUser } from "../services/api";

function Login() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const result = await loginUser(formData);

    if (result.success) {
      alert("Login Successful!");

      localStorage.setItem("user", JSON.stringify(result.user));

      navigate("/");
    } else {
      alert(result.message);
    }
  };

  return (
    <>
      <Navbar />

      <section className="auth-container">
        <div className="auth-card">
          <h2>Welcome Back</h2>

          <p className="auth-subtitle">
            Login to continue your health journey.
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>
            <input
              type="email"
              name="email"
              placeholder="Email Address"
              value={formData.email}
              onChange={handleChange}
              required
            />

            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                required
              />

              <button
                type="button"
                className="toggle-password"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
            </div>

            <p className="forgot-password">
              <Link to="/forgot-password">Forgot Password?</Link>
            </p>

            <button type="submit">
              Login
            </button>
          </form>

          <p className="auth-footer">
            Don't have an account?
            <Link to="/signup"> Sign Up</Link>
          </p>
        </div>
      </section>
    </>
  );
}

export default Login;