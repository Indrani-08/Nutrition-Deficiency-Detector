import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import Navbar from "../components/Navbar";
import { registerUser } from "../services/api";

function Signup() {

  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);

  const [formData, setFormData] = useState({
    fullname: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match.");
      return;
    }

    const result = await registerUser({
      fullname: formData.fullname,
      email: formData.email,
      password: formData.password,
    });

    if (result.success) {
      alert(result.message);
      navigate("/login");
    } else {
      alert(result.message);
    }
  };

  return (
    <>
      <Navbar />

      <section className="auth-container">

        <div className="auth-card">

          <h2>Create Your Account</h2>

          <p className="auth-subtitle">
            Join Nail Nutrition and start tracking your health with AI.
          </p>

          <form className="auth-form" onSubmit={handleSubmit}>

            <input
              type="text"
              name="fullname"
              placeholder="Full Name"
              value={formData.fullname}
              onChange={handleChange}
              required
            />

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

            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                name="confirmPassword"
                placeholder="Confirm Password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
              />
            </div>

            <button type="submit">
              Create Account
            </button>

          </form>

          <p className="auth-footer">
            Already have an account?
            <Link to="/login"> Login</Link>
          </p>

        </div>

      </section>
    </>
  );
}

export default Signup;