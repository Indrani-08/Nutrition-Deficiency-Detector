import { NavLink, useNavigate } from "react-router-dom";

function Navbar() {
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem("user"));

  const handleLogout = () => {
    localStorage.removeItem("user");
    alert("Logged out successfully!");
    navigate("/login");
  };

  return (
    <nav className="navbar">

      <div className="logo">
        <h2>Nail Nutrition</h2>
        <span className="logo-subtitle">
          AI-Powered Health Screening
        </span>
      </div>

      <ul className="nav-links">

        <li>
          <NavLink to="/" end>
            Home
          </NavLink>
        </li>

        <li>
          <NavLink to="/how-it-works">
            How It Works
          </NavLink>
        </li>

        <li>
          <NavLink to="/about">
            About
          </NavLink>
        </li>

        {user ? (
          <>
            <li>
              <NavLink to="/profile">
                Profile
              </NavLink>
            </li>

            <li>
              <button
                className="logout-btn"
                onClick={handleLogout}
              >
                Logout
              </button>
            </li>
          </>
        ) : (
          <>
            <li>
              <NavLink to="/signup">
                Sign Up
              </NavLink>
            </li>

            <li>
              <NavLink to="/login">
                Login
              </NavLink>
            </li>
          </>
        )}

      </ul>

    </nav>
  );
}

export default Navbar;