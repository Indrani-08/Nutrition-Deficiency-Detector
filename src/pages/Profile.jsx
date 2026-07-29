import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";

function Profile() {
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem("user"));

  const [profileImage, setProfileImage] = useState(
    localStorage.getItem("profileImage")
  );

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onloadend = () => {
      localStorage.setItem("profileImage", reader.result);
      setProfileImage(reader.result);
    };

    reader.readAsDataURL(file);
  };

  return (
    <>
      <Navbar />

      <section className="auth-container">
        <div className="auth-card">

          {/* Profile Image */}
          <div className="profile-image-container">

            {profileImage ? (
              <img
                src={profileImage}
                alt="Profile"
                className="profile-image"
              />
            ) : (
              <div className="profile-avatar">
                {user?.fullname?.charAt(0).toUpperCase()}
              </div>
            )}

            <label className="upload-btn">
              📷 Change Photo
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={handleImageChange}
              />
            </label>

          </div>

          <h2>My Profile</h2>

          <div className="profile-details">

            <p>
              <strong>Full Name:</strong><br />
              {user?.fullname}
            </p>

            <p>
              <strong>Email:</strong><br />
              {user?.email}
            </p>

          </div>

          {/* My Reports Button */}
          <button
            className="profile-btn"
            onClick={() => navigate("/reports")}
          >
            📜 My Reports
          </button>

        </div>
      </section>
    </>
  );
}

export default Profile;