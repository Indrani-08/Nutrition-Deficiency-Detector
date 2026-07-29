const BASE_URL = "http://127.0.0.1:5000";

// ---------------- AI Prediction ----------------

export async function predictNail(image) {
  const formData = new FormData();
  formData.append("image", image);

  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  return response.json();
}

// ---------------- Register ----------------

export async function registerUser(userData) {
  const response = await fetch(`${BASE_URL}/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(userData),
  });

  return response.json();
}

// ---------------- Login ----------------

export async function loginUser(userData) {
  const response = await fetch(`${BASE_URL}/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(userData),
  });

  return response.json();
}

// ---------------- Save Report ----------------

export async function saveReport(reportData) {
  const response = await fetch(`${BASE_URL}/save-report`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(reportData),
  });

  return response.json();
}

// ---------------- Forgot Password ----------------

export async function forgotPassword(email) {
  const response = await fetch(`${BASE_URL}/forgot-password`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email }),
  });

  return response.json();
}

// ---------------- Verify OTP ----------------

export async function verifyOtp(data) {
  const response = await fetch(`${BASE_URL}/verify-otp`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  return response.json();
}

// ---------------- Reset Password ----------------

export async function resetPassword(data) {
  const response = await fetch(`${BASE_URL}/reset-password`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  return response.json();
}

// ---------------- Get Reports ----------------

export async function getReports(userId) {
  const response = await fetch(`${BASE_URL}/reports/${userId}`);

  return response.json();
}

// ---------------- Delete Report ----------------

export async function deleteReport(reportId) {
  const response = await fetch(
    `${BASE_URL}/report/${reportId}`,
    {
      method: "DELETE",
    }
  );

  return response.json();
}