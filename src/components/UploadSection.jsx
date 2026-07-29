import { useRef, useState } from "react";
import { predictNail } from "../services/api";
import ResultCard from "./ResultCard";

function UploadSection() {
  const fileInputRef = useRef(null);

  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (file) => {
    if (!file) return;

    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleFileInput = (e) => {
    handleImageChange(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleImageChange(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      alert("Please upload a nail image.");
      return;
    }

    try {
      setLoading(true);

      const response = await predictNail(selectedImage);

      setResult(response);
    } catch (err) {
      console.error(err);
      alert("Unable to analyze the image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="upload-section">

        <div
          className="upload-box"
          onClick={() => fileInputRef.current.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {!preview ? (
            <>
              <div className="upload-icon">Upload</div>

              <h3>Upload Nail Image</h3>

              <p>
                Drag and drop a nail image here
              </p>

              <span>or click to browse your device</span>

              <small>Supported Formats: PNG, JPG, JPEG</small>
            </>
          ) : (
            <>
              <img src={preview} alt="Nail Preview" />

              <p className="success">
                Image Ready for Analysis
              </p>
            </>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={handleFileInput}
          />
        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? "Analyzing Image..." : "Analyze Nail Image"}
        </button>

      </div>

      <ResultCard result={result} />
    </>
  );
}

export default UploadSection;