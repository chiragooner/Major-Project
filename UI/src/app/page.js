"use client";

import { useState, useEffect } from "react";
import Image from "next/image";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [voicePrediction, setVoicePrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [voiceEncoding, setVoiceEncoding] = useState("");
  const [selectedModel, setSelectedModel] = useState("vgg19");

  // Clear preview image when selecting voice model
  useEffect(() => {
    if (selectedModel === "randomForest") {
      setPreviewImage(null);
    }
  }, [selectedModel]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
      setVoicePrediction(null);
    }
  };

  const handleVoiceUpload = (e) => {
    setVoiceEncoding(e.target.value);
    setPrediction(null);
    setError(null);
    setVoicePrediction(null);
  };

  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
    
    // Clear preview image when switching to voice model
    if (newModel === "randomForest") {
      setPreviewImage(null);
    }
    
    // Reset predictions when model changes
    setPrediction(null);
    setVoicePrediction(null);
  };

  const handlePredict = async () => {
    console.log(`Selected model: ${selectedModel}`);

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      console.log(`Sending image to backend using ${selectedModel} model...`);
      const response = await fetch(
        `http://127.0.0.1:8000/predict/${selectedModel}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response data:", data);
      setLoading(false);
      setPrediction(data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
      setError(`Error: ${error.message}`);
    }
  };

  const handlePredictVoice = async () => {
    console.log(`Selected model: ${selectedModel}`);

    setLoading(true);
    setError(null);

    try {
      console.log(`Sending voice encoding to backend using ${selectedModel} model...`);
      console.log(voiceEncoding);
      const formData = new FormData();
      formData.append("file", voiceEncoding);
      const response = await fetch(
        `http://127.0.0.1:8000/predict/${selectedModel}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        console.log("I am here");
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      setVoicePrediction(data.prediction);
      setLoading(false);
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
      setError(`Error: ${error.message}`);
    }
  };

  return (
    <div className="container">
      <h1 className="titleName">Parkinson&apos;s Disease Detection</h1>

      <div className="layoutStructure">
        <div className="form-container">
          <div className="form-box">
            <div className="form-group">
              <label className="model-label">Select Model</label>
              <select className="model-select" value={selectedModel} onChange={handleModelChange}>
                <option value="vgg19">VGG19 (Wave Flipping)</option>
                <option value="vgg16">VGG16 (Spiral No Aug)</option>
                <option value="randomForest">Voice Model</option>
              </select>
            </div>

            {selectedModel !== "randomForest" && (
              <div className="form-group">
                <label className="model-label">Upload Image</label>
                <input
                className="model-select"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </div>
            )}

            {selectedModel === "randomForest" && (
              <div className="form-group">
                <label className="model-label">Upload Voice Encoding</label>
                <input
                className="model-select"
                  placeholder="Enter voice encoding"
                  onChange={handleVoiceUpload}
                />
              </div>
            )}

            {selectedModel !== "randomForest" ? (
              <button
                onClick={handlePredict}
                className="btn"
                disabled={loading || !selectedImage}
              >
                {loading ? "Processing..." : "Predict"}
              </button>
            ) : (
              <button
                onClick={handlePredictVoice}
                className="btn"
                disabled={loading || !voiceEncoding}
              >
                {loading ? "Processing..." : "Predict"}
              </button>
            )}
          </div>
        </div>

        <div className="right-side-content">
          {previewImage && selectedModel !== "randomForest" && (
            <div className="preview-container">
              <p className="preview-title">Input Image</p>
              <div className="image-container">
                <Image
                  src={previewImage}
                  alt="Uploaded"
                  width={250}
                  height={250}
                  className="preview-image"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="error-container">
              <p className="error-text">{error}</p>
            </div>
          )}

          {prediction !== null && (
            <div className="results-container">
              <h2>Results</h2>

              <div className="results-box">
                <p className="result-item">
                  Model: <span className="result-label">{selectedModel.toUpperCase()}</span>
                </p>
                <p className="result-item">
                  Prediction:{" "}
                  <span className="prediction-value">
                    {prediction === 0 ? "Healthy" : "Parkinson's Disease"}
                  </span>
                </p>
              </div>
            </div>
          )}
          
          {voicePrediction !== null && (
            <div className="results-container">
              <h2>Results</h2>

              <div className="results-box">
                <p className="result-item">
                  Model: <span className="result-label">{selectedModel.toUpperCase()}</span>
                </p>
                <p className="result-item">
                  Prediction: <span className="prediction-value">{voicePrediction}</span>
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}