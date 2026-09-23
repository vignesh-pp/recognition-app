import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import "./App.css";

const knownPeople = {
  vignesh: { name: "Vignesh", image: "vignesh.jpg" },
  pooarasu: { name: "Pooarasu", image: "pooarasu.jpg" },
  madesh: { name: "Madesh", image: "madesh.jpg" },
  gokul: { name: "Gokul", image: "gokul.jpg" },
  arun: { name: "Arun", image: "arun.jpg" },
};

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [expression, setExpression] = useState("");
  const [identity, setIdentity] = useState(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelError, setModelError] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");
  const labeledFaceDescriptorsRef = useRef([]);

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = "/models";
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(
            `${MODEL_URL}/ssd_mobilenetv1`
          ),
          faceapi.nets.faceExpressionNet.loadFromUri(
            `${MODEL_URL}/face_expression`
          ),
          faceapi.nets.faceLandmark68Net.loadFromUri(
            `${MODEL_URL}/face_landmark_68`
          ),
          faceapi.nets.faceRecognitionNet.loadFromUri(
            `${MODEL_URL}/face_recognition`
          ),
        ]);
        labeledFaceDescriptorsRef.current = await loadLabeledImages();
        setModelsLoaded(true);
      } catch (error) {
        console.error("Model load error:", error);
        setModelError(true);
      }
    };
    loadModels();
  }, []);

  const loadLabeledImages = async () => {
    const labels = Object.entries(knownPeople);
    return Promise.all(
      labels.map(async ([label, profile]) => {
        const img = await faceapi.fetchImage(`/known/${profile.image}`);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (!detections) return null;
        return new faceapi.LabeledFaceDescriptors(label, [
          detections.descriptor,
        ]);
      })
    ).then((data) => data.filter(Boolean));
  };

  useEffect(() => {
    if (!modelsLoaded) return;
    const interval = setInterval(() => detect(), 1000);
    return () => clearInterval(interval);
  }, [modelsLoaded]);

  const detect = async () => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };
      faceapi.matchDimensions(canvasRef.current, displaySize);

      const detection = await faceapi
        .detectSingleFace(video, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions();

      const canvas = canvasRef.current;
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      if (detection) {
        setFaceDetected(true);
        const resized = faceapi.resizeResults(detection, displaySize);
        faceapi.draw.drawDetections(canvas, resized);
        faceapi.draw.drawFaceExpressions(canvas, resized);

        const topExpression = Object.entries(detection.expressions).reduce(
          (a, b) => (a[1] > b[1] ? a : b)
        )[0];
        setExpression(topExpression);

        if (labeledFaceDescriptorsRef.current.length) {
          const faceMatcher = new faceapi.FaceMatcher(
            labeledFaceDescriptorsRef.current,
            0.6
          );
          const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
          setIdentity({
            label: bestMatch.label,
            distance: bestMatch.distance,
          });
        }
        setLastUpdated(
          new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })
        );
      } else {
        setExpression("");
        setIdentity(null);
        setFaceDetected(false);
      }
    }
  };

  const systemState = modelError || cameraError ? "error" : modelsLoaded ? "ready" : "loading";
  const systemLabel = cameraError
    ? "Camera unavailable"
    : modelError
    ? "Models unavailable"
    : modelsLoaded
      ? "Analysis ready"
      : "Loading models";
  const isKnownPerson = identity && identity.label !== "unknown";
  const identityName = isKnownPerson
    ? knownPeople[identity.label]?.name || identity.label
    : identity
      ? "Unknown person"
      : "Awaiting face";
  const matchConfidence = isKnownPerson
    ? `${Math.round((1 - identity.distance) * 100)}% match`
    : identity
      ? "No enrolled match"
      : "No reading yet";

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Vision analysis</p>
            <h1>Emotion recognition</h1>
          </div>
          <div className={`system-status ${systemState}`}>
            <span className="status-dot" />
            {systemLabel}
          </div>
        </header>

        <div className="analysis-grid">
          <section className="camera-panel" aria-label="Live camera feed">
            <div className="panel-heading">
              <p className="panel-label">Camera input</p>
              <span className="live-label">LIVE</span>
            </div>
            <div className="camera-frame">
            <Webcam
              ref={webcamRef}
              audio={false}
              videoConstraints={videoConstraints}
              className="camera-feed"
              onUserMedia={() => setCameraError(false)}
              onUserMediaError={() => setCameraError(true)}
            />
            <canvas
              ref={canvasRef}
              width={videoConstraints.width}
              height={videoConstraints.height}
              className="detection-layer"
            />
            </div>
            <div className="camera-caption">
              <span>{cameraError ? "Allow camera access to start analysis" : "Face landmark and expression tracking"}</span>
              <strong>{faceDetected ? "Face found" : modelsLoaded ? "Searching" : "Standby"}</strong>
            </div>
          </section>

          <aside className="insights-panel" aria-live="polite">
            <p className="panel-label">Current reading</p>
            <div className="emotion-display">
              <p className="detail-label">Dominant expression</p>
              <p className={`emotion-value ${expression ? "" : "waiting"}`}>
                {expression || "Scanning frame"}
              </p>
              <p className="emotion-copy">
                {expression
                  ? "The strongest expression detected in the current frame."
                  : "Position one face inside the camera frame to begin analysis."}
              </p>
            </div>
            <div className="readings">
              <div className="reading">
                <div><p className="detail-label">Who is this?</p></div>
                <p className={`detail-value ${isKnownPerson ? "" : "unknown"}`}>
                  {identityName}
                </p>
              </div>
              <div className="reading">
                <div><p className="detail-label">Match confidence</p></div>
                <p className={`detail-value ${isKnownPerson ? "" : "unknown"}`}>
                  {matchConfidence}
                </p>
              </div>
              <div className="reading">
                <div><p className="detail-label">Last analysis</p></div>
                <p className={`detail-value ${lastUpdated ? "" : "unknown"}`}>
                  {lastUpdated || "No reading yet"}
                </p>
              </div>
            </div>
            {/* <div className="enrolled-people">
              <p className="detail-label">Enrolled people</p>
              <div className="profile-list">
                {Object.values(knownPeople).map((profile) => (
                  <span className="profile-name" key={profile.name}>{profile.name}</span>
                ))}
              </div>
            </div> */}
            <p className={`notice ${modelError || cameraError ? "error" : ""}`}>
              {cameraError
                ? "Camera access is required for live analysis. Check your browser permission, then reload this page."
                : modelError
                ? "The analysis models could not be loaded. Refresh the page and check that the local model files are available."
                : "Analysis runs locally in your browser. Camera frames are not sent to a server."}
            </p>
          </aside>
        </div>
      </section>
    </main>
  );
}
