import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";

const expressionEmoji = {
  happy: "😄",
  sad: "😢",
  angry: "😠",
  surprised: "😲",
  neutral: "😐",
  fearful: "😨",
  disgusted: "🤢",
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
  const [person, setPerson] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
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
      }
    };
    loadModels();
  }, []);

  const loadLabeledImages = async () => {
    const labels = ["vignesh"]; // add more names here
    return Promise.all(
      labels.map(async (label) => {
        const img = await faceapi.fetchImage(`/known/${label}.jpg`);
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
        const resized = faceapi.resizeResults(detection, displaySize);
        faceapi.draw.drawDetections(canvas, resized);
        faceapi.draw.drawFaceExpressions(canvas, resized);

        const topExpression = Object.entries(detection.expressions).reduce(
          (a, b) => (a[1] > b[1] ? a : b)
        )[0];
        setExpression(topExpression);

        const faceMatcher = new faceapi.FaceMatcher(
          labeledFaceDescriptorsRef.current,
          0.6
        );
        const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
        setPerson(bestMatch.toString());
      } else {
        setExpression("");
        setPerson("");
      }
    }
  };

  return (
    <div className="container py-4">
      <div className="card shadow-lg">
        <div className="card-header bg-primary text-white text-center">
          <h3 className="mb-0">🎓 Emotion Recognition System</h3>
        </div>
        <div className="card-body">
          <div className="d-flex justify-content-center position-relative">
            <Webcam
              ref={webcamRef}
              audio={false}
              videoConstraints={videoConstraints}
              className="border rounded shadow-sm"
              style={{ width: "100%", maxWidth: 640 }}
            />
            <canvas
              ref={canvasRef}
              width={videoConstraints.width}
              height={videoConstraints.height}
              style={{
                position: "absolute",
                top: 0,
                left: "50%",
                transform: "translateX(-50%)",
                zIndex: 2,
              }}
            />
          </div>

          <div className="text-center mt-4">
            <h1 style={{ fontSize: "4rem" }}>
              {expression ? expressionEmoji[expression] || "🤔" : "🕵️‍♂️"}
            </h1>
            <h5>
              {expression ? (
                <span className="badge bg-success px-3 py-2">
                  Detected Emotion: {expression}
                </span>
              ) : (
                <span className="badge bg-secondary px-3 py-2">
                  Scanning for emotion...
                </span>
              )}
            </h5>
            {/* <h5 className="mt-3">
              {person ? (
                <span className="badge bg-info text-dark px-3 py-2">
                  Recognized: {person}
                </span>
              ) : (
                <span className="badge bg-warning text-dark px-3 py-2">
                  Identifying face...
                </span>
              )}
            </h5>
            <div className="text-muted mt-3">
              Last update: {new Date().toLocaleTimeString()}
            </div> */}
          </div>
        </div>
        <div className="card-footer text-center text-muted small">
          Powered by face-api.js | Built with ❤️ by Vignesh
        </div>
      </div>
    </div>
  );
}
