import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import { SelfieSegmentation } from "@mediapipe/selfie_segmentation";
import "./App.css";

const defaultKnownPeople = {
  vignesh: { name: "Vignesh", image: "vignesh.jpg",  },
  pooarasu: { name: "Pooarasu", image: "pooarasu.jpg",  },
  madesh: { name: "Madesh", image: "madesh.jpg",  },
  gokul: { name: "Gokul", image: "gokul.jpg",  },
  arun: { name: "Arun", image: "arun.jpg",  },
};

const EMOTION_MAP = {
  neutral: { label: "Neutral", emoji: "😐", color: "#94a3b8" },
  happy: { label: "Happy", emoji: "😊", color: "#10b981" },
  sad: { label: "Sad", emoji: "😔", color: "#60a5fa" },
  angry: { label: "Angry", emoji: "😠", color: "#f43f5e" },
  fearful: { label: "Fearful", emoji: "😨", color: "#a855f7" },
  disgusted: { label: "Disgusted", emoji: "🤢", color: "#84cc16" },
  surprised: { label: "Surprised", emoji: "😲", color: "#f59e0b" },
};

const virtualBackgrounds = [
  { id: "none", name: "Natural Camera", icon: "📷" },
  { id: "blur", name: "Studio Blur", icon: "✨" },
  {
    id: "paris",
    name: "Paris, France",
    icon: "🗼",
    image: "https://images.unsplash.com/photo-1637329096986-62486d0c4380?auto=format&fit=crop&w=1600&q=85",
  },
  {
    id: "new-york",
    name: "New York, USA",
    icon: "🗽",
    image: "https://images.unsplash.com/photo-1512621450499-28dfc7415645?auto=format&fit=crop&w=1600&q=85",
  },
  {
    id: "santorini",
    name: "Santorini, Greece",
    icon: "🏛️",
    image: "https://images.unsplash.com/photo-1677651647819-e590284af894?auto=format&fit=crop&w=1600&q=85",
  },
  {
    id: "taj-mahal",
    name: "Taj Mahal, India",
    icon: "🕌",
    image: "https://images.unsplash.com/photo-1523979934836-981ca2ded72f?auto=format&fit=crop&w=1600&q=85",
  },
  {
    id: "tokyo",
    name: "Tokyo, Japan",
    icon: "🗾",
    image: "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=1600&q=85",
  },
];

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

// Box interpolation helper for anti-jitter smoothing
function lerp(start, end, factor) {
  return start + (end - start) * factor;
}

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const backgroundCanvasRef = useRef(null);
  const foregroundCanvasRef = useRef(null);
  const segmentationRef = useRef(null);
  const segmentationBusyRef = useRef(false);
  const backgroundImageRef = useRef(null);
  const detectionBusyRef = useRef(false);

  // Smoothing & persistence refs
  const trackedFacesRef = useRef([]);
  const emptyFrameCountRef = useRef(0);
  const labeledFaceDescriptorsRef = useRef([]);

  // States
  const [knownProfiles, setKnownProfiles] = useState(defaultKnownPeople);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelError, setModelError] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [analysisError, setAnalysisError] = useState(false);
  const [backgroundMode, setBackgroundMode] = useState("none");
  const [lastUpdated, setLastUpdated] = useState("");
  const [inferenceTime, setInferenceTime] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [detectorModel, setDetectorModel] = useState("tiny"); // "tiny" | "ssd"
  const [minConfidence, setMinConfidence] = useState(0.35);
  const [activeTab, setActiveTab] = useState("insights"); // "insights" | "enrolled"
  const [isLivePaused, setIsLivePaused] = useState(false);

  // Modal Enrollment state
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [newPersonName, setNewPersonName] = useState("");
  const [newPersonRole, setNewPersonRole] = useState("Team Member");
  const [enrollError, setEnrollError] = useState("");
  const [enrollSuccess, setEnrollSuccess] = useState("");
  const [isEnrolling, setIsEnrolling] = useState(false);

  // 1. Virtual Background & Selfie Segmentation (Fixed Canvas Reset Bug)
  useEffect(() => {
    if (backgroundMode === "none") return;

    let active = true;
    let timer;
    const background = virtualBackgrounds.find((item) => item.id === backgroundMode);
    if (background?.image) {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.src = background.image;
      backgroundImageRef.current = image;
    } else {
      backgroundImageRef.current = null;
    }

    try {
      const segmentation = new SelfieSegmentation({
        locateFile: (file) => `/models/selfie_segmentation/${file}`,
      });
      segmentation.setOptions({ modelSelection: 1 });
      segmentation.onResults((results) => {
        const output = backgroundCanvasRef.current;
        const foreground = foregroundCanvasRef.current;
        if (!active || !output || !foreground) return;

        // Only resize canvas if dimensions actually changed (prevents canvas flashing)
        if (output.width !== results.image.width || output.height !== results.image.height) {
          output.width = results.image.width;
          output.height = results.image.height;
          foreground.width = results.image.width;
          foreground.height = results.image.height;
        }

        const outputContext = output.getContext("2d");
        const foregroundContext = foreground.getContext("2d");
        if (!outputContext || !foregroundContext) return;

        foregroundContext.clearRect(0, 0, foreground.width, foreground.height);
        foregroundContext.drawImage(results.image, 0, 0, foreground.width, foreground.height);
        foregroundContext.globalCompositeOperation = "destination-in";
        foregroundContext.drawImage(results.segmentationMask, 0, 0, foreground.width, foreground.height);
        foregroundContext.globalCompositeOperation = "source-over";

        outputContext.clearRect(0, 0, output.width, output.height);
        const replacementImage = backgroundImageRef.current;
        if (replacementImage?.complete && replacementImage.naturalWidth) {
          const imageScale = Math.max(
            output.width / replacementImage.naturalWidth,
            output.height / replacementImage.naturalHeight
          );
          const drawWidth = replacementImage.naturalWidth * imageScale;
          const drawHeight = replacementImage.naturalHeight * imageScale;
          outputContext.drawImage(
            replacementImage,
            (output.width - drawWidth) / 2,
            (output.height - drawHeight) / 2,
            drawWidth,
            drawHeight
          );
        } else {
          outputContext.filter = "blur(14px)";
          outputContext.drawImage(results.image, -14, -14, output.width + 28, output.height + 28);
          outputContext.filter = "none";
        }
        outputContext.drawImage(foreground, 0, 0);
      });
      segmentationRef.current = segmentation;

      const processFrame = async () => {
        const video = webcamRef.current?.video;
        if (!active || !video || video.readyState !== 4 || segmentationBusyRef.current) return;
        segmentationBusyRef.current = true;
        try {
          await segmentation.send({ image: video });
        } catch {
          // ignore frame skip
        } finally {
          segmentationBusyRef.current = false;
        }
      };

      timer = window.setInterval(processFrame, 80);
      return () => {
        active = false;
        window.clearInterval(timer);
        segmentation.close();
        segmentationRef.current = null;
        segmentationBusyRef.current = false;
      };
    } catch (err) {
      console.warn("Selfie segmentation error:", err);
    }
  }, [backgroundMode]);

  // 2. Load face-api models & Labeled Faces (Loads both TinyFace and SSD MobileNet)
  useEffect(() => {
    let isMounted = true;
    const loadModels = async () => {
      const MODEL_URL = "/models";
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(`${MODEL_URL}/tiny_face_detector`),
          faceapi.nets.ssdMobilenetv1.loadFromUri(`${MODEL_URL}/ssd_mobilenetv1`),
          faceapi.nets.faceExpressionNet.loadFromUri(`${MODEL_URL}/face_expression`),
          faceapi.nets.faceLandmark68Net.loadFromUri(`${MODEL_URL}/face_landmark_68`),
          faceapi.nets.faceRecognitionNet.loadFromUri(`${MODEL_URL}/face_recognition`),
        ]);
        if (!isMounted) return;

        const descriptors = await loadLabeledImages();
        labeledFaceDescriptorsRef.current = descriptors;
        setModelsLoaded(true);
        setModelError(false);
      } catch (error) {
        console.error("Model load error:", error);
        if (isMounted) setModelError(true);
      }
    };
    loadModels();

    return () => {
      isMounted = false;
    };
  }, []);

  const loadLabeledImages = async () => {
    const labels = Object.entries(knownProfiles);
    const descriptors = await Promise.all(
      labels.map(async ([label, profile]) => {
        try {
          if (!profile.image) return null;
          const img = await faceapi.fetchImage(`/known/${profile.image}`);
          const detections = await faceapi
            .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.2 }))
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (!detections) {
            console.warn(`No face detected in reference photo for ${label}`);
            return null;
          }
          return new faceapi.LabeledFaceDescriptors(label, [detections.descriptor]);
        } catch (err) {
          console.warn(`Could not load labeled face image for ${label}:`, err);
          return null;
        }
      })
    );
    return descriptors.filter(Boolean);
  };

  // 3. Custom Canvas HUD Drawing with Smooth Anti-Jitter Coordinates
  const drawHUD = useCallback((canvas, displaySize, faces, drawMesh, detections) => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (drawMesh && detections) {
      try {
        faceapi.draw.drawFaceLandmarks(canvas, detections);
      } catch (err) {
        console.warn("Error drawing landmarks:", err);
      }
    }

    faces.forEach((face) => {
      const box = face.box;
      if (!box) return;

      const isKnown = face.isKnown;
      const mainColor = isKnown ? "#10b981" : "#06b6d4"; // Emerald for known, Cyan for unknown
      const cornerLength = Math.min(22, box.width * 0.2, box.height * 0.2);

      ctx.save();

      // 1. Subtle Bounding Box
      ctx.strokeStyle = isKnown ? "rgba(16, 185, 129, 0.35)" : "rgba(6, 182, 212, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      ctx.setLineDash([]);

      // 2. Futuristic Glowing Corner Accents
      ctx.strokeStyle = mainColor;
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      ctx.shadowColor = mainColor;
      ctx.shadowBlur = 8;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(box.x, box.y + cornerLength);
      ctx.lineTo(box.x, box.y);
      ctx.lineTo(box.x + cornerLength, box.y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(box.x + box.width - cornerLength, box.y);
      ctx.lineTo(box.x + box.width, box.y);
      ctx.lineTo(box.x + box.width, box.y + cornerLength);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(box.x, box.y + box.height - cornerLength);
      ctx.lineTo(box.x, box.y + box.height);
      ctx.lineTo(box.x + cornerLength, box.y + box.height);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(box.x + box.width - cornerLength, box.y + box.height);
      ctx.lineTo(box.x + box.width, box.y + box.height);
      ctx.lineTo(box.x + box.width, box.y + box.height - cornerLength);
      ctx.stroke();

      ctx.shadowBlur = 0;

      // 3. Top Identity Pill Tag
      const idText = isKnown
        ? `${face.name} • ${face.matchScore}%`
        : `${face.name} • Active`;
      ctx.font = "bold 13px 'Inter', sans-serif";
      const idWidth = ctx.measureText(idText).width + 24;
      const tagHeight = 26;
      let tagY = box.y - tagHeight - 8;
      if (tagY < 8) tagY = box.y + 8; // If face is at top edge, draw inside

      // Pill Background
      ctx.fillStyle = "rgba(10, 15, 29, 0.9)";
      ctx.beginPath();
      ctx.roundRect(box.x, tagY, idWidth, tagHeight, 6);
      ctx.fill();
      ctx.strokeStyle = mainColor;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Identity Status Dot
      ctx.fillStyle = mainColor;
      ctx.beginPath();
      ctx.arc(box.x + 12, tagY + tagHeight / 2, 4, 0, Math.PI * 2);
      ctx.fill();

      // Identity Text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(idText, box.x + 22, tagY + 17);

      // 4. Bottom Emotion Badge Tag
      const emoText = `${face.emotionEmoji} ${face.emotionLabel} (${Math.round(face.emotionScore * 100)}%)`;
      ctx.font = "600 12px 'Inter', sans-serif";
      const emoWidth = ctx.measureText(emoText).width + 18;
      const emoHeight = 24;
      let emoY = box.y + box.height + 6;
      if (emoY + emoHeight > displaySize.height - 8) emoY = box.y + box.height - emoHeight - 8;

      ctx.fillStyle = "rgba(10, 15, 29, 0.9)";
      ctx.beginPath();
      ctx.roundRect(box.x, emoY, emoWidth, emoHeight, 6);
      ctx.fill();
      ctx.strokeStyle = EMOTION_MAP[face.dominantEmotion]?.color || "#94a3b8";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.fillStyle = "#f1f5f9";
      ctx.fillText(emoText, box.x + 9, emoY + 16);

      ctx.restore();
    });
  }, []);

  // 4. Real-time Multi-Face Detection Loop (with Temporal Smoothing & Stable Keys)
  const detect = useCallback(async () => {
    if (
      isLivePaused ||
      detectionBusyRef.current ||
      !webcamRef.current ||
      !webcamRef.current.video ||
      webcamRef.current.video.readyState !== 4
    ) {
      return;
    }

    detectionBusyRef.current = true;
    const startTime = performance.now();

    try {
      const video = webcamRef.current.video;
      const displaySize = {
        width: video.videoWidth || 640,
        height: video.videoHeight || 480,
      };

      // ONLY match dimensions if canvas dimensions changed (Prevents continuous canvas wiping)
      const canvas = canvasRef.current;
      if (canvas && (canvas.width !== displaySize.width || canvas.height !== displaySize.height)) {
        canvas.width = displaySize.width;
        canvas.height = displaySize.height;
        faceapi.matchDimensions(canvas, displaySize);
      }

      // Select Detection Model based on user preference
      const detectorOptions =
        detectorModel === "tiny"
          ? new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: minConfidence })
          : new faceapi.SsdMobilenetv1Options({ minConfidence });

      const detections = await faceapi
        .detectAllFaces(video, detectorOptions)
        .withFaceLandmarks()
        .withFaceDescriptors()
        .withFaceExpressions();

      setAnalysisError(false);

      if (detections && detections.length > 0) {
        emptyFrameCountRef.current = 0;
        setFaceDetected(true);

        const faceMatcher = labeledFaceDescriptorsRef.current.length
          ? new faceapi.FaceMatcher(labeledFaceDescriptorsRef.current, 0.58)
          : null;

        const previousTracked = trackedFacesRef.current;

        const faces = detections.map((detection, index) => {
          // Sort all expressions by probability
          const expressionsList = Object.entries(detection.expressions)
            .map(([name, val]) => ({
              name,
              val: Math.max(0, Math.min(1, val)),
              label: EMOTION_MAP[name]?.label || name,
              emoji: EMOTION_MAP[name]?.emoji || "😐",
              color: EMOTION_MAP[name]?.color || "#94a3b8",
            }))
            .sort((a, b) => b.val - a.val);

          const topExpression = expressionsList[0];
          const bestMatch = faceMatcher?.findBestMatch(detection.descriptor);
          const isKnown = Boolean(bestMatch && bestMatch.label !== "unknown");
          const matchScore = isKnown
            ? Math.max(0, Math.min(100, Math.round((1 - bestMatch.distance) * 100)))
            : 0;
          const personInfo = isKnown ? knownProfiles[bestMatch.label] : null;

          const rawBox = detection.detection.box;

          // Find closest previous face for coordinate smoothing (Anti-Jitter)
          const matchedPrev = previousTracked.find((p) => {
            if (isKnown && p.rawLabel === bestMatch.label) return true;
            const dist = Math.hypot(p.box.x - rawBox.x, p.box.y - rawBox.y);
            return dist < 90;
          });

          // Smooth bounding box coordinates using Lerp
          const smoothedBox = matchedPrev
            ? {
                x: lerp(matchedPrev.box.x, rawBox.x, 0.65),
                y: lerp(matchedPrev.box.y, rawBox.y, 0.65),
                width: lerp(matchedPrev.box.width, rawBox.width, 0.65),
                height: lerp(matchedPrev.box.height, rawBox.height, 0.65),
              }
            : rawBox;

          // Stable ID per person to avoid React DOM unmount/remount flashing
          const stableId = isKnown ? `known-${bestMatch.label}` : `visitor-${matchedPrev ? matchedPrev.trackIndex : index}`;

          return {
            id: stableId,
            trackIndex: matchedPrev ? matchedPrev.trackIndex : index,
            rawLabel: bestMatch?.label || "unknown",
            isKnown,
            name: personInfo?.name || (isKnown ? bestMatch.label : `Person ${index + 1}`),
            role: personInfo?.role || "Visitor",
            avatar: personInfo?.image ? (personInfo.image.startsWith("data:") ? personInfo.image : `/known/${personInfo.image}`) : null,
            distance: bestMatch?.distance || 1,
            matchScore,
            dominantEmotion: topExpression.name,
            emotionLabel: topExpression.label,
            emotionEmoji: topExpression.emoji,
            emotionScore: topExpression.val,
            allExpressions: expressionsList,
            box: smoothedBox,
          };
        });

        trackedFacesRef.current = faces;
        setDetectedFaces(faces);
        setLastUpdated(
          new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })
        );

        if (canvasRef.current) {
          const resized = faceapi.resizeResults(detections, displaySize);
          drawHUD(canvasRef.current, displaySize, faces, showLandmarks, resized);
        }
      } else {
        // Debounced empty state (Grace period of 3 consecutive empty frames before disappearing)
        emptyFrameCountRef.current += 1;
        if (emptyFrameCountRef.current >= 3) {
          trackedFacesRef.current = [];
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          }
          setDetectedFaces([]);
          setFaceDetected(false);
        }
      }

      const elapsed = Math.round(performance.now() - startTime);
      setInferenceTime(elapsed);
    } catch (error) {
      console.error("Face detection error:", error);
      setAnalysisError(true);
    } finally {
      detectionBusyRef.current = false;
    }
  }, [isLivePaused, detectorModel, minConfidence, showLandmarks, knownProfiles, drawHUD]);

  // Run detection loop at smooth intervals
  useEffect(() => {
    if (!modelsLoaded) return;
    const interval = setInterval(() => {
      detect();
    }, 150);
    return () => clearInterval(interval);
  }, [modelsLoaded, detect]);

  // 5. Enroll New Person Action
  const handleEnrollPerson = async (e) => {
    e.preventDefault();
    if (!newPersonName.trim()) {
      setEnrollError("Please enter a name for the new profile.");
      return;
    }

    if (!webcamRef.current || !webcamRef.current.video) {
      setEnrollError("Webcam stream is not available.");
      return;
    }

    setIsEnrolling(true);
    setEnrollError("");
    setEnrollSuccess("");

    try {
      const video = webcamRef.current.video;
      // Capture high-res detection
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.25 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) {
        setEnrollError("No clear face detected in the frame. Please look directly at the camera.");
        setIsEnrolling(false);
        return;
      }

      // Capture snapshot image as avatar data URL
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = video.videoWidth || 640;
      tempCanvas.height = video.videoHeight || 480;
      const ctx = tempCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.85);

      const profileKey = newPersonName.trim().toLowerCase().replace(/\s+/g, "_");

      // Register new labeled face descriptor
      const newDescriptor = new faceapi.LabeledFaceDescriptors(profileKey, [detection.descriptor]);
      labeledFaceDescriptorsRef.current = [...labeledFaceDescriptorsRef.current, newDescriptor];

      // Update known profiles
      setKnownProfiles((prev) => ({
        ...prev,
        [profileKey]: {
          name: newPersonName.trim(),
          role: newPersonRole.trim() || "Team Member",
          image: dataUrl,
        },
      }));

      setEnrollSuccess(`Successfully registered ${newPersonName.trim()}!`);
      setNewPersonName("");
      setTimeout(() => {
        setIsEnrollModalOpen(false);
        setEnrollSuccess("");
      }, 1400);
    } catch (err) {
      console.error("Enrollment error:", err);
      setEnrollError("Failed to enroll face. Please try again.");
    } finally {
      setIsEnrolling(false);
    }
  };

  // Overall Group Mood Calculation
  const getOverallMood = () => {
    if (!detectedFaces.length) return null;
    const counts = {};
    detectedFaces.forEach((f) => {
      counts[f.dominantEmotion] = (counts[f.dominantEmotion] || 0) + 1;
    });
    const topMoodKey = Object.entries(counts).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    return EMOTION_MAP[topMoodKey] || { label: topMoodKey, emoji: "✨" };
  };

  const groupMood = getOverallMood();

  const systemState = modelError || cameraError || analysisError
    ? "error"
    : modelsLoaded
    ? isLivePaused
      ? "paused"
      : "ready"
    : "loading";

  const systemLabel = cameraError
    ? "Camera Unavailable"
    : modelError
    ? "Models Unavailable"
    : analysisError
    ? "Analysis Paused"
    : !modelsLoaded
    ? "Loading Vision Models..."
    : isLivePaused
    ? "Analysis Paused"
    : "Vision Engine Active";

  return (
    <div className="app-layout">
      {/* Background Ambient Glow */}
      <div className="ambient-glow glow-1" />
      <div className="ambient-glow glow-2" />

      {/* Header Bar */}
      <header className="main-header">
        <div className="brand-group">
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <div>
            <div className="brand-sub">Neural Biometrics & Emotion Analysis</div>
            <h1 className="brand-title">VisionAI Intelligence</h1>
          </div>
        </div>

        <div className="header-meta">
          {/* Active Faces Count */}
          <div className={`meta-pill ${detectedFaces.length > 0 ? "highlight" : ""}`}>
            <span className="pill-dot pulse" />
            <span>
              {detectedFaces.length === 0
                ? "No Faces in View"
                : `${detectedFaces.length} ${detectedFaces.length === 1 ? "Face" : "Faces"} Active`}
            </span>
          </div>

          {/* Engine Latency */}
          {inferenceTime > 0 && (
            <div className="meta-pill latency">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>{inferenceTime}ms / frame</span>
            </div>
          )}

          {/* System Status */}
          <div className={`system-badge ${systemState}`}>
            <span className="status-indicator" />
            <span>{systemLabel}</span>
          </div>
        </div>
      </header>

      {/* Main Grid Content */}
      <main className="dashboard-grid">
        {/* Left Column: Live Camera & HUD Viewport */}
        <section className="camera-section">
          <div className="section-card camera-card">
            {/* Camera Toolbar */}
            <div className="camera-toolbar">
              <div className="toolbar-left">
                <span className="live-indicator">
                  <span className="live-bullet" />
                  LIVE
                </span>
                <span className="feed-title">Primary Camera Feed</span>
              </div>

              <div className="toolbar-controls">
                {/* AI Model Selector */}
                <div className="control-item">
                  <label htmlFor="model-select" className="control-label">
                    <span>Model</span>
                  </label>
                  <select
                    id="model-select"
                    className="select-input"
                    value={detectorModel}
                    onChange={(e) => setDetectorModel(e.target.value)}
                  >
                    <option value="tiny">⚡ TinyFace (Fast)</option>
                    <option value="ssd">🎯 SSD MobileNet</option>
                  </select>
                </div>

                {/* Virtual Background Picker */}
                <div className="control-item">
                  <label htmlFor="bg-select" className="control-label">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                      <circle cx="9" cy="9" r="2" />
                      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                    </svg>
                    <span>Background</span>
                  </label>
                  <select
                    id="bg-select"
                    className="select-input"
                    value={backgroundMode}
                    onChange={(e) => setBackgroundMode(e.target.value)}
                  >
                    {virtualBackgrounds.map((bg) => (
                      <option key={bg.id} value={bg.id}>
                        {bg.icon} {bg.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Face Mesh / Landmark Toggle */}
                <button
                  type="button"
                  className={`action-btn ${showLandmarks ? "active" : ""}`}
                  onClick={() => setShowLandmarks((prev) => !prev)}
                  title="Toggle Facial Landmark Mesh"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="1" />
                    <circle cx="12" cy="5" r="1" />
                    <circle cx="12" cy="19" r="1" />
                    <circle cx="5" cy="12" r="1" />
                    <circle cx="19" cy="12" r="1" />
                    <circle cx="7" cy="7" r="1" />
                    <circle cx="17" cy="17" r="1" />
                    <circle cx="7" cy="17" r="1" />
                    <circle cx="17" cy="7" r="1" />
                  </svg>
                  <span>Mesh</span>
                </button>

                {/* Pause / Resume Button */}
                <button
                  type="button"
                  className={`action-btn ${isLivePaused ? "active-warning" : ""}`}
                  onClick={() => setIsLivePaused((prev) => !prev)}
                  title={isLivePaused ? "Resume Live Tracking" : "Pause Tracking"}
                >
                  {isLivePaused ? (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      <span>Resume</span>
                    </>
                  ) : (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <rect x="6" y="4" width="4" height="16" />
                        <rect x="14" y="4" width="4" height="16" />
                      </svg>
                      <span>Pause</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Video Viewport & Canvas Layers */}
            <div className="viewport-container">
              <Webcam
                ref={webcamRef}
                audio={false}
                videoConstraints={videoConstraints}
                className={`camera-video ${backgroundMode !== "none" ? "is-hidden" : ""}`}
                onUserMedia={() => setCameraError(false)}
                onUserMediaError={() => setCameraError(true)}
              />
              <canvas
                ref={backgroundCanvasRef}
                className={`background-canvas ${backgroundMode !== "none" ? "is-visible" : ""}`}
                aria-hidden="true"
              />
              <canvas
                ref={canvasRef}
                width={videoConstraints.width}
                height={videoConstraints.height}
                className="hud-overlay-canvas"
              />
              <canvas ref={foregroundCanvasRef} className="foreground-canvas" aria-hidden="true" />

              {/* Viewport HUD Crosshairs Overlay */}
              <div className="hud-corner top-left" />
              <div className="hud-corner top-right" />
              <div className="hud-corner bottom-left" />
              <div className="hud-corner bottom-right" />

              {/* Scanning Laser Animation */}
              <div className={`scan-line ${detectedFaces.length > 0 ? "active" : ""}`} />

              {/* In-Frame Empty State Overlay */}
              {!cameraError && !faceDetected && modelsLoaded && !isLivePaused && (
                <div className="viewport-watermark">
                  <div className="watermark-box">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <path d="M7 3H5a2 2 0 0 0-2 2v2M21 7V5a2 2 0 0 0-2-2h-2M7 21H5a2 2 0 0 1-2-2v-2M21 17v2a2 2 0 0 1-2 2h-2" />
                      <circle cx="12" cy="12" r="4" />
                    </svg>
                    <span>Scanning for Multiple Faces...</span>
                    <small>Position face within frame for real-time tracking</small>
                  </div>
                </div>
              )}

              {/* Error Watermark */}
              {(cameraError || modelError || analysisError) && (
                <div className="viewport-watermark error-state">
                  <div className="watermark-box">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <span className="error-title">
                      {cameraError
                        ? "Camera Permission Denied"
                        : modelError
                        ? "AI Models Failed to Load"
                        : "Analysis Interrupted"}
                    </span>
                    <small>
                      {cameraError
                        ? "Please grant webcam permissions in your browser and reload."
                        : "Check network connection or refresh the page."}
                    </small>
                  </div>
                </div>
              )}
            </div>

            {/* Camera Footer Info Bar */}
            <div className="camera-footer">
              <div className="footer-left">
                <span className="footer-label">Detector:</span>
                <span className="footer-val">
                  {detectorModel === "tiny" ? "TinyFace v2 (SIMD)" : "SSD MobileNet v1"} + 68 Landmarks
                </span>
                <span className="divider">•</span>
                <span className="footer-label">Threshold:</span>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.05"
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                  className="sensitivity-slider"
                  title={`Detection threshold: ${Math.round(minConfidence * 100)}%`}
                />
                <span className="footer-val">{Math.round(minConfidence * 100)}%</span>
              </div>

              <div className="footer-right">
                <span className="footer-label">Last Frame:</span>
                <span className="footer-val">{lastUpdated || "Awaiting signal"}</span>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Multi-Face Analysis & Insights Dashboard */}
        <aside className="insights-section">
          {/* Navigation Tabs */}
          <div className="tabs-header">
            <button
              type="button"
              className={`tab-btn ${activeTab === "insights" ? "active" : ""}`}
              onClick={() => setActiveTab("insights")}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
              <span>Live Analysis ({detectedFaces.length})</span>
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "enrolled" ? "active" : ""}`}
              onClick={() => setActiveTab("enrolled")}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <line x1="19" y1="8" x2="19" y2="14" />
                <line x1="22" y1="11" x2="16" y2="11" />
              </svg>
              <span>Team Roster ({Object.keys(knownProfiles).length})</span>
            </button>
          </div>

          {/* TAB 1: Live Multi-Face Insights */}
          {activeTab === "insights" && (
            <div className="tab-content">
              {/* Overall Mood Summary Banner */}
              <div className="overview-card">
                <div className="overview-header">
                  <span className="overview-tag">Group Analysis</span>
                  <span className="overview-count">
                    {detectedFaces.length === 0
                      ? "Standby"
                      : `${detectedFaces.length} ${detectedFaces.length === 1 ? "Subject" : "Subjects"} Tracked`}
                  </span>
                </div>

                <div className="overview-body">
                  <div className="mood-display">
                    <span className="mood-emoji">{groupMood ? groupMood.emoji : "🔍"}</span>
                    <div>
                      <div className="mood-title">
                        {groupMood ? `Primary Emotion: ${groupMood.label}` : "Waiting for Subject"}
                      </div>
                      <p className="mood-desc">
                        {detectedFaces.length > 0
                          ? `Real-time biometrics active for ${detectedFaces.length} person(s) in frame.`
                          : "Position one or more faces in front of the camera for real-time identity & emotion tracking."}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Individual Face Cards */}
              <div className="faces-container">
                {detectedFaces.length > 0 ? (
                  detectedFaces.map((face, index) => (
                    <article className="face-card" key={face.id}>
                      {/* Face Card Header */}
                      <div className="face-card-header">
                        <div className="face-avatar-wrap">
                          {face.avatar ? (
                            <img src={face.avatar} alt={face.name} className="face-avatar" />
                          ) : (
                            <div className="face-avatar placeholder">
                              {face.isKnown ? face.name.charAt(0) : `#${index + 1}`}
                            </div>
                          )}
                          <span className={`avatar-status ${face.isKnown ? "known" : "unknown"}`} />
                        </div>

                        <div className="face-info">
                          <div className="face-title-row">
                            <h3 className="face-name-heading">{face.name}</h3>
                            <span className={`badge-pill ${face.isKnown ? "badge-known" : "badge-unknown"}`}>
                              {face.isKnown ? `Match ${face.matchScore}%` : "Visitor"}
                            </span>
                          </div>
                          <span className="face-role-text">{face.role}</span>
                        </div>
                      </div>

                      {/* Primary Emotion Banner */}
                      <div className="dominant-emotion-row">
                        <span
                          className="emotion-icon-pill"
                          style={{
                            backgroundColor: `${EMOTION_MAP[face.dominantEmotion]?.color}22`,
                            color: EMOTION_MAP[face.dominantEmotion]?.color,
                          }}
                        >
                          {face.emotionEmoji} {face.emotionLabel}
                        </span>
                        <span className="emotion-confidence-text">
                          {Math.round(face.emotionScore * 100)}% confidence
                        </span>
                      </div>

                      {/* Emotion Spectrum Progress Bars */}
                      <div className="emotion-spectrum">
                        {face.allExpressions.slice(0, 4).map((expr) => (
                          <div className="spectrum-item" key={expr.name}>
                            <div className="spectrum-label-row">
                              <span className="spectrum-label">
                                {expr.emoji} {expr.label}
                              </span>
                              <span className="spectrum-val">{Math.round(expr.val * 100)}%</span>
                            </div>
                            <div className="progress-track">
                              <div
                                className="progress-fill"
                                style={{
                                  width: `${Math.round(expr.val * 100)}%`,
                                  backgroundColor: expr.color,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </article>
                  ))
                ) : (
                  <div className="empty-state-box">
                    <div className="empty-icon-wrap">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                        <circle cx="12" cy="7" r="4" />
                      </svg>
                    </div>
                    <h4>No Detected Faces</h4>
                    <p>All active subjects in the frame will be recognized, tracked, and analyzed simultaneously.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: Enrolled Team Directory */}
          {activeTab === "enrolled" && (
            <div className="tab-content roster-tab">
              <div className="roster-header-info">
                <div className="roster-title-row">
                  <div>
                    <h4>Enrolled Biometric Profiles</h4>
                    <p>Registered team members for live facial identification.</p>
                  </div>
                  <button
                    type="button"
                    className="enroll-btn"
                    onClick={() => {
                      setEnrollError("");
                      setEnrollSuccess("");
                      setIsEnrollModalOpen(true);
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    <span>Enroll Face</span>
                  </button>
                </div>
              </div>

              <div className="roster-list">
                {Object.entries(knownProfiles).map(([key, person]) => {
                  const isPresent = detectedFaces.some(
                    (f) => f.rawLabel.toLowerCase() === key.toLowerCase()
                  );
                  return (
                    <div className={`roster-item ${isPresent ? "is-present" : ""}`} key={key}>
                      <img
                        src={person.image.startsWith("data:") ? person.image : `/known/${person.image}`}
                        alt={person.name}
                        className="roster-avatar"
                      />
                      <div className="roster-meta">
                        <div className="roster-name-row">
                          <span className="roster-name">{person.name}</span>
                          <span className={`presence-badge ${isPresent ? "present" : "away"}`}>
                            {isPresent ? "In Frame" : "Away"}
                          </span>
                        </div>
                        <span className="roster-role">{person.role}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Privacy & Engine Information Card */}
          <div className="privacy-badge">
            <div className="privacy-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
            </div>
            <p>
              <strong>Edge AI Processing:</strong> All facial landmark computation, descriptor matching, and emotion inference run 100% locally on your browser using WebGL & SIMD.
            </p>
          </div>
        </aside>
      </main>

      {/* ENROLL FACE MODAL */}
      {isEnrollModalOpen && (
        <div className="modal-backdrop" onClick={() => !isEnrolling && setIsEnrollModalOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <div className="modal-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                    <circle cx="9" cy="7" r="4" />
                    <line x1="19" y1="8" x2="19" y2="14" />
                    <line x1="22" y1="11" x2="16" y2="11" />
                  </svg>
                </div>
                <h3>Enroll New Person</h3>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => !isEnrolling && setIsEnrollModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleEnrollPerson} className="modal-form">
              <p className="modal-instruction">
                Position your face directly in front of the camera and enter your details to register your biometric profile in real-time.
              </p>

              <div className="form-group">
                <label htmlFor="person-name">Full Name</label>
                <input
                  id="person-name"
                  type="text"
                  className="modal-input"
                  placeholder="e.g. Sarah Connor"
                  value={newPersonName}
                  onChange={(e) => setNewPersonName(e.target.value)}
                  disabled={isEnrolling}
                  autoFocus
                />
              </div>

              <div className="form-group">
                <label htmlFor="person-role">Designation / Role</label>
                <input
                  id="person-role"
                  type="text"
                  className="modal-input"
                  placeholder="e.g. Lead Engineer"
                  value={newPersonRole}
                  onChange={(e) => setNewPersonRole(e.target.value)}
                  disabled={isEnrolling}
                />
              </div>

              {enrollError && <div className="modal-alert error">{enrollError}</div>}
              {enrollSuccess && <div className="modal-alert success">{enrollSuccess}</div>}

              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setIsEnrollModalOpen(false)}
                  disabled={isEnrolling}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary" disabled={isEnrolling}>
                  {isEnrolling ? (
                    <>
                      <span className="spinner" />
                      <span>Capturing Face...</span>
                    </>
                  ) : (
                    <>
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <circle cx="12" cy="12" r="10" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      <span>Capture & Register</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
