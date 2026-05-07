import { useState, useRef, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const TABS = ["Video Tracking", "Frame Analysis"];

export default function App() {
  const [activeTab, setActiveTab]     = useState(0);
  const [dragging,  setDragging]      = useState(false);
  const [file,      setFile]          = useState(null);
  const [preview,   setPreview]       = useState(null);
  const [loading,   setLoading]       = useState(false);
  const [progress,  setProgress]      = useState("");
  const [result,    setResult]        = useState(null);
  const [frameData, setFrameData]     = useState(null);
  const [error,     setError]         = useState(null);
  const fileRef = useRef();

  const reset = () => {
    setFile(null); setPreview(null); setResult(null);
    setFrameData(null); setError(null); setProgress("");
  };

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [activeTab]);

  const handleFile = (f) => {
    reset();
    setFile(f);
    if (activeTab === 1 && f.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(f));
    }
  };

  const trackVideo = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    setProgress("Uploading video...");
    try {
      const fd = new FormData();
      fd.append("file", file);
      setProgress("Processing frames — this may take a minute...");
      const res = await fetch(`${API_URL}/track-video`, {
        method: "POST", body: fd
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      setResult({
        url,
        frames:  res.headers.get("X-Frames-Processed"),
        time:    res.headers.get("X-Processing-Time"),
        tracks:  res.headers.get("X-Total-Tracks"),
      });
      setProgress("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const trackFrame = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    setProgress("Analyzing frame...");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_URL}/track-frame`, {
        method: "POST", body: fd
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setFrameData(data);
      setProgress("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🚗</span>
        <div>
          <h1 className="text-xl font-bold text-white">Vehicle Tracker</h1>
          <p className="text-xs text-gray-400">
            Deformable DETR + Re-ID Attention Head
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          {TABS.map((t, i) => (
            <button key={i}
              onClick={() => { setActiveTab(i); reset(); }}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition
                ${activeTab === i
                  ? "bg-green-500 text-black"
                  : "bg-gray-800 text-gray-300 hover:bg-gray-700"}`}>
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">

        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current.click()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center
            cursor-pointer transition
            ${dragging
              ? "border-green-400 bg-green-900/20"
              : "border-gray-700 hover:border-gray-500 bg-gray-900"}`}>
          <input
            ref={fileRef} type="file" className="hidden"
            accept={activeTab === 0
              ? "video/mp4,video/avi,video/mov"
              : "image/*"}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className="text-5xl mb-4">
            {activeTab === 0 ? "🎥" : "🖼️"}
          </div>
          {file ? (
            <div>
              <p className="text-green-400 font-semibold">{file.name}</p>
              <p className="text-gray-400 text-sm mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div>
              <p className="text-gray-300 font-medium">
                Drop {activeTab === 0 ? "a video" : "an image"} here
              </p>
              <p className="text-gray-500 text-sm mt-1">
                {activeTab === 0
                  ? "MP4, AVI, MOV supported"
                  : "JPG, PNG supported"}
              </p>
            </div>
          )}
        </div>

        {/* Image Preview */}
        {preview && (
          <div className="rounded-xl overflow-hidden border border-gray-700">
            <img src={preview} alt="preview"
              className="w-full max-h-64 object-contain bg-black" />
          </div>
        )}

        {/* Action Button */}
        {file && !loading && !result && !frameData && (
          <button
            onClick={activeTab === 0 ? trackVideo : trackFrame}
            className="w-full py-3 rounded-xl bg-green-500 hover:bg-green-400
              text-black font-bold text-lg transition">
            {activeTab === 0 ? "🚗 Track Vehicles" : "🔍 Analyze Frame"}
          </button>
        )}

        {/* Loading */}
        {loading && (
          <div className="bg-gray-900 rounded-xl p-6 text-center space-y-3">
            <div className="flex justify-center">
              <div className="w-10 h-10 border-4 border-green-400
                border-t-transparent rounded-full animate-spin"/>
            </div>
            <p className="text-green-400 font-medium">{progress}</p>
            <p className="text-gray-500 text-sm">
              Large videos may take 1–3 minutes depending on server
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-700
            rounded-xl p-4 text-red-300 text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* Video Result */}
        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              {[
                ["🎞️ Frames", result.frames],
                ["⏱️ Time",   result.time],
                ["🚗 Tracks", result.tracks],
              ].map(([label, val]) => (
                <div key={label}
                  className="bg-gray-900 rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-green-400">{val}</p>
                  <p className="text-gray-400 text-sm mt-1">{label}</p>
                </div>
              ))}
            </div>
            <video controls className="w-full rounded-xl border border-gray-700">
              <source src={result.url} type="video/mp4" />
            </video>
            <div className="flex gap-3">
              <a href={result.url} download="tracked_output.mp4"
                className="flex-1 py-3 rounded-xl bg-green-500
                  hover:bg-green-400 text-black font-bold text-center
                  transition block">
                ⬇️ Download Tracked Video
              </a>
              <button onClick={reset}
                className="px-6 py-3 rounded-xl bg-gray-800
                  hover:bg-gray-700 text-gray-300 font-medium transition">
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Frame Analysis Result */}
        {frameData && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded-xl p-4 text-center">
                <p className="text-2xl font-bold text-green-400">
                  {frameData.detections}
                </p>
                <p className="text-gray-400 text-sm">Vehicles Detected</p>
              </div>
              <div className="bg-gray-900 rounded-xl p-4 text-center">
                <p className="text-2xl font-bold text-green-400">
                  {frameData.frame_id}
                </p>
                <p className="text-gray-400 text-sm">Frame ID</p>
              </div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 space-y-2">
              <p className="text-gray-400 text-sm font-medium mb-3">
                Track Details
              </p>
              {frameData.tracks.map((t) => (
                <div key={t.track_id}
                  className="flex justify-between items-center
                    bg-gray-800 rounded-lg px-4 py-2 text-sm">
                  <span className="text-green-400 font-bold">
                    ID:{t.track_id}
                  </span>
                  <span className="text-gray-300">
                    Age: {t.age}
                  </span>
                  <span className="text-gray-400 font-mono text-xs">
                    [{t.box.join(", ")}]
                  </span>
                </div>
              ))}
            </div>
            <button onClick={reset}
              className="w-full py-3 rounded-xl bg-gray-800
                hover:bg-gray-700 text-gray-300 font-medium transition">
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
