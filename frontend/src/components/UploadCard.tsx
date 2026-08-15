"use client";

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Upload, CheckCircle2, AlertCircle, Loader2, Download, RefreshCw } from "lucide-react";

interface UploadResponse {
  message: string;
  scan_id: string;
  status: string;
}

interface StatusResponse {
  scan_id: string;
  status: string;
  file_type: string;
}

interface ResultsResponse {
  scan_id: string;
  mask_url: string;
  xai_url: string;
  report_url: string;
}

export default function UploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [scanId, setScanId] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<string | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMessage(null);
      setScanId(null);
      setScanStatus(null);
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMessage("Select an MRI scan file to proceed.");
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post<UploadResponse>(
        "http://127.0.0.1:8000/api/v1/scans/upload",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setScanId(response.data.scan_id);
      setScanStatus("PROCESSING");
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        setErrorMessage(String(err.response.data.detail));
      } else {
        setErrorMessage("Failed to reach FastAPI backend on port 8000.");
      }
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    if (!scanId || scanStatus === "SEGMENTED" || scanStatus === "FAILED") return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get<StatusResponse>(
          `http://127.0.0.1:8000/api/v1/scans/status/${scanId}`
        );
        setScanStatus(res.data.status);
        
        if (res.data.status === "SEGMENTED") {
          clearInterval(interval);
          const resultsRes = await axios.get<ResultsResponse>(
            `http://127.0.0.1:8000/api/v1/scans/results/${scanId}`
          );
          setResults(resultsRes.data);
        } else if (res.data.status === "FAILED") {
          clearInterval(interval);
          setErrorMessage("AI Pipeline processing failed.");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [scanId, scanStatus]);

  const resetState = () => {
    setFile(null);
    setScanId(null);
    setScanStatus(null);
    setResults(null);
    setErrorMessage(null);
  };

  return (
    <div className="w-full max-w-2xl p-8 rounded-2xl bg-[#0f0f11]/60 backdrop-blur-xl border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] z-10 text-white">
      <div className="flex items-center space-x-3 mb-6">
        <div className={`h-3 w-3 rounded-full ${scanStatus === "SEGMENTED" ? "bg-green-400" : "bg-[#00f0ff] animate-pulse"}`} />
        <h2 className="text-xl font-semibold tracking-wide text-white">
          Neural Diagnostics Dashboard
        </h2>
      </div>

      {!scanId ? (
        <>
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-white/20 hover:border-[#00f0ff]/60 transition-colors duration-200 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-white/[0.02]"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.nii,.gz"
              className="hidden"
              onChange={handleFileSelect}
            />
            <Upload className="h-10 w-10 text-[#00f0ff] mb-3 opacity-80" />
            <p className="text-sm font-medium text-gray-300">
              {file ? file.name : "Click to select or drag and drop an MRI slice"}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Supports Kaggle 2D Slices (.jpg) & BraTS NIfTI (.nii)
            </p>
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || isUploading}
            className="mt-6 w-full py-3 px-4 rounded-xl bg-gradient-to-r from-[#00f0ff] to-[#7000ff] text-black font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center space-x-2 cursor-pointer"
          >
            {isUploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-black" />
                <span>Uploading to MinIO...</span>
              </>
            ) : (
              <span>Execute ARMT-GAN Pipeline</span>
            )}
          </button>
        </>
      ) : (
        <div className="space-y-6">
          <div className="bg-white/[0.03] p-4 rounded-xl border border-white/10 space-y-2">
            <div className="flex justify-between text-xs text-gray-400">
              <span>Scan ID:</span>
              <span className="font-mono text-gray-200">{scanId}</span>
            </div>
            <div className="flex justify-between text-xs text-gray-400">
              <span>Pipeline Status:</span>
              <span className={`font-semibold ${scanStatus === "SEGMENTED" ? "text-green-400" : "text-[#00f0ff] animate-pulse"}`}>
                {scanStatus}
              </span>
            </div>
          </div>

          {scanStatus === "PROCESSING" && (
            <div className="flex flex-col items-center justify-center py-12 space-y-3">
              <Loader2 className="h-10 w-10 animate-spin text-[#00f0ff]" />
              <p className="text-sm text-gray-300">ARMT-GAN analyzing tensors & generating XAI heatmaps...</p>
            </div>
          )}

          {scanStatus === "SEGMENTED" && results && (
            <div className="space-y-6">
              <div className="flex items-center space-x-2 text-green-400 text-sm bg-green-500/10 p-3 rounded-lg border border-green-500/20">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>Clinical Segmentation & Report Compiled Successfully!</span>
              </div>

              {/* Improved Visual Asset Previews */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/40 p-4 rounded-xl border border-white/10 flex flex-col items-center">
                  <span className="text-xs text-gray-300 mb-3 font-semibold tracking-wide">Isolated Tumor Mask</span>
                  <div className="w-full aspect-square relative rounded-xl overflow-hidden border border-white/10 bg-black flex items-center justify-center">
                    <img 
                      src={results.mask_url} 
                      alt="Tumor Mask" 
                      className="object-contain w-full h-full p-2"
                    />
                  </div>
                </div>

                <div className="bg-black/40 p-4 rounded-xl border border-white/10 flex flex-col items-center">
                  <span className="text-xs text-gray-300 mb-3 font-semibold tracking-wide">XAI Activation Map</span>
                  <div className="w-full aspect-square relative rounded-xl overflow-hidden border border-white/10 bg-black flex items-center justify-center">
                    <img 
                      src={results.xai_url} 
                      alt="XAI Heatmap" 
                      className="object-contain w-full h-full p-2"
                    />
                  </div>
                </div>
              </div>

              {/* PDF Download Button */}
              <a
                href={results.report_url}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-[#00f0ff] to-[#7000ff] text-black font-semibold text-sm hover:opacity-90 transition-opacity flex items-center justify-center space-x-2 cursor-pointer shadow-neon"
              >
                <Download className="h-4 w-4" />
                <span>Download Clinical PDF Report</span>
              </a>

              <button
                onClick={resetState}
                className="w-full py-3 px-4 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm transition-colors cursor-pointer flex items-center justify-center space-x-2"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Analyze Another Scan</span>
              </button>
            </div>
          )}
        </div>
      )}

      {errorMessage && (
        <div className="mt-4 flex items-center space-x-2 text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}