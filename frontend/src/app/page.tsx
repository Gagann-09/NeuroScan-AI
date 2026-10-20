"use client"

import { useState, useRef, useEffect, useCallback, useMemo, Fragment } from "react"
import { BrainCircuit, AlertCircle, LogOut, Activity } from "lucide-react"
import { BrainNodes } from "@/components/brain-nodes"
import { UploadPill } from "@/components/upload-pill"
import { KaggleUploadPill } from "@/components/kaggle-upload"
import { ProcessingView } from "@/components/processing-view"
import { ResultsDashboard, KaggleView } from "@/components/results-dashboard"
import { MOCK_KAGGLE_RESULTS, type ScanResults, type KaggleResults } from "@/lib/mock-data"
import { createApiClient } from "@/lib/api"
import { useAuth } from "@/lib/auth-context"
import { auth } from "@/lib/firebase"
import { signOut } from "firebase/auth"

type Status = "idle" | "processing" | "segmented" | "kaggle_idle" | "kaggle_processing" | "kaggle_completed"
type Workflow = "brats" | "kaggle"

export default function Dashboard() {
  const [workflow, setWorkflow] = useState<Workflow>("brats")
  const [status, setStatus] = useState<Status>("idle")
  const [results, setResults] = useState<ScanResults | null>(null)
  const [kaggleResults, setKaggleResults] = useState<KaggleResults | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const { user, getIdToken, logout } = useAuth()

  // Create authenticated API client
  const api = useMemo(
    () =>
      createApiClient(
        getIdToken,
        () => {
          // On 401: clear local state and logout
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setResults(null)
          setKaggleResults(null)
          setPipelineError("Session expired. Please log in again.")
          setStatus("idle")
          setWorkflow("brats")
          logout()
        }
      ),
    [getIdToken, logout]
  )

  // Centralized scan result loader
  const loadScanResults = useCallback(async (scan_id: string) => {
    try {
      const rawApiResults = await api.getScanResults(scan_id)
      const apiResults = rawApiResults as any

      const maxProb = apiResults.max_tumor_probability != null
        ? Math.round(apiResults.max_tumor_probability * 1000) / 10
        : null

      setResults({
        scanId: scan_id,
        confidence: maxProb,
        anomalyArea: null,
        predictedMaskPixels: apiResults.predicted_mask_pixels ?? null,
        sliceCoveragePercent: apiResults.slice_coverage_percent ?? null,
        anomalyPills: 0,
        tumorType: "Not determined",
        tumorTypeSource: "ARMT-GAN segmentation does not classify tumor type",
        tumorGrade: "Not determined",
        tumorGradeSource: "WHO grade/stage requires separate validated evidence",
        volume: null,
        dicom: [],
        confidencePills: maxProb != null ? Math.min(7, Math.max(1, Math.round((maxProb / 100) * 7))) : 0,
        maskUrl: apiResults.mask_url || "",
        reportUrl: apiResults.report_url || "",
        mask_url: apiResults.mask_url,
        xai_url: apiResults.xai_url,
        report_url: apiResults.report_url,
        classProbabilities: [
          {
            label: "Peak tumor probability (raw sigmoid)",
            value: maxProb,
            accent: "lime",
          },
          {
            label: "Enhancing tumor (ET)",
            value: null,
            accent: "muted",
          },
          {
            label: "Peritumoral edema (ED)",
            value: null,
            accent: "muted",
          },
          {
            label: "Necrotic core (NCR)",
            value: null,
            accent: "muted",
          },
        ],
        telemetry: [
          { label: "Inference latency", value: "N/A", sub: "Per-scan timing is not recorded" },
          { label: "Model parameters", value: "N/A", sub: "Not verified from the loaded model" },
          { label: "Benchmark Dice", value: "N/A", sub: "See the separately evaluated test-set results" },
        ],
        patient: [
          { label: "Patient ID", value: "PT-ANONYMIZED" },
          { label: "Study ID", value: scan_id },
          { label: "Modality", value: "MRI (BraTS)" },
        ],
      })
      window.history.pushState({ neuroscanView: "segmented" }, "", "/")
      setStatus("segmented")
    } catch (err: any) {
      setPipelineError(err.message || "Failed to fetch segmentation results from backend.")
      setStatus("idle")
    }
  }, [api])

  // ── Kaggle Classification Pipeline ────────────────────────────────────────

  const loadKaggleResults = useCallback(async (scan_id: string) => {
    try {
      const rawApiResults = await api.getKaggleScanResults(scan_id)
      const apiResults = rawApiResults as any

      const className = apiResults.class_name || "Unknown"
      const classIndex = apiResults.class_index ?? 0
      const probabilities = apiResults.probabilities || { glioma: 0, meningioma: 0, notumor: 0, pituitary: 0 }
      const modelVersion = apiResults.model_version || "kaggle-cnn-v1.0"
      const imageUrl = apiResults.image_url || ""

      setKaggleResults({
        scanId: scan_id,
        class_name: className,
        class_index: classIndex,
        probabilities,
        model_version: modelVersion,
        image_url: imageUrl,
      })
      window.history.pushState({ neuroscanView: "kaggle_classified" }, "", "/")
      setStatus("kaggle_completed")
    } catch (err: any) {
      setPipelineError(err.message || "Failed to fetch classification results from backend.")
      setStatus("kaggle_idle")
    }
  }, [api])

  const handleKaggleAnalyze = async (file: File) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setPipelineError(null)
    setKaggleResults(null)
    setStatus("kaggle_processing")

    let scan_id: string

    // ── Step 1: Upload Image ──────────────────────────────────────────────────
    try {
      const uploadResponse = await api.uploadKaggleScan(file)
      scan_id = uploadResponse.scan_id
    } catch (err: any) {
      setPipelineError(err.message || "Failed to upload image. Ensure the backend server is running on port 8000.")
      setStatus("kaggle_idle")
      return
    }

    // ── Step 2: Poll Status ──────────────────────────────────────────────────
    intervalRef.current = setInterval(async () => {
      try {
        const { status: backendStatus } = await api.getKaggleScanStatus(scan_id)

        if (backendStatus === "FAILED") {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setPipelineError("Image classification failed during processing.")
          setStatus("kaggle_idle")
          return
        }

        if (backendStatus === "COMPLETED") {
          // Stop polling immediately
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          await loadKaggleResults(scan_id)
        }
      } catch (err: any) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        if (!err.message?.includes("Session expired")) {
          setPipelineError("Lost connection to inference backend during processing.")
          setStatus("kaggle_idle")
        }
      }
    }, 2000)
  }

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  // Handle browser back button: if viewing results, cleanly navigate back to idle scan screen
  useEffect(() => {
    const handlePopState = () => {
      if (status === "segmented") {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        setResults(null)
        setPipelineError(null)
        setStatus("idle")
      }
      if (status === "kaggle_completed") {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        setKaggleResults(null)
        setPipelineError(null)
        setStatus("kaggle_idle")
      }
    }
    window.addEventListener("popstate", handlePopState)
    return () => window.removeEventListener("popstate", handlePopState)
  }, [status])

  // Support direct scan loading via query param for authenticated verification
  useEffect(() => {
    if (typeof window !== "undefined" && user && (status === "idle" || status === "kaggle_idle")) {
      const urlScanId = new URLSearchParams(window.location.search).get("scan_id")
      if (urlScanId) {
        loadScanResults(urlScanId)
      }
    }
  }, [user, status, loadScanResults])

  // ── Helper Functions ────────────────────────────────────────────────────

  const resetDashboard = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setResults(null)
    setKaggleResults(null)
    setPipelineError(null)
    setStatus("idle")
    setWorkflow("brats")
  }

  const resetKaggleDashboard = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setKaggleResults(null)
    setPipelineError(null)
    setStatus("kaggle_idle")
    setWorkflow("kaggle")
  }

  const handleSignOut = async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setResults(null)
    setKaggleResults(null)
    setStatus("idle")
    await logout()
  }

  /**
   * ARMT-GAN Pipeline Coordination:
   * 1. POST file → /api/v1/scans/upload → capture scan_id
   * 2. Poll GET /api/v1/scans/status/{scan_id}
   * 3. On SEGMENTED → GET /api/v1/scans/results/{scan_id} → hydrate results
   */
  const handleAnalyze = async (file: Record<"t1" | "t1ce" | "t2" | "flair", File>) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setPipelineError(null)
    setResults(null)
    setStatus("processing")

    let scan_id: string

    // ── Step 1: Upload Scan ──────────────────────────────────────────────────
    try {
      const uploadResponse = await api.uploadScan(file)
      scan_id = uploadResponse.scan_id
    } catch (err: any) {
      setPipelineError(err.message || "Failed to upload scan. Ensure the backend server is running on port 8000.")
      setStatus("idle")
      return
    }

    // ── Step 2: Poll Status ──────────────────────────────────────────────────
    intervalRef.current = setInterval(async () => {
      try {
        const { status: backendStatus } = await api.getScanStatus(scan_id)

        if (backendStatus === "FAILED") {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          setPipelineError("Scan analysis failed during processing.")
          setStatus("idle")
          return
        }

        if (backendStatus === "SEGMENTED") {
          // Stop polling immediately
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
          await loadScanResults(scan_id)
        }
      } catch (err: any) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        // Check if error is auth-related (already handled by onUnauthorized)
        if (!err.message?.includes("Session expired")) {
          setPipelineError("Lost connection to inference backend during processing.")
          setStatus("idle")
        }
      }
    }, 2000)
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col">
      <BrainNodes />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[50vw] h-[50vw] bg-neuro-cyan/5 blur-[120px] rounded-full pointer-events-none z-0" />

      {status !== "segmented" && status !== "kaggle_completed" && (
        <header className="w-full flex items-center justify-between px-6 sm:px-10 py-6 z-40 animate-in fade-in">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-md">
              <BrainCircuit className="h-5 w-5 text-neuro-cyan" />
            </div>
            <span className="text-xl font-semibold tracking-wide text-white">
              NeuroScan<span className="text-neuro-cyan">AI</span>
            </span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs text-white/40 font-mono tracking-widest hidden sm:inline">
              ARMT-GAN · v2.4 · LIVE
            </span>
            {user && (
              <button
                onClick={handleSignOut}
                title="Disconnect Session"
                className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs text-white/60 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span className="hidden md:inline">{user.email?.split("@")[0]}</span>
              </button>
            )}
          </div>
        </header>
      )}

      <main className="flex-1 flex flex-col items-center justify-center z-30 relative px-4 w-full pb-16">
        {/* Workflow selector */}
        <div className="w-full max-w-3xl mb-6 animate-in fade-in">
          <div className="flex bg-white/5 border border-white/10 rounded-full p-1">
            <button
              onClick={() => {
                if (status === "idle" || status === "kaggle_idle") {
                  setWorkflow("brats")
                  setStatus("idle")
                }
              }}
              disabled={status === "processing" || status === "kaggle_processing" || status === "segmented" || status === "kaggle_completed"}
              className={`flex-1 py-2 px-4 rounded-full text-sm font-medium transition-all cursor-pointer ${
                workflow === "brats"
                  ? "bg-white/10 text-white"
                  : "text-white/40 hover:text-white hover:bg-white/5"
              } ${status === "processing" || status === "kaggle_processing" || status === "segmented" || status === "kaggle_completed" ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <BrainCircuit className="h-4 w-4 inline mr-1" /> BraTS Segmentation
            </button>
            <button
              onClick={() => {
                if (status === "idle" || status === "kaggle_idle") {
                  setWorkflow("kaggle")
                  setStatus("kaggle_idle")
                }
              }}
              disabled={status === "processing" || status === "kaggle_processing" || status === "segmented" || status === "kaggle_completed"}
              className={`flex-1 py-2 px-4 rounded-full text-sm font-medium transition-all cursor-pointer ${
                workflow === "kaggle"
                  ? "bg-white/10 text-white"
                  : "text-white/40 hover:text-white hover:bg-white/5"
              } ${status === "processing" || status === "kaggle_processing" || status === "segmented" || status === "kaggle_completed" ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <Activity className="h-4 w-4 inline mr-1" /> Kaggle Classification
            </button>
          </div>
        </div>

        {/* Workflow content */}
        {workflow === "brats" && (
          <Fragment>
            {status === "idle" && (
              <div className="flex flex-col items-center w-full max-w-2xl animate-in fade-in">
                <UploadPill onAnalyze={handleAnalyze} />
                {pipelineError && (
                  <div className="mt-5 w-full flex items-center justify-center gap-2 text-red-400 text-xs bg-red-500/10 px-5 py-3.5 rounded-2xl border border-red-500/20 font-medium text-center animate-in fade-in">
                    <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
                    <span>{pipelineError}</span>
                  </div>
                )}
              </div>
            )}
            {status === "processing" && <ProcessingView />}
          </Fragment>
        )}

        {workflow === "kaggle" && (
          <Fragment>
            {status === "kaggle_idle" && (
              <div className="flex flex-col items-center w-full max-w-2xl animate-in fade-in">
                <KaggleUploadPill onAnalyze={handleKaggleAnalyze} />
                {pipelineError && (
                  <div className="mt-5 w-full flex items-center justify-center gap-2 text-red-400 text-xs bg-red-500/10 px-5 py-3.5 rounded-2xl border border-red-500/20 font-medium text-center animate-in fade-in">
                    <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
                    <span>{pipelineError}</span>
                  </div>
                )}
              </div>
            )}
            {status === "kaggle_processing" && <ProcessingView />}
          </Fragment>
        )}

        {status === "segmented" && results && (
          <ResultsDashboard results={results} onReset={resetDashboard} />
        )}

        {status === "kaggle_completed" && kaggleResults && (
          <KaggleView results={kaggleResults} onReset={resetKaggleDashboard} />
        )}
      </main>
    </div>
  )
}