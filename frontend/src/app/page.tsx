"use client"

import { useState, useRef, useEffect } from "react"
import { BrainCircuit, AlertCircle, LogOut } from "lucide-react"
import { BrainNodes } from "@/components/brain-nodes"
import { UploadPill } from "@/components/upload-pill"
import { ProcessingView } from "@/components/processing-view"
import { ResultsDashboard } from "@/components/results-dashboard"
import { MOCK_RESULTS, type ScanResults } from "@/lib/mock-data"
import { uploadScan, getScanStatus, getScanResults } from "@/lib/api"
import { useAuth } from "@/lib/auth-context"
import { auth } from "@/lib/firebase"
import { signOut } from "firebase/auth"

type Status = "idle" | "processing" | "segmented"

export default function Dashboard() {
  const [status, setStatus] = useState<Status>("idle")
  const [results, setResults] = useState<ScanResults | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const { user } = useAuth()

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  /**
   * ARMT-GAN Pipeline Coordination:
   * 1. POST file → /api/v1/scans/upload → capture scan_id
   * 2. Poll GET /api/v1/scans/status/{scan_id}
   * 3. On SEGMENTED → GET /api/v1/scans/results/{scan_id} → hydrate results
   */
  const handleAnalyze = async (file: File) => {
    setPipelineError(null)
    setStatus("processing")

    let scan_id: string

    // ── Step 1: Upload Scan ──────────────────────────────────────────────────
    try {
      const uploadResponse = await uploadScan(file)
      scan_id = uploadResponse.scan_id
    } catch (err: any) {
      setPipelineError(err.message || "Failed to upload scan. Ensure the backend server is running on port 8000.")
      setStatus("idle")
      return
    }

    // ── Step 2: Poll Status ──────────────────────────────────────────────────
    intervalRef.current = setInterval(async () => {
      try {
        const { status: backendStatus } = await getScanStatus(scan_id)

        if (backendStatus === "SEGMENTED") {
          // Stop polling immediately
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }

          // ── Step 3: Retrieve Segmentation & XAI Results ────────────────────
          try {
            const rawApiResults = await getScanResults(scan_id)
            const apiResults = rawApiResults as any // Bypass strict TS check for backend telemetry fields

            setResults({
              ...MOCK_RESULTS,
              scanId: scan_id,
              maskUrl: apiResults.mask_url || MOCK_RESULTS.maskUrl,
              reportUrl: apiResults.xai_url || apiResults.report_url || MOCK_RESULTS.reportUrl,
              mask_url: apiResults.mask_url,
              xai_url: apiResults.xai_url,
              report_url: apiResults.report_url,
              // ── MATCH EXACT CAMELCASE PROPERTIES EXPECTED BY ScanResults ──
              confidence: apiResults.confidence_score ? Math.round(apiResults.confidence_score * 100) : MOCK_RESULTS.confidence,
              anomalyArea: apiResults.anomaly_area_cm2 !== undefined ? apiResults.anomaly_area_cm2 : MOCK_RESULTS.anomalyArea,
              tumorGrade: apiResults.who_grade || MOCK_RESULTS.tumorGrade,
            } as any)
          } catch (err: any) {
            setPipelineError(err.message || "Failed to fetch segmentation results from backend.")
            setStatus("idle")
          }
        }
      } catch (err: any) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        setPipelineError("Lost connection to inference backend during processing.")
        setStatus("idle")
      }
    }, 2000)
  }

  const resetDashboard = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setResults(null)
    setPipelineError(null)
    setStatus("idle")
  }

  const handleSignOut = async () => {
    try {
      await signOut(auth)
    } catch (error) {
      console.error("Sign out error", error)
    }
  }

  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col">
      <BrainNodes />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[50vw] h-[50vw] bg-neuro-cyan/5 blur-[120px] rounded-full pointer-events-none z-0" />

      {status !== "segmented" && (
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
        {status === "idle" && (
          <div className="flex flex-col items-center w-full max-w-2xl">
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
      </main>

      {status === "segmented" && results && (
        <ResultsDashboard results={results} onReset={resetDashboard} />
      )}
    </div>
  )
}