"use client"

import { useState, useRef, useEffect } from "react"
import { BrainCircuit } from "lucide-react"
import { BrainNodes } from "@/components/brain-nodes"
import { UploadPill } from "@/components/upload-pill"
import { ProcessingView } from "@/components/processing-view"
import { ResultsDashboard } from "@/components/results-dashboard"
import { MOCK_RESULTS, type ScanResults } from "@/lib/mock-data"
import { uploadScan, getScanStatus, getScanResults } from "@/lib/api"

type Status = "idle" | "processing" | "segmented"

export default function Dashboard() {
  const [status, setStatus] = useState<Status>("idle")
  const [results, setResults] = useState<ScanResults | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  /**
   * Real ARMT-GAN pipeline:
   * 1. POST file → /api/v1/scans/upload → capture scan_id
   * 2. Poll GET /api/v1/scans/status/{scan_id} every 2 s
   * 3. On SEGMENTED → GET /api/v1/scans/results/{scan_id} → hydrate results
   */
  const handleAnalyze = async (file: File) => {
    setPipelineError(null)
    setStatus("processing")

    let scan_id: string

    // ── Step 1: Upload ────────────────────────────────────────────────────────
    try {
      const uploadResponse = await uploadScan(file)
      scan_id = uploadResponse.scan_id
    } catch (err: any) {
      setPipelineError(err.message ?? "Upload failed. Ensure the backend is running.")
      setStatus("idle")
      return
    }

    // ── Step 2: Poll status every 2 s ─────────────────────────────────────────
    intervalRef.current = setInterval(async () => {
      try {
        const { status: backendStatus } = await getScanStatus(scan_id)

        if (backendStatus === "SEGMENTED") {
          // Stop polling immediately
          clearInterval(intervalRef.current!)
          intervalRef.current = null

          // ── Step 3: Fetch results ───────────────────────────────────────────
          try {
            const apiResults = await getScanResults(scan_id)

            // Merge real backend URLs into the MOCK_RESULTS scaffold so the
            // existing v0 ResultsDashboard renders without structural changes.
            setResults({
              ...MOCK_RESULTS,
              scanId: scan_id,
              mask_url: apiResults.mask_url,
              xai_url: apiResults.xai_url,
              report_url: apiResults.report_url,
            })
            setStatus("segmented")
          } catch (err: any) {
            setPipelineError(err.message ?? "Failed to retrieve segmentation results.")
            setStatus("idle")
          }
        }
        // PENDING / PROCESSING → keep showing ProcessingView (no state change needed)
      } catch (err: any) {
        clearInterval(intervalRef.current!)
        intervalRef.current = null
        setPipelineError("Lost connection to backend during processing.")
        setStatus("idle")
      }
    }, 2000)
  }

  const resetDashboard = () => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = null
    setResults(null)
    setPipelineError(null)
    setStatus("idle")
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
          <span className="text-xs text-white/40 font-mono tracking-widest hidden sm:inline">
            ARMT-GAN · v2.4 · LIVE
          </span>
        </header>
      )}

      <main className="flex-1 flex flex-col items-center justify-center z-30 relative px-4 w-full pb-16">
        {status === "idle" && (
          <>
            <UploadPill onAnalyze={handleAnalyze} />
            {pipelineError && (
              <div className="mt-4 text-red-400 text-xs bg-red-500/10 px-5 py-3 rounded-xl border border-red-500/20 font-medium max-w-md text-center">
                ⚠️ {pipelineError}
              </div>
            )}
          </>
        )}
        {status === "processing" && <ProcessingView />}
      </main>

      {status === "segmented" && results && <ResultsDashboard results={results} onReset={resetDashboard} />}
    </div>
  )
}
