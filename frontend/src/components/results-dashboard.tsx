"use client"

import { useState, useEffect, useCallback } from "react"
import {
  BrainCircuit,
  Home,
  Share2,
  Target,
  Settings,
  Network,
  Database,
  Activity,
  Download,
  FileText,
  ArrowUpRight,
  ShieldAlert,
  Plus,
  ArrowLeft,
  X,
  Check,
  Loader2,
} from "lucide-react"
import type { ScanResults } from "@/lib/mock-data"

type TabKey = "Segmentation" | "XAI Heatmaps" | "Patient Metadata" | "Network Confidence" | "DICOM Export"
const TABS: TabKey[] = ["Segmentation", "XAI Heatmaps", "Patient Metadata", "Network Confidence", "DICOM Export"]

const NAV = [
  { key: "Segmentation" as TabKey, icon: Home, label: "Overview" },
  { key: "XAI Heatmaps" as TabKey, icon: Share2, label: "XAI heatmaps" },
  { key: "Patient Metadata" as TabKey, icon: Target, label: "Patient metadata" },
  { key: "Network Confidence" as TabKey, icon: Network, label: "Network confidence" },
  { key: "DICOM Export" as TabKey, icon: Database, label: "DICOM export" },
]

type ModalKey = "xai" | "dicom" | "guidelines" | null

const ACCENT: Record<string, string> = {
  cyan: "bg-neuro-cyan",
  lime: "bg-neuro-lime",
  violet: "bg-neuro-violet",
  muted: "bg-white/30",
}

/* ---------------- Toast ---------------- */
function Toast({ message, onDone }: { message: string; onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 3200)
    return () => clearTimeout(t)
  }, [onDone])
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[70] animate-in fade-in slide-from-bottom-4 duration-300">
      <div className="flex items-center gap-3 bg-white/[0.06] backdrop-blur-[40px] border border-white/10 text-white px-5 py-3.5 rounded-2xl shadow-[0_20px_60px_rgba(0,0,0,0.6)] max-w-[90vw]">
        <span className="w-6 h-6 rounded-full bg-neuro-lime/20 border border-neuro-lime/40 flex items-center justify-center shrink-0">
          <Check className="h-3.5 w-3.5 text-neuro-lime" />
        </span>
        <span className="text-sm font-medium text-pretty">{message}</span>
      </div>
    </div>
  )
}

/* ---------------- Modal shell ---------------- */
function Modal({
  title,
  accent,
  onClose,
  children,
}: {
  title: React.ReactNode
  accent: string
  onClose: () => void
  children: React.ReactNode
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])
  return (
    <div
      className="fixed inset-0 z-[65] flex items-center justify-center p-4 animate-in fade-in duration-200"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" aria-hidden="true" />
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-white/[0.05] backdrop-blur-[40px] border border-white/10 rounded-[2rem] p-6 sm:p-8 shadow-[0_40px_100px_rgba(0,0,0,0.8)] animate-in zoom-in-95 duration-300"
      >
        <div className="flex items-start justify-between gap-4 mb-6">
          <h2 className={`text-xl sm:text-2xl font-semibold tracking-tight ${accent}`}>{title}</h2>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="w-9 h-9 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors shrink-0 cursor-pointer"
          >
            <X className="h-4 w-4 text-white/70" />
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}

function Pills({ filled, total = 7, dark }: { filled: number; total?: number; dark?: boolean }) {
  return (
    <div className="flex gap-1.5 mt-auto pt-6">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className={`flex-1 h-11 rounded-full border ${
            i < filled
              ? dark
                ? "bg-black border-black/10"
                : "bg-white border-white/10"
              : "bg-transparent border-dashed border-current/20"
          }`}
        />
      ))}
    </div>
  )
}

export function ResultsDashboard({ results, onReset }: { results: ScanResults; onReset: () => void }) {
  const [activeTab, setActiveTab] = useState<TabKey>("Segmentation")
  const [modal, setModal] = useState<ModalKey>(null)
  const [toast, setToast] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  const notify = useCallback((msg: string) => setToast(msg), [])

  const xaiImageUrl = results.reportUrl || results.xai_url || "/mri-heatmap.png"
  const maskImageUrl = results.maskUrl || results.mask_url || "/mri-segmentation.png"

  // PDF report download or generation
  const handleGenerateReport = () => {
    if (generating) return
    setGenerating(true)
    setTimeout(() => {
      setGenerating(false)
      const downloadTarget = results.report_url || results.reportUrl || results.xai_url || "/mri-heatmap.png"
      window.open(downloadTarget, "_blank")
      notify(`Report for scan ${results.scanId} generated and opened.`)
    }, 1200)
  }

  const handleRawMask = () => {
    window.open(maskImageUrl, "_blank")
    notify("Segmentation mask exported (NIfTI .nii.gz / PNG).")
  }

  const handleFlag = () => notify("Scan successfully flagged and routed to head radiologist queue.")

  return (
    <div className="fixed inset-0 z-50 flex gap-4 p-3 sm:gap-5 sm:p-5 animate-in fade-in zoom-in-95 duration-500 overflow-y-auto">
      {/* Sidebar */}
      <nav className="hidden md:flex w-20 bg-card rounded-[2rem] border border-white/5 flex-col items-center py-8 shadow-2xl shrink-0 sticky top-0 self-start h-[calc(100vh-2.5rem)]">
        <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-10 shadow-[0_0_18px_rgba(56,224,255,0.25)]">
          <BrainCircuit className="h-5 w-5 text-neuro-cyan" />
        </div>
        <div className="flex flex-col gap-4 flex-1 w-full items-center">
          {NAV.map(({ icon: Icon, label, key }) => (
            <button
              key={label}
              aria-label={label}
              aria-current={activeTab === key}
              onClick={() => setActiveTab(key)}
              className={`p-3 rounded-2xl transition-colors cursor-pointer ${
                activeTab === key ? "text-white bg-white/10" : "text-white/40 hover:text-white hover:bg-white/5"
              }`}
            >
              <Icon className="h-5 w-5" />
            </button>
          ))}
        </div>
        <div className="flex flex-col gap-4 w-full items-center">
          <button
            aria-label="Settings"
            onClick={() => notify("Settings panel is currently configured for institutional mode.")}
            className="p-3 text-white/40 hover:text-white transition-colors cursor-pointer"
          >
            <Settings className="h-5 w-5" />
          </button>
        </div>
      </nav>

      {/* Main panel */}
      <div className="flex-1 min-w-0 bg-white/[0.02] backdrop-blur-2xl rounded-[2rem] sm:rounded-[2.5rem] border border-white/10 p-5 sm:p-8 lg:p-10 flex flex-col shadow-[0_30px_80px_rgba(0,0,0,0.8)]">
        {/* Header */}
        <div className="flex flex-wrap justify-between items-center gap-4 mb-8">
          <div className="flex items-center gap-4">
            <button
              onClick={onReset}
              aria-label="Analyze another scan"
              className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors md:hidden cursor-pointer"
            >
              <ArrowLeft className="h-5 w-5 text-white/70" />
            </button>
            <div>
              <p className="font-mono text-xs text-white/40 tracking-widest uppercase">Scan {results.scanId}</p>
              <h1 className="text-3xl sm:text-[38px] font-semibold text-white tracking-tight leading-tight">
                Scan <span className="text-neuro-lime">metrics</span>
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onReset}
              className="hidden md:flex bg-white/5 border border-white/10 text-white/80 px-5 py-3 rounded-full font-medium text-sm items-center gap-2 hover:bg-white/10 transition-colors cursor-pointer"
            >
              <Plus className="h-4 w-4" /> New scan
            </button>
            <button
              onClick={handleGenerateReport}
              disabled={generating}
              className="bg-white text-black px-5 py-3 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/90 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.18)] disabled:opacity-70 cursor-pointer"
            >
              {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />}
              {generating ? "Generating..." : "Generate report"}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 overflow-x-auto pb-1 -mx-1 px-1 shrink-0">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors cursor-pointer ${
                activeTab === tab
                  ? "bg-white/10 text-white border border-white/20"
                  : "text-white/40 hover:text-white hover:bg-white/5 border border-transparent"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Central content area */}
        <div key={activeTab} className="animate-in fade-in duration-300">
          {activeTab === "Segmentation" && (
            <SegmentationView
              results={results}
              xaiImageUrl={xaiImageUrl}
              onExpand={() => setActiveTab("XAI Heatmaps")}
              onExpandImage={() => setModal("xai")}
              onRawMask={handleRawMask}
              onDicom={() => setModal("dicom")}
              onGuidelines={() => setModal("guidelines")}
              onFlag={handleFlag}
            />
          )}
          {activeTab === "XAI Heatmaps" && (
            <XaiView results={results} xaiImageUrl={xaiImageUrl} onExpandImage={() => setModal("xai")} />
          )}
          {activeTab === "Patient Metadata" && <PatientView results={results} />}
          {activeTab === "Network Confidence" && <ConfidenceView results={results} />}
          {activeTab === "DICOM Export" && (
            <DicomView results={results} onRawMask={handleRawMask} onViewPayload={() => setModal("dicom")} />
          )}
        </div>
      </div>

      {/* Modals */}
      {modal === "xai" && (
        <Modal title={<>Full XAI activation map</>} accent="text-neuro-cyan" onClose={() => setModal(null)}>
          <div className="relative rounded-2xl overflow-hidden border border-white/10 aspect-square max-h-[55vh] mx-auto bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={xaiImageUrl} alt="High-resolution XAI activation heatmap of the MRI scan" className="w-full h-full object-cover" />
          </div>
          <p className="text-sm text-white/50 mt-4 leading-relaxed">
            Grad-CAM activation overlay highlighting the voxels that most influenced the ARMT-GAN classification. Warmer
            lime regions indicate higher attribution toward the detected tumor class.
          </p>
        </Modal>
      )}
      {modal === "dicom" && (
        <Modal title={<>DICOM metadata payload</>} accent="text-white" onClose={() => setModal(null)}>
          <div className="rounded-2xl border border-white/10 overflow-hidden">
            <div className="grid grid-cols-[auto_1fr_auto] gap-x-4 text-sm font-mono">
              {results.dicom.map((row, i) => (
                <div key={row.tag} className="contents">
                  <span className={`px-4 py-2.5 text-neuro-cyan/80 ${i % 2 ? "bg-white/[0.02]" : ""}`}>{row.tag}</span>
                  <span className={`px-2 py-2.5 text-white/60 ${i % 2 ? "bg-white/[0.02]" : ""}`}>{row.label}</span>
                  <span className={`px-4 py-2.5 text-white text-right ${i % 2 ? "bg-white/[0.02]" : ""}`}>
                    {row.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Modal>
      )}
      {modal === "guidelines" && (
        <Modal title={<>WHO tumor grading</>} accent="text-neuro-lime" onClose={() => setModal(null)}>
          <div className="space-y-4 text-sm leading-relaxed text-white/70">
            {[
              ["Grade I", "Slow-growing, well-differentiated. Low proliferative potential and possibility of cure after resection."],
              ["Grade II", "Relatively slow-growing but infiltrative. Tend to recur and may progress to higher grades."],
              ["Grade III", "Malignant with anaplasia and active mitosis. Typically require adjuvant radio/chemotherapy."],
              ["Grade IV", "Highly malignant, mitotically active, necrosis-prone (e.g. glioblastoma). Poorest prognosis."],
            ].map(([grade, desc]) => (
              <div key={grade} className="flex gap-4">
                <span className="shrink-0 w-16 font-semibold text-white">{grade}</span>
                <span className="text-pretty">{desc}</span>
              </div>
            ))}
            <p className="text-xs text-white/40 pt-2 border-t border-white/10">
              Reference: WHO Classification of Tumors of the Central Nervous System. Informational only — not a clinical
              diagnosis.
            </p>
          </div>
        </Modal>
      )}

      {toast && <Toast message={toast} onDone={() => setToast(null)} />}
    </div>
  )
}

/* ================= Tab views ================= */

function SegmentationView({
  results,
  xaiImageUrl,
  onExpand,
  onExpandImage,
  onRawMask,
  onDicom,
  onGuidelines,
  onFlag,
}: {
  results: ScanResults
  xaiImageUrl: string
  onExpand: () => void
  onExpandImage: () => void
  onRawMask: () => void
  onDicom: () => void
  onGuidelines: () => void
  onFlag: () => void
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-4 sm:gap-5">
      {/* Confidence */}
      <div className="md:col-span-4 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px]">
        <div className="flex items-center gap-2 text-white/60 text-sm font-medium mb-4">
          <Target className="h-4 w-4" /> AI confidence
        </div>
        <span className="text-5xl sm:text-6xl font-semibold text-white tracking-tighter">
          {Math.floor(results.confidence)}
          <span className="text-2xl text-white/40">.{results.confidence.toString().includes(".") ? results.confidence.toString().split(".")[1] : "0"}%</span>
        </span>
        <Pills filled={results.confidencePills} />
      </div>

      {/* Anomaly area */}
      <div className="md:col-span-4 bg-neuro-lime text-black rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] shadow-[0_0_50px_rgba(190,255,60,0.15)]">
        <div className="flex items-center gap-2 text-black text-sm font-semibold mb-4">
          <Activity className="h-4 w-4" /> Detected anomaly area
        </div>
        <span className="text-5xl sm:text-6xl font-semibold tracking-tighter">
          {Math.floor(results.anomalyArea)}
          <span className="text-2xl text-black/60">.{results.anomalyArea.toString().includes(".") ? results.anomalyArea.toString().split(".")[1] : "0"} cm²</span>
        </span>
        <Pills filled={results.anomalyPills} dark />
      </div>

      {/* XAI heatmap preview */}
      <div className="md:col-span-4 relative rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] overflow-hidden border border-white/10">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-55"
          style={{ backgroundImage: `url('${xaiImageUrl}')` }}
          aria-hidden="true"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-black/10" aria-hidden="true" />
        <div className="relative z-10 flex flex-col h-full">
          <h3 className="text-xl font-semibold text-white leading-tight">
            Full <span className="text-neuro-cyan">XAI activation map</span>
          </h3>
          <div className="mt-auto flex flex-wrap gap-2">
            <button
              onClick={onExpandImage}
              className="bg-white text-black px-5 py-2.5 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/90 transition-colors cursor-pointer"
            >
              Expand view <ArrowUpRight className="h-4 w-4" />
            </button>
            <button
              onClick={onExpand}
              className="bg-white/10 border border-white/15 text-white px-4 py-2.5 rounded-full font-medium text-sm hover:bg-white/20 transition-colors cursor-pointer"
            >
              Open tab
            </button>
          </div>
        </div>
      </div>

      {/* Layer analysis chart */}
      <div className="md:col-span-8 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col">
        <div className="flex flex-wrap justify-between items-center gap-3 mb-8">
          <div className="flex items-center gap-4">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Activity className="h-5 w-5 text-neuro-lime" /> Layer analysis
            </h3>
            <div className="flex gap-3 text-xs font-medium text-white/40">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-white" /> Robustness
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-neuro-lime" /> Adversarial loss
              </span>
            </div>
          </div>
          <span className="bg-white/10 px-4 py-1.5 rounded-full text-xs font-medium text-white/80">Latest epoch</span>
        </div>

        <div className="flex items-end justify-between gap-3 h-48">
          {results.layers.map((bar) => (
            <div key={bar.label} className="flex flex-col items-center gap-3 flex-1 h-full justify-end relative">
              <div className="w-full max-w-[36px] mx-auto h-full bg-black/40 rounded-full relative overflow-hidden border border-white/5">
                <div
                  className={`absolute bottom-0 w-full rounded-full transition-all duration-1000 ${
                    bar.peak ? "bg-black" : "bg-white"
                  }`}
                  style={{ height: `${bar.robustness}%` }}
                />
                <div
                  className="absolute bottom-0 w-full rounded-full bg-neuro-lime transition-all duration-1000"
                  style={{ height: `${bar.adversarial}%` }}
                >
                  <div className="absolute top-2 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-black rounded-full" />
                </div>
              </div>
              {bar.peak && (
                <div className="absolute top-0 -mt-6 bg-black text-white text-[10px] font-semibold px-2 py-1 rounded-md">
                  {bar.robustness}%
                </div>
              )}
              <span className="text-[10px] font-mono text-white/40">{bar.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Action stack */}
      <div className="md:col-span-4 grid grid-rows-3 gap-4">
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={onRawMask}
            className="bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center p-4 hover:bg-white/10 transition-colors group cursor-pointer"
          >
            <Download className="h-6 w-6 text-white/50 mb-2 group-hover:text-white" />
            <span className="text-xs font-semibold text-white">Raw mask</span>
          </button>
          <button
            onClick={onDicom}
            className="bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center p-4 hover:bg-white/10 transition-colors group cursor-pointer"
          >
            <FileText className="h-6 w-6 text-white/50 mb-2 group-hover:text-neuro-cyan" />
            <span className="text-xs font-semibold text-white">DICOM data</span>
          </button>
        </div>
        <button
          onClick={onGuidelines}
          className="bg-white/5 border border-white/10 rounded-2xl p-5 flex items-center justify-between hover:bg-white/10 transition-colors text-left group cursor-pointer"
        >
          <div>
            <h4 className="text-sm font-semibold text-white mb-0.5">Clinical guidelines</h4>
            <p className="text-[11px] text-white/40">Review WHO tumor grading metrics</p>
          </div>
          <ArrowUpRight className="h-4 w-4 text-white/40 group-hover:text-white" />
        </button>
        <button
          onClick={onFlag}
          className="bg-destructive/10 border border-destructive/25 rounded-2xl p-5 flex items-center justify-between hover:bg-destructive/20 transition-colors text-left group cursor-pointer"
        >
          <div>
            <h4 className="text-sm font-semibold text-destructive mb-0.5 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4" /> Flag for review
            </h4>
            <p className="text-[11px] text-destructive/70">Send scan to head radiologist</p>
          </div>
          <ArrowUpRight className="h-4 w-4 text-destructive" />
        </button>
      </div>
    </div>
  )
}

function XaiView({ results, xaiImageUrl, onExpandImage }: { results: ScanResults; xaiImageUrl: string; onExpandImage: () => void }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5">
      <div className="lg:col-span-7 relative rounded-[1.75rem] overflow-hidden border border-white/10 min-h-[320px] bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={xaiImageUrl}
          alt="Full-size XAI activation heatmap over the MRI scan"
          className="absolute inset-0 w-full h-full object-cover opacity-90"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" aria-hidden="true" />
        <div className="absolute bottom-0 left-0 right-0 p-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-xs text-neuro-cyan/80 tracking-widest uppercase mb-1">Grad-CAM overlay</p>
            <h3 className="text-2xl font-semibold text-white">Activation map</h3>
          </div>
          <button
            onClick={onExpandImage}
            className="bg-white text-black px-5 py-2.5 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/90 transition-colors cursor-pointer"
          >
            Full screen <ArrowUpRight className="h-4 w-4" />
          </button>
        </div>
      </div>
      <div className="lg:col-span-5 grid grid-cols-2 gap-4">
        {results.telemetry.map((t) => (
          <div key={t.label} className="bg-white/5 border border-white/10 rounded-2xl p-5 flex flex-col justify-between min-h-[130px]">
            <span className="text-xs text-white/40 font-medium leading-snug">{t.label}</span>
            <div>
              <p className="text-3xl font-semibold text-white tracking-tight">{t.value}</p>
              <p className="text-[11px] text-neuro-cyan/70 font-mono mt-1">{t.sub}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PatientView({ results }: { results: ScanResults }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
      <div className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8">
        <h3 className="text-base font-semibold text-white flex items-center gap-2 mb-6">
          <Target className="h-5 w-5 text-neuro-cyan" /> Study parameters
        </h3>
        <dl className="divide-y divide-white/5">
          {results.patient.map((row) => (
            <div key={row.label} className="flex items-center justify-between py-3.5">
              <dt className="text-sm text-white/50">{row.label}</dt>
              <dd className="text-sm font-medium text-white font-mono">{row.value}</dd>
            </div>
          ))}
        </dl>
      </div>
      <div className="grid grid-rows-2 gap-4 sm:gap-5">
        <div className="bg-neuro-lime text-black rounded-[1.75rem] p-6 sm:p-8 flex flex-col justify-between shadow-[0_0_50px_rgba(190,255,60,0.15)]">
          <span className="text-sm font-semibold flex items-center gap-2">
            <Activity className="h-4 w-4" /> Tumor grade
          </span>
          <div>
            <p className="text-5xl font-semibold tracking-tighter">{results.tumorGrade}</p>
            <p className="text-sm text-black/60 mt-1">WHO classification (predicted)</p>
          </div>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col justify-between">
          <span className="text-sm font-medium text-white/50 flex items-center gap-2">
            <Database className="h-4 w-4" /> Estimated volume
          </span>
          <p className="text-5xl font-semibold text-white tracking-tighter">
            {results.volume}
            <span className="text-2xl text-white/40"> cm³</span>
          </p>
        </div>
      </div>
    </div>
  )
}

function ConfidenceView({ results }: { results: ScanResults }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5">
      <div className="lg:col-span-7 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8">
        <h3 className="text-base font-semibold text-white flex items-center gap-2 mb-8">
          <Network className="h-5 w-5 text-neuro-cyan" /> Class probabilities
        </h3>
        <div className="space-y-6">
          {results.classProbabilities.map((c) => (
            <div key={c.label}>
              <div className="flex justify-between items-baseline mb-2">
                <span className="text-sm text-white/70">{c.label}</span>
                <span className="text-sm font-semibold text-white font-mono">{c.value}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-black/40 border border-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${ACCENT[c.accent]}`}
                  style={{ width: `${c.value}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-5 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col items-center justify-center text-center">
        <span className="text-sm text-white/50 mb-2">Softmax top-1 confidence</span>
        <div className="relative w-40 h-40 flex items-center justify-center my-2">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100" aria-hidden="true">
            <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
            <circle
              cx="50"
              cy="50"
              r="44"
              fill="none"
              stroke="var(--neuro-lime)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${(results.confidence / 100) * 276.4} 276.4`}
            />
          </svg>
          <span className="absolute text-3xl font-semibold text-white">{results.confidence}%</span>
        </div>
        <p className="text-xs text-white/40 leading-relaxed max-w-[220px]">
          Ensemble agreement across 5 ARMT-GAN folds. Above the 95% auto-report threshold.
        </p>
      </div>
    </div>
  )
}

function DicomView({
  results,
  onRawMask,
  onViewPayload,
}: {
  results: ScanResults
  onRawMask: () => void
  onViewPayload: () => void
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5">
      <div className="lg:col-span-8 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8">
        <div className="flex items-center justify-between gap-4 mb-6">
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Database className="h-5 w-5 text-neuro-cyan" /> DICOM tags
          </h3>
          <button
            onClick={onViewPayload}
            className="text-xs font-medium text-neuro-cyan hover:text-white transition-colors flex items-center gap-1 cursor-pointer"
          >
            View full payload <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="rounded-2xl border border-white/10 overflow-hidden text-sm font-mono">
          {results.dicom.slice(0, 6).map((row, i) => (
            <div
              key={row.tag}
              className={`grid grid-cols-[auto_1fr_auto] gap-x-4 px-4 py-2.5 ${i % 2 ? "bg-white/[0.02]" : ""}`}
            >
              <span className="text-neuro-cyan/80">{row.tag}</span>
              <span className="text-white/50">{row.label}</span>
              <span className="text-white text-right">{row.value}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-4 grid grid-rows-2 gap-4 sm:gap-5">
        <button
          onClick={onRawMask}
          className="bg-neuro-lime text-black rounded-[1.75rem] p-6 flex flex-col justify-between text-left hover:brightness-95 transition-all shadow-[0_0_50px_rgba(190,255,60,0.15)] cursor-pointer"
        >
          <Download className="h-6 w-6" />
          <div>
            <p className="text-lg font-semibold">Export mask</p>
            <p className="text-xs text-black/60">NIfTI .nii.gz segmentation</p>
          </div>
        </button>
        <button
          onClick={onViewPayload}
          className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 flex flex-col justify-between text-left hover:bg-white/10 transition-colors cursor-pointer"
        >
          <FileText className="h-6 w-6 text-neuro-cyan" />
          <div>
            <p className="text-lg font-semibold text-white">Raw payload</p>
            <p className="text-xs text-white/40">Inspect all DICOM headers</p>
          </div>
        </button>
      </div>
    </div>
  )
}
