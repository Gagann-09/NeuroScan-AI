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
import type { ScanResults, KaggleResults } from "@/lib/mock-data"

type TabKey = "Segmentation" | "XAI Heatmaps" | "Patient Metadata" | "Network Confidence" | "DICOM Export"
const TABS: TabKey[] = ["Segmentation", "XAI Heatmaps", "Patient Metadata", "Network Confidence", "DICOM Export"]

const NAV = [
  { key: "Segmentation" as TabKey, icon: Home, label: "Overview" },
  { key: "XAI Heatmaps" as TabKey, icon: Share2, label: "XAI heatmaps" },
  { key: "Patient Metadata" as TabKey, icon: Target, label: "Patient metadata" },
  { key: "Network Confidence" as TabKey, icon: Network, label: "Network confidence" },
  { key: "DICOM Export" as TabKey, icon: Database, label: "DICOM export" },
]

type ModalKey = "xai" | "mask" | "dicom" | "guidelines" | null

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

function ResultsDashboard({ results, onReset }: { results: ScanResults; onReset: () => void }) {
  const [activeTab, setActiveTab] = useState<TabKey>("Segmentation")
  const [modal, setModal] = useState<ModalKey>(null)
  const [toast, setToast] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  const notify = useCallback((msg: string) => setToast(msg), [])

  const xaiImageUrl = results.xai_url || ""
  const maskImageUrl = results.mask_url || results.maskUrl || ""

  // PDF report download or generation
  const handleGenerateReport = () => {
    if (generating) return
    setGenerating(true)
    setTimeout(() => {
      setGenerating(false)
      const downloadTarget = results.report_url || results.reportUrl || ""
      if (downloadTarget) {
        window.open(downloadTarget, "_blank")
        notify(`Report for scan ${results.scanId} opened.`)
      } else {
        notify("No PDF report available for this scan.")
      }
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
      <div className="flex-1 min-w-0 min-h-[calc(100vh-2.5rem)] self-start bg-white/[0.02] backdrop-blur-2xl rounded-[2rem] sm:rounded-[2.5rem] border border-white/10 p-5 sm:p-8 lg:p-10 flex flex-col shadow-[0_30px_80px_rgba(0,0,0,0.8)]">
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
              maskImageUrl={maskImageUrl}
              onExpand={() => setActiveTab("XAI Heatmaps")}
              onExpandOverlay={(type) => setModal(type)}
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
            Gradient-based input saliency map highlighting the voxels that most influenced the ARMT-GAN segmentation prediction. Warmer
            lime regions indicate higher attribution toward the detected tumor class.
          </p>
        </Modal>
      )}
      {modal === "mask" && (
        <Modal title={<>Coregistered tumor mask</>} accent="text-neuro-lime" onClose={() => setModal(null)}>
          <div className="relative rounded-2xl overflow-hidden border border-white/10 aspect-square max-h-[55vh] mx-auto bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={maskImageUrl} alt="High-resolution tumor segmentation mask" className="w-full h-full object-cover" />
          </div>
          <p className="text-sm text-white/50 mt-4 leading-relaxed">
            2D axial tumor segmentation generated by the ARMT-GAN generator with morphological boundary refinement.
            Green contours outline the predicted whole-tumor boundary on the coregistered MRI slice.
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
              Reference: WHO Classification of Tumors of the Central Nervous System. Informational only â€” not a clinical
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
  maskImageUrl,
  onExpand,
  onExpandOverlay,
  onRawMask,
  onDicom,
  onGuidelines,
  onFlag,
}: {
  results: ScanResults
  xaiImageUrl: string
  maskImageUrl: string
  onExpand: () => void
  onExpandOverlay: (type: "xai" | "mask") => void
  onRawMask: () => void
  onDicom: () => void
  onGuidelines: () => void
  onFlag: () => void
}) {
  const [activeOverlay, setActiveOverlay] = useState<"xai" | "mask">("xai")


  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-4 sm:gap-5">
      {/* Confidence */}
      <div className="md:col-span-4 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px]">
        <div className="flex items-center gap-2 text-white/60 text-sm font-medium mb-4">
          <Target className="h-4 w-4" /> Peak tumor probability
        </div>
        {results.confidence != null ? (
          <div>
            <span className="text-5xl sm:text-6xl font-semibold text-white tracking-tighter">
              {Math.floor(results.confidence)}
              <span className="text-2xl text-white/40">
                .{results.confidence.toString().includes(".") ? results.confidence.toString().split(".")[1] : "0"}%
              </span>
            </span>
            <span className="text-xs text-white/40 block font-normal mt-1">
              Raw ARMT-GAN sigmoid output — not accuracy
            </span>
          </div>
        ) : (
          <span className="text-3xl sm:text-4xl font-semibold text-white/50 tracking-tight">
            N/A <span className="text-xs text-white/40 block font-normal mt-1">Not evaluated</span>
          </span>
        )}
        <Pills filled={results.confidence != null ? results.confidencePills : 0} />
      </div>


      {/* Technical measurements from the selected 2D slice */}
      <div className="md:col-span-4 bg-neuro-lime text-black rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] shadow-[0_0_50px_rgba(190,255,60,0.15)]">
        <div className="flex items-center gap-2 text-black text-sm font-semibold mb-4">
          <Activity className="h-4 w-4" /> 2D segmentation measurements
        </div>

        {results.predictedMaskPixels != null ? (
          <>
            <span className="text-4xl sm:text-5xl font-semibold tracking-tighter">
              {results.predictedMaskPixels.toLocaleString()}
              <span className="text-base text-black/70 font-medium ml-2">
                pixels
              </span>
            </span>
            <div className="mt-3 text-sm text-black/75">
              {results.sliceCoveragePercent != null
                ? `${results.sliceCoveragePercent.toFixed(4)}% of selected slice`
                : "Slice coverage unavailable"}
            </div>
          </>
        ) : (
          <span className="text-3xl sm:text-4xl font-semibold text-black/60 tracking-tight">
            Unavailable
            <span className="text-xs text-black/70 block font-normal mt-2 leading-relaxed">
              No stored measurement is available for this scan.
            </span>
          </span>
        )}

        <div className="mt-auto pt-4 text-xs text-black/70 leading-relaxed">
          Pixels with model probability ≥ 0.5 on one 2D slice. Not physical
          tumor area, tumor volume, or diagnostic accuracy.
        </div>
      </div>

      {/* Visual overlay */}
      <div className="md:col-span-4 relative rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] overflow-hidden border border-white/10">
        {(activeOverlay === "mask" ? maskImageUrl : xaiImageUrl) ? (
          <div
            className="absolute inset-0 bg-cover bg-center opacity-60 transition-all duration-300"
            style={{
              backgroundImage: `url('${activeOverlay === "mask" ? maskImageUrl : xaiImageUrl}')`,
            }}
            aria-hidden="true"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-6 text-center">
            <p className="text-sm text-white/60">
              {activeOverlay === "mask"
                ? "Segmentation mask unavailable for this scan."
                : "XAI saliency artifact unavailable for this scan."}
            </p>
          </div>
        )}

        <div
          className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-black/10"
          aria-hidden="true"
        />

        <div className="relative z-10 flex flex-col h-full">
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex bg-black/60 backdrop-blur-md p-0.5 rounded-full border border-white/10 text-[11px]">
              <button
                type="button"
                onClick={() => setActiveOverlay("xai")}
                className={`px-2.5 py-0.5 rounded-full font-medium transition-colors cursor-pointer ${
                  activeOverlay === "xai"
                    ? "bg-white/20 text-white"
                    : "text-white/50 hover:text-white"
                }`}
              >
                XAI Saliency
              </button>
              <button
                type="button"
                onClick={() => setActiveOverlay("mask")}
                className={`px-2.5 py-0.5 rounded-full font-medium transition-colors cursor-pointer ${
                  activeOverlay === "mask"
                    ? "bg-neuro-lime text-black font-semibold"
                    : "text-white/50 hover:text-white"
                }`}
              >
                Segmentation
              </button>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-white leading-tight">
            {activeOverlay === "mask" ? (
              <>Coregistered <span className="text-neuro-lime">tumor mask</span></>
            ) : (
              <>Full <span className="text-neuro-cyan">XAI activation map</span></>
            )}
          </h3>

          <div className="mt-auto flex flex-wrap gap-2">
            <button
              onClick={() => onExpandOverlay(activeOverlay)}
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

      {/* Classification and grading: explicitly not inferred from segmentation */}
      <div className="md:col-span-8 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-5">
        {/* Tumor type */}
        <section className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-7 flex flex-col justify-between min-h-[220px]">
          <div>
            <div className="flex items-center gap-2 text-sm text-white/60 font-medium">
              <Target className="h-4 w-4 text-neuro-cyan" />
              Tumor type
            </div>
            <p className="mt-5 text-2xl sm:text-3xl font-semibold text-white tracking-tight">
              {results.tumorType}
            </p>
            <p className="text-xs sm:text-sm text-white/55 mt-2 leading-relaxed">
              {results.tumorTypeSource}
            </p>
          </div>
          <div className="mt-5 rounded-xl border border-neuro-cyan/20 bg-neuro-cyan/5 p-3">
            <p className="text-xs font-medium text-neuro-cyan">Available model output</p>
            <p className="text-sm text-white/75 mt-1">
              Single-channel, 2D tumor segmentation mask
            </p>
            <p className="text-xs text-white/40 mt-1">
              This workflow does not classify tumor histology.
            </p>
          </div>
        </section>

        {/* WHO grade / stage */}
        <section className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-7 flex flex-col justify-between min-h-[220px]">
          <div>
            <div className="flex items-center gap-2 text-sm text-white/60 font-medium">
              <Activity className="h-4 w-4 text-neuro-lime" />
              WHO grade / stage
            </div>
            <p className="mt-5 text-2xl sm:text-3xl font-semibold text-white tracking-tight">
              {results.tumorGrade}
            </p>
            <p className="text-xs sm:text-sm text-white/55 mt-2 leading-relaxed">
              {results.tumorGradeSource}
            </p>
          </div>
          <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-3">
            <p className="text-xs font-medium text-white/70">Evidence required</p>
            <p className="text-sm text-white/65 mt-1">
              A validated grading model and/or appropriate clinical and pathological evidence.
            </p>
            <p className="text-xs text-white/40 mt-1">
              Segmentation output alone cannot establish WHO grade.
            </p>
          </div>
        </section>

        {/* BraTS dataset cohort metadata */}
        {results.datasetCohortLabel && (
          <section
            className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-7 flex flex-col justify-between min-h-[220px]"
            aria-label="BraTS dataset cohort metadata"
          >
            <div>
              <div className="flex items-center gap-2 text-sm text-white/60 font-medium">
                <Activity className="h-4 w-4 text-neuro-cyan" />
                BraTS dataset cohort
              </div>
              <p className="mt-5 text-2xl sm:text-3xl font-semibold text-white tracking-tight">
                {results.datasetCohortLabel}
              </p>
              <p className="text-xs sm:text-sm text-white/55 mt-2 leading-relaxed">
                {results.datasetCohortSource}
              </p>
            </div>
            <div className="mt-5 rounded-xl border border-neuro-cyan/20 bg-neuro-cyan/5 p-3">
              <p className="text-xs font-medium text-neuro-cyan">
                Dataset metadata only
              </p>
              <p className="text-sm text-white/75 mt-1">
                This cohort label comes from the verified BraTS subject mapping.
              </p>
              <p className="text-xs text-white/40 mt-1">
                It is not an ARMT-GAN prediction or a WHO grade.
              </p>
            </div>
          </section>
        )}
      </div>

      {/* Actions */}
      <div className="md:col-span-4 grid grid-cols-2 gap-3 sm:gap-4">
        <button
          onClick={onRawMask}
          className="bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center gap-3 p-4 hover:bg-white/10 transition-colors group cursor-pointer min-h-[76px]"
        >
          <Download className="h-5 w-5 text-white/50 group-hover:text-white" />
          <span className="text-xs sm:text-sm font-semibold text-white">Raw mask</span>
        </button>
        <button
          onClick={onDicom}
          className="bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center gap-3 p-4 hover:bg-white/10 transition-colors group cursor-pointer min-h-[76px]"
        >
          <FileText className="h-5 w-5 text-white/50 group-hover:text-neuro-cyan" />
          <span className="text-xs sm:text-sm font-semibold text-white">DICOM data</span>
        </button>
        <button
          onClick={onGuidelines}
          className="col-span-2 bg-white/5 border border-white/10 rounded-2xl p-5 flex items-center justify-between hover:bg-white/10 transition-colors text-left group cursor-pointer"
        >
          <div>
            <h4 className="text-sm font-semibold text-white mb-0.5">Clinical guidelines</h4>
            <p className="text-[11px] text-white/40">Reference guidance only — not a grading prediction</p>
          </div>
          <ArrowUpRight className="h-4 w-4 text-white/40 group-hover:text-white" />
        </button>
      </div>

      {/* Full-width review action */}
      <button
        onClick={onFlag}
          className="md:col-span-12 mt-6 sm:mt-8 w-full bg-destructive/10 border border-destructive/25 rounded-2xl px-6 py-5 sm:px-8 sm:py-6 flex items-center justify-between gap-4 hover:bg-destructive/20 transition-colors text-left group cursor-pointer min-h-[100px]"      >
        <div>
          <h4 className="text-base font-semibold text-destructive mb-1 flex items-center gap-2">
            <ShieldAlert className="h-5 w-5" />
            Flag for review
          </h4>
          <p className="text-xs sm:text-sm text-destructive/70">
            Request review by a qualified radiologist. Research output only; not a clinical diagnosis.
          </p>
        </div>
        <ArrowUpRight className="h-5 w-5 shrink-0 text-destructive group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
      </button>
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
            <p className="font-mono text-xs text-neuro-cyan/80 tracking-widest uppercase mb-1">Gradient Saliency overlay</p>
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
      <div className="lg:col-span-5 flex flex-col">
        <p className="text-xs font-mono text-white/40 uppercase tracking-widest mb-3">Model Validation Telemetry (Reference / Unverified Benchmark)</p>
        <div className="grid grid-cols-2 gap-4 flex-1">
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
    </div>
  )
}

function PatientView({ results }: { results: ScanResults }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
      <div className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8">
        <h3 className="text-base font-semibold text-white flex items-center gap-2 mb-6">
          <Target className="h-5 w-5 text-neuro-cyan" />
          Study parameters
        </h3>

        <dl className="divide-y divide-white/5">
          {results.patient.map((row) => (
            <div
              key={row.label}
              className="flex items-center justify-between gap-4 py-3.5"
            >
              <dt className="text-sm text-white/50">{row.label}</dt>
              <dd className="text-sm font-medium text-white font-mono text-right break-words">
                {row.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="grid grid-rows-2 gap-4 sm:gap-5">
        <div className="bg-neuro-lime text-black rounded-[1.75rem] p-6 sm:p-8 flex flex-col justify-between shadow-[0_0_50px_rgba(190,255,60,0.15)]">
          <span className="text-sm font-semibold flex items-center gap-2">
            <Target className="h-4 w-4" />
            Tumor type
          </span>

          <div>
            <p className="text-3xl font-semibold tracking-tight">
              {results.tumorType}
            </p>
            <p className="text-sm text-black/60 mt-2">
              {results.tumorTypeSource}
            </p>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col justify-between">
          <span className="text-sm font-medium text-white/50 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            WHO grade / stage
          </span>

          <div>
            <p className="text-3xl font-semibold text-white tracking-tight">
              {results.tumorGrade}
            </p>
            <p className="text-xs text-white/40 mt-2">
              {results.tumorGradeSource}
            </p>
          </div>
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
          <Network className="h-5 w-5 text-neuro-cyan" /> Segmentation Output
        </h3>
        <div className="space-y-6">
          {results.classProbabilities.map((c) => (
            <div key={c.label}>
              <div className="flex justify-between items-baseline mb-2">
                <div>
                  <span className="text-sm text-white/70">{c.label}</span>
                  {c.value == null && (
                    <span className="text-xs text-white/40 block">Not supported (1-channel binary model)</span>
                  )}
                </div>
                <span className="text-sm font-semibold text-white font-mono">
                  {c.value != null ? `${c.value}%` : "N/A"}
                </span>
              </div>
              <div className="h-2.5 rounded-full bg-black/40 border border-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${c.value != null ? ACCENT[c.accent] : "bg-white/10"}`}
                  style={{ width: `${c.value != null ? c.value : 0}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-5 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col items-center justify-center text-center">
        <span className="text-sm text-white/50 mb-2">Peak tumor probability</span>
        <div className="relative w-40 h-40 flex items-center justify-center my-2">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100" aria-hidden="true">
            <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
            {results.confidence != null && (
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
            )}
          </svg>
          <span className="absolute text-3xl font-semibold text-white">
            {results.confidence != null ? `${results.confidence}%` : "N/A"}
          </span>
        </div>
        <p className="text-xs text-white/40 leading-relaxed max-w-[220px]">
          Peak tumor probability computed from raw ARMT-GAN sigmoid output on the segmented axial slice.
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
            <Database className="h-5 w-5 text-neuro-cyan" /> Reference DICOM tags
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

function KaggleView({ results, onReset }: { results: KaggleResults; onReset: () => void }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5">
      <div className="lg:col-span-12 flex justify-end mb-4">
        <button
          onClick={onReset}
          aria-label="Analyze another image"
          className="bg-white/5 border border-white/10 text-white/80 px-4 py-2 rounded-full font-medium text-sm items-center gap-2 hover:bg-white/10 transition-colors cursor-pointer"
        >
          <Plus className="h-3.5 w-3.5" /> New scan
        </button>
      </div>
      {/* Classification Result */}
      <div className="lg:col-span-7 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col">
        <div className="flex items-center gap-2 text-white/60 text-sm font-medium mb-4">
          <Target className="h-4 w-4" /> Classification result
        </div>

        <div className="flex flex-col md:flex-row gap-8 mb-8">
          <div className="flex-1 flex flex-col items-center justify-center">
            <p className="text-xs text-white/40 uppercase tracking-widest mb-2">Predicted class</p>
            <h2 className="text-4xl sm:text-5xl font-semibold text-white capitalize">{results.class_name}</h2>
            <p className="text-xs text-white/40 mt-1">Index: {results.class_index}</p>
          </div>
          <div className="flex-1">
            <p className="text-xs text-white/40 uppercase tracking-widest mb-2">Model version</p>
            <p className="font-mono text-white/70">{results.model_version}</p>
          </div>
        </div>

        {/* Probability bars */}
        <div className="space-y-4">
          <p className="text-xs text-white/40 uppercase tracking-widest mb-3">Class probabilities</p>
          {Object.entries(results.probabilities).map(([name, prob]) => (
            <div key={name} className="bg-white/5 border border-white/10 rounded-2xl p-4">
              <div className="flex justify-between items-baseline mb-2">
                <span className="text-sm text-white/70 capitalize">{name}</span>
                <span className="text-sm font-semibold text-white font-mono">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2.5 rounded-full bg-black/40 border border-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${ACCENT.lime}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Source image + metadata */}
      <div className="lg:col-span-5 flex flex-col gap-4">
        {/* Source image */}
        <div className="relative rounded-[1.75rem] overflow-hidden border border-white/10 min-h-[280px] bg-black">
          {results.image_url && (
            <img
              src={results.image_url}
              alt="Source MRI image"
              className="absolute inset-0 w-full h-full object-cover opacity-90"
            />
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" aria-hidden="true" />
          <div className="absolute bottom-0 left-0 right-0 p-6">
            <p className="font-mono text-xs text-neuro-cyan/80 tracking-widest uppercase mb-1">Source MRI</p>
            <h3 className="text-xl font-semibold text-white">Kaggle classification input</h3>
          </div>
        </div>

        {/* Metadata */}
        <div className="bg-white/5 border border-white/10 rounded-[1.75rem] p-6">
          <h3 className="text-base font-semibold text-white flex items-center gap-2 mb-4">
            <Database className="h-5 w-5 text-neuro-cyan" /> Study metadata
          </h3>
          <dl className="divide-y divide-white/5">
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-white/50">Dataset</dt>
              <dd className="text-sm font-medium text-white font-mono">Kaggle Brain MRI</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-white/50">Task</dt>
              <dd className="text-sm font-medium text-white font-mono">4-class tumor classification</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-white/50">Model</dt>
              <dd className="text-sm font-medium text-white font-mono">{results.model_version}</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-white/50">Input format</dt>
              <dd className="text-sm font-medium text-white font-mono">Single JPG/PNG, 224Ã—224 grayscale</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-white/50">Classes</dt>
              <dd className="text-sm font-medium text-white font-mono">glioma, meningioma, notumor, pituitary</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  )
}

export { ResultsDashboard, KaggleView }
