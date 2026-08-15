"use client"

import { useState } from "react"
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
} from "lucide-react"
import type { ScanResults } from "@/lib/mock-data"

const NAV = [
  { icon: Home, label: "Overview", active: true },
  { icon: Share2, label: "Share" },
  { icon: Target, label: "Segmentation" },
  { icon: Network, label: "Confidence" },
  { icon: Database, label: "Datasets" },
]

const TABS = ["Segmentation", "XAI Heatmaps", "Patient Metadata", "Network Confidence", "DICOM Export"]

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
  const [activeTab, setActiveTab] = useState("Segmentation")
  const [activeNav, setActiveNav] = useState("Overview")
  const [isXaiModalOpen, setIsXaiModalOpen] = useState(false)
  const [isDicomModalOpen, setIsDicomModalOpen] = useState(false)
  const [isGuidelinesModalOpen, setIsGuidelinesModalOpen] = useState(false)

  const handleFlagForReview = () => {
    alert("Scan successfully flagged and routed to head radiologist queue.")
  }

  return (
    <div className="fixed inset-0 z-50 flex gap-4 p-3 sm:gap-5 sm:p-5 animate-in fade-in zoom-in-95 duration-500 overflow-y-auto">
      {/* Sidebar */}
      <nav className="hidden md:flex w-20 bg-card rounded-[2rem] border border-white/5 flex-col items-center py-8 shadow-2xl shrink-0 sticky top-0 self-start h-[calc(100vh-2.5rem)]">
        <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-10 shadow-[0_0_18px_rgba(56,224,255,0.25)]">
          <BrainCircuit className="h-5 w-5 text-neuro-cyan" />
        </div>
        <div className="flex flex-col gap-4 flex-1 w-full items-center">
          {NAV.map(({ icon: Icon, label }) => (
            <button
              key={label}
              aria-label={label}
              onClick={() => setActiveNav(label)}
              className={`p-3 rounded-2xl transition-colors ${
                activeNav === label ? "text-white bg-white/10" : "text-white/40 hover:text-white hover:bg-white/5"
              }`}
            >
              <Icon className="h-5 w-5" />
            </button>
          ))}
        </div>
        <div className="flex flex-col gap-4 w-full items-center">
          <button aria-label="Settings" className="p-3 text-white/40 hover:text-white transition-colors">
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
              className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-colors md:hidden"
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
              className="hidden md:flex bg-white/5 border border-white/10 text-white/80 px-5 py-3 rounded-full font-medium text-sm items-center gap-2 hover:bg-white/10 transition-colors"
            >
              <Plus className="h-4 w-4" /> New scan
            </button>
            <a
              href={results.report_url ?? '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white text-black px-5 py-3 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/90 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.18)]"
            >
              <FileText className="h-4 w-4" /> Generate report
            </a>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 overflow-x-auto pb-1 -mx-1 px-1 shrink-0">
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
                activeTab === tab
                  ? "bg-white/10 text-white border border-white/20"
                  : "text-white/40 hover:text-white hover:bg-white/5"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Bento grid */}
        {activeTab === "Segmentation" && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 sm:gap-5 animate-in fade-in duration-300">
          {/* Confidence */}
          <div className="md:col-span-4 bg-white/5 border border-white/10 rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px]">
            <div className="flex items-center gap-2 text-white/60 text-sm font-medium mb-4">
              <Target className="h-4 w-4" /> AI confidence
            </div>
            <span className="text-5xl sm:text-6xl font-semibold text-white tracking-tighter">
              {Math.floor(results.confidence)}
              <span className="text-2xl text-white/40">.{results.confidence.toString().split(".")[1]}%</span>
            </span>
            <Pills filled={results.confidencePills} />
          </div>

          {/* Anomaly area (lime accent) */}
          <div className="md:col-span-4 bg-neuro-lime text-black rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] shadow-[0_0_50px_rgba(190,255,60,0.15)]">
            <div className="flex items-center gap-2 text-black text-sm font-semibold mb-4">
              <Activity className="h-4 w-4" /> Detected anomaly area
            </div>
            <span className="text-5xl sm:text-6xl font-semibold tracking-tighter">
              {Math.floor(results.anomalyArea)}
              <span className="text-2xl text-black/60">.{results.anomalyArea.toString().split(".")[1]} cm²</span>
            </span>
            <Pills filled={results.anomalyPills} dark />
          </div>

          {/* XAI heatmap */}
          <div className="md:col-span-4 relative rounded-[1.75rem] p-6 sm:p-8 flex flex-col min-h-[220px] overflow-hidden border border-white/10">
            <div
              className="absolute inset-0 bg-cover bg-center opacity-55"
              style={{ backgroundImage: results.xai_url ? `url('${results.xai_url}')` : "url('/mri-heatmap.png')" }}
              aria-hidden="true"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-black/10" aria-hidden="true" />
            <div className="relative z-10 flex flex-col h-full">
              <h3 className="text-xl font-semibold text-white leading-tight">
                Full <span className="text-neuro-cyan">XAI activation map</span>
              </h3>
              <button
                onClick={(e) => {
                  e.preventDefault()
                  setIsXaiModalOpen(true)
                }}
                className="mt-auto self-start bg-white text-black px-5 py-2.5 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/90 transition-colors"
              >
                Expand view <ArrowUpRight className="h-4 w-4" />
              </button>
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
              <span className="bg-white/10 px-4 py-1.5 rounded-full text-xs font-medium text-white/80">
                Latest epoch
              </span>
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
              <a
                href={results.mask_url ?? '#'}
                download
                target="_blank"
                rel="noopener noreferrer"
                className="bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center p-4 hover:bg-white/10 transition-colors group"
              >
                <Download className="h-6 w-6 text-white/50 mb-2 group-hover:text-white" />
                <span className="text-xs font-semibold text-white">Raw mask</span>
              </a>
              <button 
                onClick={() => setIsDicomModalOpen(true)}
                className="bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center p-4 hover:bg-white/10 transition-colors group"
              >
                <FileText className="h-6 w-6 text-white/50 mb-2 group-hover:text-neuro-cyan" />
                <span className="text-xs font-semibold text-white">DICOM data</span>
              </button>
            </div>
            <button 
              onClick={() => setIsGuidelinesModalOpen(true)}
              className="bg-white/5 border border-white/10 rounded-2xl p-5 flex items-center justify-between hover:bg-white/10 transition-colors text-left group"
            >
              <div>
                <h4 className="text-sm font-semibold text-white mb-0.5">Clinical guidelines</h4>
                <p className="text-[11px] text-white/40">Review WHO tumor grading metrics</p>
              </div>
              <ArrowUpRight className="h-4 w-4 text-white/40 group-hover:text-white" />
            </button>
            <button 
              onClick={handleFlagForReview}
              className="bg-destructive/10 border border-destructive/25 rounded-2xl p-5 flex items-center justify-between hover:bg-destructive/20 transition-colors text-left group"
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
        )}

        {/* Other Tabs Content placeholders */}
        {activeTab === "XAI Heatmaps" && (
          <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-white/5 p-8 flex flex-col items-center justify-center animate-in fade-in duration-300 min-h-[400px]">
             <Activity className="h-12 w-12 text-neuro-cyan mb-4 opacity-50" />
             <h3 className="text-xl font-semibold text-white mb-2">XAI Activation Maps</h3>
             <p className="text-white/40 max-w-md text-center mb-6">Detailed gradient-weighted class activation mapping (Grad-CAM) visualization for anomaly explanation.</p>
             <button onClick={() => setIsXaiModalOpen(true)} className="bg-white text-black px-6 py-3 rounded-full font-semibold text-sm">
                View Fullscreen Heatmap
             </button>
          </div>
        )}
        
        {activeTab === "Patient Metadata" && (
          <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-white/5 p-8 animate-in fade-in duration-300 min-h-[400px]">
             <h3 className="text-xl font-semibold text-white mb-4">Patient Metadata</h3>
             <div className="bg-black/20 rounded-xl p-4 font-mono text-sm text-white/70">
                <p>Scan ID: {results.scanId}</p>
                <p>Date: {new Date().toLocaleDateString()}</p>
                <p>Modality: MRI</p>
                <p>Sequence: T1/T2 FLAIR</p>
             </div>
          </div>
        )}
        
        {activeTab === "Network Confidence" && (
           <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-white/5 p-8 animate-in fade-in duration-300 min-h-[400px]">
              <h3 className="text-xl font-semibold text-white mb-4">Network Telemetry</h3>
              <p className="text-white/40 mb-4">Confidence score distribution and layer-wise feature analysis.</p>
              <div className="text-4xl font-semibold text-neuro-lime">{results.confidence.toFixed(2)}%</div>
           </div>
        )}
        
        {activeTab === "DICOM Export" && (
           <div className="flex-1 rounded-[1.75rem] border border-white/10 bg-white/5 p-8 animate-in fade-in duration-300 min-h-[400px]">
              <h3 className="text-xl font-semibold text-white mb-4">DICOM Export</h3>
              <p className="text-white/40 mb-6">Download the raw DICOM files or standard NIfTI formats.</p>
              <button className="bg-white/10 border border-white/20 text-white px-6 py-3 rounded-full font-semibold text-sm flex items-center gap-2 hover:bg-white/20 transition-colors">
                 <Download className="h-4 w-4" /> Download Bundle
              </button>
           </div>
        )}

      </div>

      {/* Modals */}
      {isXaiModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="bg-[#111] border border-white/10 rounded-3xl p-6 max-w-4xl w-full flex flex-col shadow-2xl relative">
            <button onClick={() => setIsXaiModalOpen(false)} className="absolute top-4 right-4 text-white/50 hover:text-white bg-white/5 p-2 rounded-full transition-colors">
              <Plus className="h-5 w-5 rotate-45" />
            </button>
            <h3 className="text-xl font-semibold text-white mb-6">High-Resolution XAI Heatmap</h3>
            <div className="aspect-video bg-black rounded-xl border border-white/10 overflow-hidden relative mb-4">
              <div
                className="absolute inset-0 bg-cover bg-center"
                style={{ backgroundImage: results.xai_url ? `url('${results.xai_url}')` : "url('/mri-heatmap.png')" }}
              />
            </div>
            <p className="text-white/60 text-sm">Grad-CAM activation highlights regions contributing to the AI's anomaly detection.</p>
          </div>
        </div>
      )}

      {isDicomModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="bg-[#111] border border-white/10 rounded-3xl p-6 max-w-2xl w-full flex flex-col shadow-2xl relative">
            <button onClick={() => setIsDicomModalOpen(false)} className="absolute top-4 right-4 text-white/50 hover:text-white bg-white/5 p-2 rounded-full transition-colors">
              <Plus className="h-5 w-5 rotate-45" />
            </button>
            <h3 className="text-xl font-semibold text-white mb-4">Raw DICOM Metadata</h3>
            <div className="bg-black border border-white/10 rounded-xl p-4 overflow-auto max-h-[60vh]">
              <pre className="text-xs text-neuro-cyan font-mono whitespace-pre-wrap">
{JSON.stringify({
  PatientID: "ANON-9831",
  StudyDate: new Date().toISOString().split('T')[0],
  Modality: "MR",
  Manufacturer: "Siemens Healthineers",
  MagneticFieldStrength: "3.0T",
  SliceThickness: "1.0mm",
  SpacingBetweenSlices: "1.0mm",
  RepetitionTime: 2000,
  EchoTime: 20,
  InversionTime: 800,
  FlipAngle: 90,
  PixelSpacing: [1.0, 1.0],
  WindowCenter: 400,
  WindowWidth: 1000
}, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {isGuidelinesModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="bg-[#111] border border-white/10 rounded-3xl p-6 max-w-md w-full flex flex-col shadow-2xl relative">
            <button onClick={() => setIsGuidelinesModalOpen(false)} className="absolute top-4 right-4 text-white/50 hover:text-white bg-white/5 p-2 rounded-full transition-colors">
              <Plus className="h-5 w-5 rotate-45" />
            </button>
            <h3 className="text-xl font-semibold text-white mb-4">Clinical Guidelines</h3>
            <div className="space-y-4 text-sm text-white/70">
              <p><strong className="text-white">WHO Tumor Grading:</strong></p>
              <ul className="list-disc pl-5 space-y-2">
                <li><span className="text-neuro-lime">Grade I:</span> Slow-growing, benign.</li>
                <li><span className="text-yellow-400">Grade II:</span> Relatively slow-growing, slightly more abnormal cells.</li>
                <li><span className="text-orange-400">Grade III:</span> Malignant, actively reproducing abnormal cells.</li>
                <li><span className="text-red-500">Grade IV:</span> Most malignant, rapid growth, aggressive features (e.g., glioblastoma).</li>
              </ul>
              <p className="text-xs text-white/40 mt-4 pt-4 border-t border-white/10">Reference: World Health Organization Classification of Tumors of the Central Nervous System.</p>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
