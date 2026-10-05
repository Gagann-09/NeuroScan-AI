"use client"

import type React from "react"
import { useRef, useState, useEffect } from "react"
import { ArrowUp, Plus, Activity, ImageIcon, Box, X, CheckCircle } from "lucide-react"

type BraTSModality = "t1" | "t1ce" | "t2" | "flair"

const MODALITY_LABELS: Record<BraTSModality, string> = {
  t1: "T1",
  t1ce: "T1ce",
  t2: "T2",
  flair: "FLAIR",
}

const MODALITY_DESCRIPTIONS: Record<BraTSModality, string> = {
  t1: "T1-weighted",
  t1ce: "T1-weighted contrast-enhanced",
  t2: "T2-weighted",
  flair: "FLAIR",
}

type UploadPillProps = {
  onAnalyze: (files: Record<BraTSModality, File>) => void
}

export function UploadPill({ onAnalyze }: UploadPillProps) {
  const [files, setFiles] = useState<Partial<Record<BraTSModality, File>>>({})
  const [previewUrls, setPreviewUrls] = useState<Partial<Record<BraTSModality, string>>>({})
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isDragging, setIsDragging] = useState<BraTSModality | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  useEffect(() => {
    return () => {
      Object.values(previewUrls).forEach((url) => {
        if (url) URL.revokeObjectURL(url)
      })
    }
  }, [previewUrls])

  const acceptFile = (modality: BraTSModality, selected: File) => {
    setFiles((prev) => ({ ...prev, [modality]: selected }))
    setIsMenuOpen(false)
    if (previewUrls[modality]) URL.revokeObjectURL(previewUrls[modality]!)
    setPreviewUrls((prev) => ({
      ...prev,
      [modality]: selected.type.startsWith("image/") ? URL.createObjectURL(selected) : null,
    }))
  }

  const handleFileSelect = (modality: BraTSModality) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) acceptFile(modality, e.target.files[0])
  }

  const clearFile = (modality: BraTSModality) => {
    setFiles((prev) => {
      const next = { ...prev }
      delete next[modality]
      return next
    })
    if (previewUrls[modality]) URL.revokeObjectURL(previewUrls[modality]!)
    setPreviewUrls((prev) => {
      const next = { ...prev }
      delete next[modality]
      return next
    })
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleDragOver = (modality: BraTSModality) => (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(modality)
  }

  const handleDragLeave = () => {
    setIsDragging(null)
  }

  const handleDrop = (modality: BraTSModality) => (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(null)
    if (e.dataTransfer.files?.[0]) acceptFile(modality, e.dataTransfer.files[0])
  }

  const isComplete = Object.keys(files).length === 4
  const modalities: BraTSModality[] = ["t1", "t1ce", "t2", "flair"]

  return (
    <div className="w-full max-w-3xl animate-in fade-in zoom-in-95 duration-500">
      <div className="mb-8 text-center">
        <h1 className="text-balance text-4xl sm:text-5xl font-semibold tracking-tight text-white">
          Segment tumors in seconds
        </h1>
        <p className="mt-3 text-pretty text-sm sm:text-base text-white/50 leading-relaxed max-w-md mx-auto">
          Upload all four BraTS MRI modalities. The ARMT-GAN pipeline returns an explainable segmentation.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {modalities.map((modality) => {
          const file = files[modality]
          const previewUrl = previewUrls[modality]
          const dragging = isDragging === modality

          return (
            <div
              key={modality}
              onDragOver={handleDragOver(modality)}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop(modality)}
              className={`relative bg-card/80 backdrop-blur-3xl border shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex flex-col transition-all duration-300 ${
                dragging
                  ? "border-neuro-cyan/70 ring-2 ring-neuro-cyan/30"
                  : file
                  ? "border-neuro-cyan/50"
                  : "border-white/10"
              } rounded-2xl p-4 min-h-[180px]`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-white/70 uppercase tracking-wide">
                  {MODALITY_LABELS[modality]}
                </span>
                {file && (
                  <button
                    onClick={() => clearFile(modality)}
                    aria-label={`Remove ${MODALITY_LABELS[modality]}`}
                    className="bg-white/10 hover:bg-destructive border border-white/20 rounded-full p-1 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>

              <div className="flex-1 flex items-center justify-center">
                {file ? (
                  <div className="text-center">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt={`${MODALITY_LABELS[modality]} preview`}
                        className="h-24 w-24 object-cover rounded-lg border border-white/10 mx-auto mb-2"
                      />
                    ) : (
                      <div className="h-24 w-24 bg-neuro-violet/10 border border-neuro-violet/30 flex items-center justify-center rounded-lg mx-auto mb-2">
                        <Box className="h-8 w-8 text-neuro-violet" />
                      </div>
                    )}
                    <span className="text-xs text-white/60 font-mono truncate block max-w-full">
                      {file.name}
                    </span>
                    <span className="text-xs text-white/40 font-mono">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                ) : (
                  <div className="text-center text-white/40">
                    <ImageIcon className="h-10 w-10 mx-auto mb-2 opacity-50" />
                    <span className="text-sm">Drop {MODALITY_LABELS[modality]}</span>
                    <span className="text-xs block opacity-70">{MODALITY_DESCRIPTIONS[modality]}</span>
                  </div>
                )}
              </div>

              {file && (
                <div className="mt-2 flex items-center justify-center text-xs text-neuro-cyan">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Ready
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex items-center gap-3 justify-center">
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-colors ${
          isComplete
            ? "bg-neuro-cyan/20 border-neuro-cyan/50 text-neuro-cyan"
            : "bg-white/5 border-white/10 text-white/50"
        }`}>
          <Activity className={`h-4 w-4 ${isComplete ? "text-neuro-cyan animate-pulse" : "text-white/40"}`} />
          <span className="text-sm font-medium">
            {isComplete ? "All 4 modalities ready" : `${Object.keys(files).length}/4 modalities uploaded`}
          </span>
        </div>

        <div className="mr-1 px-3 py-1.5 rounded-full bg-white/5 border border-white/5 items-center gap-1.5 cursor-default hidden sm:flex">
          <Activity className="h-3 w-3 text-neuro-cyan" />
          <span className="text-xs font-medium text-white/70">ARMT-GAN</span>
        </div>

        <button
          onClick={() => isComplete && onAnalyze(files as Record<BraTSModality, File>)}
          disabled={!isComplete}
          aria-label="Analyze scan"
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 shrink-0 ${
            isComplete
              ? "bg-white text-black shadow-[0_0_18px_rgba(255,255,255,0.35)] hover:bg-white/90 cursor-pointer"
              : "bg-white/10 text-white/40 cursor-not-allowed"
          }`}
        >
          <ArrowUp className="h-6 w-6" />
        </button>
      </div>

      {!isComplete && (
        <p className="mt-4 text-center text-xs text-white/40">
          Required: T1, T1ce, T2, and FLAIR NIfTI files (.nii or .nii.gz)
        </p>
      )}
    </div>
  )
}