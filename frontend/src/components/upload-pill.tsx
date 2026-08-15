"use client"

import type React from "react"
import { useRef, useState, useEffect } from "react"
import { ArrowUp, Plus, Activity, ImageIcon, Box, X } from "lucide-react"

type UploadPillProps = {
  onAnalyze: (file: File) => void
}

export function UploadPill({ onAnalyze }: UploadPillProps) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

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
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const acceptFile = (selected: File) => {
    setFile(selected)
    setIsMenuOpen(false)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(selected.type.startsWith("image/") ? URL.createObjectURL(selected) : null)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) acceptFile(e.target.files[0])
  }

  const clearFile = () => {
    setFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.[0]) acceptFile(e.dataTransfer.files[0])
  }

  return (
    <div className="w-full max-w-2xl animate-in fade-in zoom-in-95 duration-500">
      <div className="mb-8 text-center">
        <h1 className="text-balance text-4xl sm:text-5xl font-semibold tracking-tight text-white">
          Segment tumors in seconds
        </h1>
        <p className="mt-3 text-pretty text-sm sm:text-base text-white/50 leading-relaxed max-w-md mx-auto">
          Drop an MRI slice or 3D NIfTI volume. The ARMT-GAN pipeline returns an explainable segmentation with
          clinical-grade confidence.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`bg-card/80 backdrop-blur-3xl border shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex flex-col transition-all duration-300 ${
          isDragging ? "border-neuro-cyan/70 ring-2 ring-neuro-cyan/30" : "border-white/10"
        } ${file ? "rounded-3xl p-4" : "rounded-full p-2"}`}
      >
        {file && (
          <div className="flex items-center gap-3 mb-3 ml-1 border-b border-white/5 pb-3">
            <div className="relative group">
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl || "/placeholder.svg"}
                  alt="Selected scan preview"
                  className="h-16 w-16 object-cover rounded-xl border border-white/10"
                />
              ) : (
                <div className="h-16 w-16 bg-neuro-violet/10 border border-neuro-violet/30 flex items-center justify-center rounded-xl">
                  <Box className="h-7 w-7 text-neuro-violet" />
                </div>
              )}
              <button
                onClick={clearFile}
                aria-label="Remove file"
                className="absolute -top-2 -right-2 bg-white/10 hover:bg-destructive border border-white/20 rounded-full p-1 text-white opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X size={12} />
              </button>
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-sm text-white font-medium truncate max-w-[220px] sm:max-w-[380px]">
                {file.name}
              </span>
              <span className="text-xs text-white/40 font-mono">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        )}

        <div className="flex items-center gap-1">
          <div className="relative" ref={menuRef}>
            <button
              onClick={() => setIsMenuOpen((v) => !v)}
              aria-label="Add a scan"
              className="w-10 h-10 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center border border-white/5 transition-colors"
            >
              <Plus className="text-white/60 h-5 w-5" />
            </button>
            {isMenuOpen && (
              <div className="absolute bottom-[120%] left-0 w-64 bg-card/95 backdrop-blur-xl border border-white/10 rounded-2xl p-2 shadow-2xl animate-in slide-in-from-bottom-2 fade-in z-50">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.nii,.gz"
                  className="hidden"
                  onChange={handleFileSelect}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full text-left px-4 py-3 rounded-xl hover:bg-white/5 flex items-center gap-3 text-sm text-white/80 font-medium transition-colors"
                >
                  <ImageIcon className="h-4 w-4 text-neuro-cyan" />
                  <span>Add 2D MRI slice (.jpg / .png)</span>
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full text-left px-4 py-3 rounded-xl hover:bg-white/5 flex items-center gap-3 text-sm text-white/80 font-medium transition-colors"
                >
                  <Box className="h-4 w-4 text-neuro-violet" />
                  <span>Add 3D NIfTI volume (.nii / .nii.gz)</span>
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => !file && fileInputRef.current?.click()}
            className="flex-1 px-2 sm:px-3 text-left cursor-pointer"
          >
            <span className={`text-sm sm:text-base font-medium ${file ? "text-white/80" : "text-white/40"}`}>
              {file ? "Ready to analyze scan..." : "Upload an MRI scan or 3D volume..."}
            </span>
          </button>

          <div className="mr-1 px-3 py-1.5 rounded-full bg-white/5 border border-white/5 items-center gap-1.5 cursor-default hidden sm:flex">
            <Activity className="h-3 w-3 text-neuro-cyan" />
            <span className="text-xs font-medium text-white/70">ARMT-GAN</span>
          </div>

          <button
            onClick={() => file && onAnalyze(file)}
            disabled={!file}
            aria-label="Analyze scan"
            className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 shrink-0 ${
              file
                ? "bg-white text-black shadow-[0_0_18px_rgba(255,255,255,0.35)] hover:bg-white/90 cursor-pointer"
                : "bg-white/10 text-white/40 cursor-not-allowed"
            }`}
          >
            <ArrowUp className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  )
}
