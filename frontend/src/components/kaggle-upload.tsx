"use client"

import type React from "react"
import { useRef, useState, useEffect } from "react"
import { ArrowUp, Plus, Activity, ImageIcon, Box, X, CheckCircle } from "lucide-react"

type KaggleFile = File | null

type KaggleUploadPillProps = {
  onAnalyze: (file: File) => void
}

export function KaggleUploadPill({ onAnalyze }: KaggleUploadPillProps) {
  const [file, setFile] = useState<KaggleFile>(null)
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
  const maxSize = 20 * 1024 * 1024
  const allowedTypes = ["image/jpeg", "image/png"]
  const allowedExtensions = /\.(jpe?g|png)$/i

  if (
    !allowedTypes.includes(selected.type) ||
    !allowedExtensions.test(selected.name)
  ) {
    window.alert("Please select a JPG or PNG brain MRI image.")
    return
  }

  if (selected.size > maxSize) {
    window.alert("The image must be 20 MB or smaller.")
    return
  }

  setFile(selected)
  setIsMenuOpen(false)

  if (previewUrl) URL.revokeObjectURL(previewUrl)
  setPreviewUrl(
    selected.type.startsWith("image/")
      ? URL.createObjectURL(selected)
      : null
  )
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files?.[0]) acceptFile(e.dataTransfer.files[0])
  }

  return (
    <div className="w-full max-w-3xl animate-in fade-in zoom-in-95 duration-500">
      <div className="mb-8 text-center">
        <h1 className="text-balance text-4xl sm:text-5xl font-semibold tracking-tight text-white">
          Classify brain tumors
        </h1>
        <p className="mt-3 text-pretty text-sm sm:text-base text-white/50 leading-relaxed max-w-md mx-auto">
          Upload a single brain MRI image (JPG/PNG). The classifier returns tumor type probabilities.
        </p>
      </div>

      <div className={`bg-card/80 backdrop-blur-3xl border shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex flex-col transition-all duration-300
        ${isDragging ? "border-neuro-cyan/70 ring-2 ring-neuro-cyan/30" : file ? "border-neuro-cyan/50" : "border-white/10 hover:border-white/20 cursor-pointer"}
        rounded-2xl p-4 min-h-[280px}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => {
          if (!file) {
            document.getElementById("kaggle-file-input")?.click()
          }
        }}
      >
        <input
          type="file"
          id="kaggle-file-input"
          accept=".jpg,.jpeg,.png"
          className="hidden"
          onChange={handleFileSelect}
        />
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-white/70 uppercase tracking-wide">
            Kaggle MRI Image
          </span>
          {file && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                clearFile()
              }}
              aria-label="Remove image"
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
                  alt="MRI preview"
                  className="h-48 w-48 max-w-full object-contain rounded-lg border border-white/10 mx-auto mb-2"
                />
              ) : (
                <div className="h-48 w-48 bg-neuro-violet/10 border border-neuro-violet/30 flex items-center justify-center rounded-lg mx-auto mb-2">
                  <Box className="h-10 w-10 text-neuro-violet" />
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
              <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <span className="text-sm">Drop JPG/PNG MRI image</span>
              <span className="text-xs block opacity-70">Max 20MB • .jpg, .jpeg, .png</span>
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

      <div className="flex items-center gap-3 justify-center mt-6">
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-colors ${
          file
            ? "bg-neuro-cyan/20 border-neuro-cyan/50 text-neuro-cyan"
            : "bg-white/5 border-white/10 text-white/50"
        }`}>
          <Activity className={`h-4 w-4 ${file ? "text-neuro-cyan animate-pulse" : "text-white/40"}`} />
          <span className="text-sm font-medium">
            {file ? "Image ready for classification" : "No image uploaded"}
          </span>
        </div>

        <div className="mr-1 px-3 py-1.5 rounded-full bg-white/5 border border-white/5 items-center gap-1.5 cursor-default hidden sm:flex">
          <Activity className="h-3 w-3 text-neuro-cyan" />
          <span className="text-xs font-medium text-white/70">Kaggle CNN</span>
        </div>

        <button
          onClick={() => file && onAnalyze(file)}
          disabled={!file}
          aria-label="Classify image"
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 shrink-0 ${
            file
              ? "bg-white text-black shadow-[0_0_18px_rgba(255,255,255,0.35)] hover:bg-white/90 cursor-pointer"
              : "bg-white/10 text-white/40 cursor-not-allowed"
          }`}
        >
          <ArrowUp className="h-6 w-6" />
        </button>
      </div>

      {!file && (
        <p className="mt-4 text-center text-xs text-white/40">
          Required: Single brain MRI image (.jpg, .jpeg, or .png) up to 20MB
        </p>
      )}
    </div>
  )
}