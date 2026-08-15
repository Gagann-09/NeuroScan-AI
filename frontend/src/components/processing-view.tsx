"use client"

import { useEffect, useState } from "react"
import { Loader2, Check } from "lucide-react"
import { PROCESSING_STAGES } from "@/lib/mock-data"

export function ProcessingView() {
  const [stage, setStage] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setStage((s) => Math.min(s + 1, PROCESSING_STAGES.length - 1))
    }, 600)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex flex-col items-center animate-in fade-in zoom-in duration-500">
      <div className="w-24 h-24 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-8 shadow-[0_0_50px_rgba(56,224,255,0.2)]">
        <Loader2 className="h-10 w-10 text-neuro-cyan animate-spin" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight text-white mb-1">Analyzing scan</h2>
      <p className="text-white/40 font-mono text-xs tracking-widest uppercase mb-8">ARMT-GAN inference pipeline</p>

      <ol className="w-72 space-y-2.5">
        {PROCESSING_STAGES.map((label, i) => {
          const done = i < stage
          const active = i === stage
          return (
            <li
              key={label}
              className={`flex items-center gap-3 text-sm transition-colors ${
                done ? "text-white/60" : active ? "text-white" : "text-white/25"
              }`}
            >
              <span
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
                  done
                    ? "border-neuro-cyan/50 bg-neuro-cyan/15 text-neuro-cyan"
                    : active
                      ? "border-neuro-cyan text-neuro-cyan"
                      : "border-white/15 text-transparent"
                }`}
              >
                {done ? <Check className="h-3 w-3" /> : active ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
              </span>
              {label}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
