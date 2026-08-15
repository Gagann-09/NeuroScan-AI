import React from "react";
import NeuralBackground from "@/components/NeuralBackground";
import UploadCard from "@/components/UploadCard";

export default function HomePage() {
  return (
    <main className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-[#050505] overflow-hidden">
      <NeuralBackground />

      <header className="z-10 text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2">
          NeuroScan <span className="text-[#00f0ff]">AI</span>
        </h1>
        <p className="text-sm md:text-base text-gray-400 max-w-md mx-auto">
          Adversarially Robust Medical Tumor Segmentation with Explainable Heatmap Verification
        </p>
      </header>

      <UploadCard />
    </main>
  );
}