export type ScanResults = {
  scanId: string
  confidence: number
  anomalyArea: number
  tumorGrade: string
  volume: number
  layers: { label: string; robustness: number; adversarial: number; peak?: boolean }[]
  confidencePills: number
  anomalyPills: number
  // Real backend asset URLs — populated after SEGMENTED status
  mask_url?: string
  xai_url?: string
  report_url?: string
}


// Simulated ARMT-GAN inference output.
export const MOCK_RESULTS: ScanResults = {
  scanId: "SCN-4F9A2C",
  confidence: 98.7,
  anomalyArea: 14.3,
  tumorGrade: "Grade II",
  volume: 32.6,
  confidencePills: 6,
  anomalyPills: 4,
  layers: [
    { label: "L1", robustness: 80, adversarial: 40 },
    { label: "L2", robustness: 60, adversarial: 30 },
    { label: "L3", robustness: 90, adversarial: 20 },
    { label: "L4", robustness: 100, adversarial: 45, peak: true },
    { label: "L5", robustness: 50, adversarial: 40 },
    { label: "L6", robustness: 60, adversarial: 35 },
  ],
}

export const PROCESSING_STAGES = [
  "Normalizing voxel intensities",
  "Skull-stripping volume",
  "Isolating tensors",
  "Executing ARMT-GAN inference",
  "Generating XAI activation map",
]
