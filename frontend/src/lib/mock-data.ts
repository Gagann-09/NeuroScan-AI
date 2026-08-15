export type ScanResults = {
  scanId: string
  confidence: number
  anomalyArea: number
  tumorGrade: string
  volume: number
  layers: { label: string; robustness: number; adversarial: number; peak?: boolean }[]
  confidencePills: number
  anomalyPills: number
  maskUrl: string
  reportUrl: string
  mask_url?: string
  xai_url?: string
  report_url?: string
  patient: { label: string; value: string }[]
  dicom: { tag: string; label: string; value: string }[]
  telemetry: { label: string; value: string; sub: string }[]
  classProbabilities: { label: string; value: number; accent: "cyan" | "lime" | "violet" | "muted" }[]
}

// Simulated ARMT-GAN inference output scaffold.
export const MOCK_RESULTS: ScanResults = {
  scanId: "SCN-4F9A2C",
  confidence: 98.7,
  anomalyArea: 14.3,
  tumorGrade: "Grade II",
  volume: 32.6,
  confidencePills: 6,
  anomalyPills: 4,
  maskUrl: "/mri-segmentation.png",
  reportUrl: "/mri-heatmap.png",
  mask_url: "/mri-segmentation.png",
  xai_url: "/mri-heatmap.png",
  layers: [
    { label: "L1", robustness: 80, adversarial: 40 },
    { label: "L2", robustness: 60, adversarial: 30 },
    { label: "L3", robustness: 90, adversarial: 20 },
    { label: "L4", robustness: 100, adversarial: 45, peak: true },
    { label: "L5", robustness: 50, adversarial: 40 },
    { label: "L6", robustness: 60, adversarial: 35 },
  ],
  patient: [
    { label: "Patient ID", value: "PT-90231" },
    { label: "Study date", value: "2026-08-15" },
    { label: "Modality", value: "MR / T1-weighted" },
    { label: "Field strength", value: "3.0 T" },
    { label: "Slice thickness", value: "1.0 mm" },
    { label: "Sequence", value: "MPRAGE axial" },
    { label: "Referring unit", value: "Neuro-Oncology" },
    { label: "Acquisition", value: "192 slices" },
  ],
  dicom: [
    { tag: "(0010,0020)", label: "PatientID", value: "PT-90231" },
    { tag: "(0008,0060)", label: "Modality", value: "MR" },
    { tag: "(0018,0087)", label: "MagneticFieldStrength", value: "3.0" },
    { tag: "(0028,0010)", label: "Rows", value: "256" },
    { tag: "(0028,0011)", label: "Columns", value: "256" },
    { tag: "(0018,0050)", label: "SliceThickness", value: "1.0" },
    { tag: "(0018,0080)", label: "RepetitionTime", value: "2300" },
    { tag: "(0018,0081)", label: "EchoTime", value: "2.98" },
    { tag: "(0008,103E)", label: "SeriesDescription", value: "MPRAGE_AXIAL" },
    { tag: "(0020,0013)", label: "InstanceNumber", value: "042" },
  ],
  telemetry: [
    { label: "Inference latency", value: "1.84s", sub: "ARMT-GAN forward pass" },
    { label: "Active parameters", value: "312M", sub: "quantized int8" },
    { label: "Dice coefficient", value: "0.947", sub: "vs. expert mask" },
    { label: "Hausdorff dist.", value: "2.1mm", sub: "boundary error" },
  ],
  classProbabilities: [
    { label: "Enhancing tumor", value: 98.7, accent: "lime" },
    { label: "Peritumoral edema", value: 74.2, accent: "cyan" },
    { label: "Necrotic core", value: 41.5, accent: "violet" },
    { label: "Healthy tissue", value: 12.3, accent: "muted" },
  ],
}

export const PROCESSING_STAGES = [
  "Normalizing voxel intensities",
  "Skull-stripping volume",
  "Isolating tensors",
  "Executing ARMT-GAN inference",
  "Generating XAI activation map",
]
