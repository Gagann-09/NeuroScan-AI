export type ScanResults = {
  scanId: string
  confidence: number | null
  anomalyArea: number | null
  predictedMaskPixels: number | null
  sliceCoveragePercent: number | null

  // Model-derived classification is distinct from clinical diagnosis.
  tumorType: string
  tumorTypeSource: string
  tumorGrade: string
  tumorGradeSource: string
  datasetCohortLabel: string | null
  datasetCohortSource: string | null

  // BraTS dataset-cohort HGG/LGG classifier output (not WHO grade, cancer
  // stage, histological diagnosis, or clinical diagnosis).
  hggLggClass: string | null
  hggLggProbability: number | null
  hggLggSource: string | null

  volume: number | null
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
  classProbabilities: {
    label: string
    value: number | null
    accent: "cyan" | "lime" | "violet" | "muted"
  }[]
}

// Kaggle classification results
export type KaggleResults = {
  scanId: string
  class_name: string
  class_index: number
  probabilities: Record<string, number>
  model_version: string
  image_url: string
}

// Simulated ARMT-GAN inference output scaffold.
export const MOCK_RESULTS: ScanResults = {
  predictedMaskPixels: null,
  sliceCoveragePercent: null,
  scanId: "SCN-4F9A2C",
  confidence: 98.7,
  anomalyArea: 14.3,
  tumorType: "Not determined",
  tumorTypeSource: "Demo data - no classifier result",
  tumorGrade: "Not determined",
  tumorGradeSource: "Demo data - no validated grading result",
  datasetCohortLabel: null,
  datasetCohortSource: null,
  hggLggClass: null,
  hggLggProbability: null,
  hggLggSource: null,
  volume: 32.6,
  confidencePills: 6,
  anomalyPills: 4,
  maskUrl: "/mri-segmentation.png",
  reportUrl: "/mri-heatmap.png",
  mask_url: "/mri-segmentation.png",
  xai_url: "/mri-heatmap.png",

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
    { label: "Inference latency", value: "2.1s", sub: "ARMT-GAN forward pass" },
    { label: "Model parameters", value: "7.7M", sub: "PyTorch float32" },
    { label: "Benchmark Dice", value: "N/A", sub: "Unverified reference" },
    { label: "Boundary error", value: "N/A", sub: "Unverified reference" },
  ],
  classProbabilities: [
    { label: "Whole tumor (Binary)", value: 98.7, accent: "lime" },
    { label: "Enhancing tumor (ET)", value: null, accent: "muted" },
    { label: "Peritumoral edema (ED)", value: null, accent: "muted" },
    { label: "Necrotic core (NCR)", value: null, accent: "muted" },
  ],
}

// Kaggle classification mock results
export const MOCK_KAGGLE_RESULTS: KaggleResults = {
  scanId: "SCN-KAGGLE-001",
  class_name: "meningioma",
  class_index: 1,
  probabilities: {
    glioma: 0.12,
    meningioma: 0.78,
    notumor: 0.05,
    pituitary: 0.05,
  },
  model_version: "kaggle-cnn-v1.0",
  image_url: "/kaggle-mri-sample.jpg",
}

export const PROCESSING_STAGES = [
  "Normalizing voxel intensities",
  "Skull-stripping volume",
  "Isolating tensors",
  "Executing ARMT-GAN inference",
  "Generating XAI activation map",
]
