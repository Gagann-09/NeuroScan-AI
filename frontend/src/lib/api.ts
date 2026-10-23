const API_BASE_URL = "http://127.0.0.1:8000"

export type BraTSModality =
  | "t1"
  | "t1ce"
  | "t2"
  | "flair"

export type BraTSFiles = Record<BraTSModality, File>

export type UploadResponse = {
  scan_id: string
  status: string
  message: string
}

export type ScanStatus = {
  scan_id: string
  status: string
}


export type ScanResults = {
  scan_id: string
  mask_url: string | null
  xai_url: string | null
  report_url: string | null
  tumor_detected: boolean
  max_tumor_probability: number | null
  predicted_mask_pixels: number | null
  slice_coverage_percent: number | null
  model_version: string | null
  tumor_type: string | null
  tumor_type_source: string
  tumor_grade: string | null
  tumor_grade_source: string
  dataset_cohort_label: string | null
  dataset_cohort_source: string | null
  anomaly_area_cm2?: number | null
  confidence_score?: number | null
  who_grade?: string | null
}

/**
 * Kaggle classification API response types.
 */
export type KaggleScanResults = {
  scan_id: string
  class_index: number
  class_name: string
  probabilities: Record<string, number>
  model_version: string
  image_url: string
}

/**
 * Authenticated fetch wrapper that attaches Firebase ID token
 * and handles 401 responses by clearing auth state.
 */
async function authedFetch(
  getIdToken: () => Promise<string | null>,
  onUnauthorized: () => void,
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = await getIdToken()

  const headers = new Headers(options.headers)
  if (token) {
    headers.set("Authorization", `Bearer ${token}`)
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    onUnauthorized()
  }

  return response
}

async function parseError(response: Response): Promise<string> {
  const data = await response.json().catch(() => null)

  if (data?.detail) {
    return String(data.detail)
  }

  if (data?.message) {
    return String(data.message)
  }

  return `Request failed with HTTP ${response.status}.`
}

/**
 * Create API functions with authentication support.
 * The getIdToken and onUnauthorized callbacks are provided by the caller
 * via the useAuth hook in components.
 */
export function createApiClient(
  getIdToken: () => Promise<string | null>,
  onUnauthorized: () => void
) {
  return {
    async uploadScan(
      files: BraTSFiles,
      datasetSubjectId?: string
    ): Promise<UploadResponse> {
      const formData = new FormData()

      formData.append("t1", files.t1)
      formData.append("t1ce", files.t1ce)
      formData.append("t2", files.t2)
      formData.append("flair", files.flair)

      const normalizedSubjectId = datasetSubjectId?.trim()
      if (normalizedSubjectId) {
        formData.append("dataset_subject_id", normalizedSubjectId)
      }

      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/upload`,
        {
          method: "POST",
          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      return response.json()
    },

    async getScanStatus(scanId: string): Promise<ScanStatus> {
      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/status/${scanId}`
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      return response.json()
    },

    async getScanResults(scanId: string): Promise<ScanResults> {
      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/results/${scanId}`
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      const data = await response.json()

      return {
        scan_id: data.scan_id,
        mask_url: data.mask_url ?? null,
        xai_url: data.xai_url ?? null,
        report_url: data.report_url ?? null,
        tumor_detected: Boolean(data.tumor_detected),
        max_tumor_probability: data.max_tumor_probability ?? null,
        predicted_mask_pixels: data.predicted_mask_pixels ?? null,
        slice_coverage_percent: data.slice_coverage_percent ?? null,
        model_version: data.model_version ?? null,
        tumor_type: data.tumor_type ?? null,
        tumor_type_source:
          data.tumor_type_source ?? "Not determined by ARMT-GAN segmentation",
        tumor_grade: data.tumor_grade ?? null,
        tumor_grade_source:
          data.tumor_grade_source ??
          "Not determined; no validated grading output is available",
        dataset_cohort_label: data.dataset_cohort_label ?? null,
        dataset_cohort_source: data.dataset_cohort_source ?? null,
        anomaly_area_cm2: data.anomaly_area_cm2 ?? null,
        confidence_score: data.confidence_score ?? null,
        who_grade: data.who_grade ?? null,
      }
    },

    async uploadKaggleScan(file: File): Promise<{ scan_id: string }> {
      const formData = new FormData()
      formData.append("file", file)

      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/kaggle/upload`,
        {
          method: "POST",
          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      return response.json()
    },

    async getKaggleScanStatus(scanId: string): Promise<{ status: string }> {
      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/kaggle/status/${scanId}`
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      return response.json()
    },

    async getKaggleScanResults(scanId: string): Promise<KaggleScanResults> {
      const response = await authedFetch(
        getIdToken,
        onUnauthorized,
        `${API_BASE_URL}/api/v1/scans/kaggle/results/${scanId}`
      )

      if (!response.ok) {
        throw new Error(await parseError(response))
      }

      return response.json()
    },
  }
}