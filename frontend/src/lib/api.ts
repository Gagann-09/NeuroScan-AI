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
  anomaly_area_cm2: number | null
  confidence_score: number | null
  who_grade: string | null
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

export async function uploadScan(
  files: BraTSFiles,
): Promise<UploadResponse> {
  const formData = new FormData()

  formData.append("t1", files.t1)
  formData.append("t1ce", files.t1ce)
  formData.append("t2", files.t2)
  formData.append("flair", files.flair)

  const response = await fetch(
    `${API_BASE_URL}/api/v1/scans/upload`,
    {
      method: "POST",
      body: formData,
    },
  )

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  return response.json()
}

export async function getScanStatus(
  scanId: string,
): Promise<ScanStatus> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/scans/status/${scanId}`,
  )

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  return response.json()
}

export async function getScanResults(
  scanId: string,
): Promise<ScanResults> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/scans/results/${scanId}`,
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
    anomaly_area_cm2:
      data.anomaly_area_cm2 ?? null,
    confidence_score:
      data.confidence_score ?? null,
    who_grade:
      data.who_grade ?? null,
  }
}