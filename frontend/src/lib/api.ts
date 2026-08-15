
const API_BASE_URL = "http://127.0.0.1:8000";

export async function uploadScan(file: File): Promise<{ scan_id: string; status: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/v1/scans/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to upload scan to backend.");
    }

    return response.json();
}

export async function getScanStatus(scanId: string): Promise<{ scan_id: string; status: string; file_type?: string }> {
    const response = await fetch(`${API_BASE_URL}/api/v1/scans/status/${scanId}`, {
        method: "GET",
    });

    if (!response.ok) {
        throw new Error("Failed to fetch scan status.");
    }

    return response.json();
}

export async function getScanResults(scanId: string): Promise<{ scan_id: string; mask_url: string; xai_url: string; report_url?: string }> {
    const response = await fetch(`${API_BASE_URL}/api/v1/scans/results/${scanId}`, {
        method: "GET",
    });

    if (!response.ok) {
        throw new Error("Failed to fetch scan results.");
    }

    const data = await response.json();

    // Normalize URLs if they are relative paths from backend
    const fixUrl = (url: string) => (url && url.startsWith("http") ? url : `${API_BASE_URL}${url}`);

    return {
        scan_id: data.scan_id,
        mask_url: fixUrl(data.mask_url),
        xai_url: fixUrl(data.xai_url),
        report_url: data.report_url ? fixUrl(data.report_url) : undefined,
    };
}