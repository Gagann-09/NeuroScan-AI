import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createApiClient } from '@/lib/api'

describe('API Client', () => {
  const mockGetIdToken = vi.fn()
  const mockOnUnauthorized = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn()
  })

  afterEach(() => {
    vi.resetAllMocks()
  })

  it('attaches Authorization header when token is available', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({ scan_id: 'test-scan', status: 'PROCESSING', message: 'OK' }),
    }
    ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    await api.uploadScan({
      t1: new File([''], 't1.nii'),
      t1ce: new File([''], 't1ce.nii'),
      t2: new File([''], 't2.nii'),
      flair: new File([''], 'flair.nii'),
    })

    expect(global.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/v1/scans/upload',
      expect.objectContaining({
        method: 'POST',
        headers: expect.any(Headers),
      })
    )

    const fetchCall = (global.fetch as vi.Mock).mock.calls[0]
    const headers = fetchCall[1]?.headers as Headers
    expect(headers.get('Authorization')).toBe('Bearer test-id-token')
  })

  it('does not attach Authorization header when token is null', async () => {
    mockGetIdToken.mockResolvedValue(null)

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({ scan_id: 'test-scan', status: 'PROCESSING', message: 'OK' }),
    }
    ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    await api.uploadScan({
      t1: new File([''], 't1.nii'),
      t1ce: new File([''], 't1ce.nii'),
      t2: new File([''], 't2.nii'),
      flair: new File([''], 'flair.nii'),
    })

    const fetchCall = (global.fetch as vi.Mock).mock.calls[0]
    const headers = fetchCall[1]?.headers as Headers
    expect(headers.get('Authorization')).toBeNull()
  })

  it('calls onUnauthorized and throws on 401 response', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: false,
      status: 401,
      json: vi.fn().mockResolvedValue({ detail: 'Invalid token' }),
    }
    ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await expect(
      api.getScanStatus('test-scan-id')
    ).rejects.toThrow()

    expect(mockOnUnauthorized).toHaveBeenCalledTimes(1)
  })

  it('calls onUnauthorized on 401 for getScanResults', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: false,
      status: 401,
      json: vi.fn().mockResolvedValue({ detail: 'Invalid token' }),
    }
    ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await expect(
      api.getScanResults('test-scan-id')
    ).rejects.toThrow()

    expect(mockOnUnauthorized).toHaveBeenCalledTimes(1)
  })

  it('parses error detail from response', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: false,
      status: 500,
      json: vi.fn().mockResolvedValue({ detail: 'Internal server error' }),
    }
    ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await expect(
      api.getScanStatus('test-scan-id')
    ).rejects.toThrow('Internal server error')

    expect(mockOnUnauthorized).not.toHaveBeenCalled()
  })

  describe('Kaggle classification API', () => {
    it('uploadKaggleScan attaches Authorization header and posts to kaggle upload endpoint', async () => {
      mockGetIdToken.mockResolvedValue('test-id-token')

      const mockResponse = {
        ok: true,
        json: vi.fn().mockResolvedValue({ scan_id: 'kaggle-scan-123', status: 'PROCESSING' }),
      }
      ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

      const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
      await api.uploadKaggleScan(new File([''], 'mri.jpg', { type: 'image/jpeg' }))

      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/v1/scans/kaggle/upload',
        expect.objectContaining({
          method: 'POST',
          headers: expect.any(Headers),
        })
      )

      const fetchCall = (global.fetch as vi.Mock).mock.calls[0]
      const headers = fetchCall[1]?.headers as Headers
      expect(headers.get('Authorization')).toBe('Bearer test-id-token')
    })

    it('uploadKaggleScan throws on non-ok response', async () => {
      mockGetIdToken.mockResolvedValue('test-id-token')

      const mockResponse = {
        ok: false,
        status: 400,
        json: vi.fn().mockResolvedValue({ detail: 'File must be .jpg, .jpeg, or .png format' }),
      }
      ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

      const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

      await expect(
        api.uploadKaggleScan(new File([''], 'bad.xyz'))
      ).rejects.toThrow('File must be .jpg, .jpeg, or .png format')
    })

    it('getKaggleScanStatus calls kaggle status endpoint', async () => {
      mockGetIdToken.mockResolvedValue('test-id-token')

      const mockResponse = {
        ok: true,
        json: vi.fn().mockResolvedValue({ scan_id: 'kaggle-scan-123', status: 'COMPLETED' }),
      }
      ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

      const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
      const result = await api.getKaggleScanStatus('kaggle-scan-123')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/v1/scans/kaggle/status/kaggle-scan-123',
        expect.anything()
      )
      expect(result.status).toBe('COMPLETED')
    })

    it('getKaggleScanResults calls kaggle results endpoint', async () => {
      mockGetIdToken.mockResolvedValue('test-id-token')

      const mockResponse = {
        ok: true,
        json: vi.fn().mockResolvedValue({
          scan_id: 'kaggle-scan-123',
          class_index: 1,
          class_name: 'meningioma',
          probabilities: { glioma: 0.12, meningioma: 0.78, notumor: 0.05, pituitary: 0.05 },
          model_version: 'kaggle-cnn-v1.0',
          image_url: '/kaggle-mri-sample.jpg',
        }),
      }
      ;(global.fetch as vi.Mock).mockResolvedValue(mockResponse)

      const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
      const result = await api.getKaggleScanResults('kaggle-scan-123')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/v1/scans/kaggle/results/kaggle-scan-123',
        expect.anything()
      )
      expect(result.class_name).toBe('meningioma')
      expect(result.image_url).toBe('/kaggle-mri-sample.jpg')
    })
  })
})