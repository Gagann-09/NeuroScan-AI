import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createApiClient } from '@/lib/api'
import type { Mock } from "vitest"

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
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

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

    const fetchCall = (global.fetch as Mock).mock.calls[0]
    const headers = fetchCall[1]?.headers as Headers
    expect(headers.get('Authorization')).toBe('Bearer test-id-token')
  })

  it('does not attach Authorization header when token is null', async () => {
    mockGetIdToken.mockResolvedValue(null)

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({ scan_id: 'test-scan', status: 'PROCESSING', message: 'OK' }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    await api.uploadScan({
      t1: new File([''], 't1.nii'),
      t1ce: new File([''], 't1ce.nii'),
      t2: new File([''], 't2.nii'),
      flair: new File([''], 'flair.nii'),
    })

    const fetchCall = (global.fetch as Mock).mock.calls[0]
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
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await expect(
      api.getScanStatus('test-scan-id')
    ).rejects.toThrow()

    expect(mockOnUnauthorized).toHaveBeenCalledTimes(1)
  })

  it('includes the BraTS subject ID in the upload form when provided', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    ;(global.fetch as Mock).mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        scan_id: 'brats-scan-123',
        status: 'PROCESSING',
        message: 'OK',
      }),
    })

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await api.uploadScan(
      {
        t1: new File(['t1'], 't1.nii'),
        t1ce: new File(['t1ce'], 't1ce.nii'),
        t2: new File(['t2'], 't2.nii'),
        flair: new File(['flair'], 'flair.nii'),
      },
      '  BraTS20_Training_001  ',
    )

    const fetchCall = (global.fetch as Mock).mock.calls[0]
    const formData = fetchCall[1]?.body as FormData

    expect(formData).toBeInstanceOf(FormData)
    expect(formData.get('dataset_subject_id')).toBe('BraTS20_Training_001')
  })

  it('omits the BraTS subject ID from the upload form when blank', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    ;(global.fetch as Mock).mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        scan_id: 'brats-scan-456',
        status: 'PROCESSING',
        message: 'OK',
      }),
    })

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await api.uploadScan(
      {
        t1: new File(['t1'], 't1.nii'),
        t1ce: new File(['t1ce'], 't1ce.nii'),
        t2: new File(['t2'], 't2.nii'),
        flair: new File(['flair'], 'flair.nii'),
      },
      '   ',
    )

    const fetchCall = (global.fetch as Mock).mock.calls[0]
    const formData = fetchCall[1]?.body as FormData

    expect(formData).toBeInstanceOf(FormData)
    expect(formData.has('dataset_subject_id')).toBe(false)
  })

  it('calls onUnauthorized on 401 for getScanResults', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: false,
      status: 401,
      json: vi.fn().mockResolvedValue({ detail: 'Invalid token' }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)

    await expect(
      api.getScanResults('test-scan-id')
    ).rejects.toThrow()

    expect(mockOnUnauthorized).toHaveBeenCalledTimes(1)
  })

  it('maps BraTS tumor metadata returned by the API', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({
        scan_id: 'brats-scan-123',
        mask_url: null,
        xai_url: null,
        report_url: null,
        tumor_detected: true,
        max_tumor_probability: 0.9,
        predicted_mask_pixels: 1200,
        slice_coverage_percent: 2.4,
        model_version: 'armt-gan-test',
        tumor_type: 'Glioma',
        tumor_type_source: 'Dataset context; not predicted by ARMT-GAN',
        tumor_grade: null,
        tumor_grade_source: 'No validated grading output is available',
      }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    const result = await api.getScanResults('brats-scan-123')

    expect(result.tumor_type).toBe('Glioma')
    expect(result.tumor_type_source).toBe(
      'Dataset context; not predicted by ARMT-GAN',
    )
    expect(result.tumor_grade).toBeNull()
    expect(result.tumor_grade_source).toBe(
      'No validated grading output is available',
    )
  })

  it('uses safe fallbacks when BraTS tumor metadata is absent', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({
        scan_id: 'brats-scan-456',
        mask_url: null,
        xai_url: null,
        report_url: null,
        tumor_detected: false,
        max_tumor_probability: null,
        predicted_mask_pixels: null,
        slice_coverage_percent: null,
        model_version: null,
      }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    const result = await api.getScanResults('brats-scan-456')

    expect(result.tumor_type).toBeNull()
    expect(result.tumor_type_source).toBe(
      'Not determined by ARMT-GAN segmentation',
    )
    expect(result.tumor_grade).toBeNull()
    expect(result.tumor_grade_source).toBe(
      'Not determined; no validated grading output is available',
    )
    expect(result.dataset_cohort_label).toBeNull()
    expect(result.dataset_cohort_source).toBeNull()
  })

  it('maps BraTS dataset cohort metadata returned by the API', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: true,
      json: vi.fn().mockResolvedValue({
        scan_id: 'brats-scan-789',
        mask_url: null,
        xai_url: null,
        report_url: null,
        tumor_detected: true,
        max_tumor_probability: 0.9,
        predicted_mask_pixels: 1200,
        slice_coverage_percent: 2.4,
        model_version: 'armt-gan-test',
        tumor_type: null,
        tumor_type_source: 'Not determined by ARMT-GAN segmentation',
        tumor_grade: null,
        tumor_grade_source: 'No validated grading output is available',
        dataset_cohort_label: 'HGG',
        dataset_cohort_source:
          'BraTS2020_TrainingData/name_mapping.csv; supplied subject ID; dataset metadata only, not an ARMT-GAN prediction or WHO grade',
      }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

    const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
    const result = await api.getScanResults('brats-scan-789')

    expect(result.dataset_cohort_label).toBe('HGG')
    expect(result.dataset_cohort_source).toContain('dataset metadata only')
    expect(result.dataset_cohort_source).toContain(
      'not an ARMT-GAN prediction or WHO grade',
    )
  })

  it('parses error detail from response', async () => {
    mockGetIdToken.mockResolvedValue('test-id-token')

    const mockResponse = {
      ok: false,
      status: 500,
      json: vi.fn().mockResolvedValue({ detail: 'Internal server error' }),
    }
    ;(global.fetch as Mock).mockResolvedValue(mockResponse)

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
      ;(global.fetch as Mock).mockResolvedValue(mockResponse)

      const api = createApiClient(mockGetIdToken, mockOnUnauthorized)
      await api.uploadKaggleScan(new File([''], 'mri.jpg', { type: 'image/jpeg' }))

      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/v1/scans/kaggle/upload',
        expect.objectContaining({
          method: 'POST',
          headers: expect.any(Headers),
        })
      )

      const fetchCall = (global.fetch as Mock).mock.calls[0]
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
      ;(global.fetch as Mock).mockResolvedValue(mockResponse)

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
      ;(global.fetch as Mock).mockResolvedValue(mockResponse)

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
      ;(global.fetch as Mock).mockResolvedValue(mockResponse)

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