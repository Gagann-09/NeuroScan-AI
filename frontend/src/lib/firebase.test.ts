import { describe, it, expect, vi, beforeEach } from 'vitest'
import { setPersistenceMock } from '../../vitest.setup'

describe('Firebase Session Persistence', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.resetModules()
  })

  it('configures browserSessionPersistence on auth initialization', async () => {
    // Import the module to trigger initialization
    await import('@/lib/firebase')

    expect(setPersistenceMock).toHaveBeenCalledWith(
      expect.anything(), // auth instance
      'SESSION' // browserSessionPersistence
    )
  })
})