import '@testing-library/jest-dom'
import { vi } from 'vitest'
import React from 'react'

// Mock Firebase globally before any tests run
const setPersistenceMock = vi.fn().mockResolvedValue(undefined)
const getAuthMock = vi.fn().mockReturnValue({})
const onAuthStateChangedMock = vi.fn()
const signOutMock = vi.fn()

vi.mock('firebase/app', () => ({
  initializeApp: vi.fn(),
}))

vi.mock('firebase/auth', () => ({
  getAuth: getAuthMock,
  setPersistence: setPersistenceMock,
  browserSessionPersistence: 'SESSION',
  onAuthStateChanged: onAuthStateChangedMock,
  signOut: signOutMock,
}))

// Mock next/navigation
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => '/',
}))

// Mock lucide-react icons
vi.mock('lucide-react', () => {
  const icons: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {}
  const iconNames = [
    'Loader2', 'BrainCircuit', 'Lock', 'LogOut', 'AlertCircle',
    'ArrowUp', 'Plus', 'Activity', 'ImageIcon', 'Box', 'X', 'CheckCircle',
    'Target', 'Settings', 'Network', 'Database', 'Share2', 'Home',
    'ArrowUpRight', 'ShieldAlert', 'ArrowLeft', 'Check', 'Download',
    'FileText', 'Activity'
  ]
  iconNames.forEach(name => {
    icons[name] = ({ ...props }: React.SVGProps<SVGSVGElement>) => (
      React.createElement('svg', { 'data-testid': name, ...props })
    )
  })
  return icons
})

// Export mocks for tests to use
export { setPersistenceMock, getAuthMock, onAuthStateChangedMock, signOutMock }