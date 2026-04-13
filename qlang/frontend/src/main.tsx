import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

function readAuthToken(): string | null {
  const fromStorage = localStorage.getItem('qo_auth_token')
  if (fromStorage && fromStorage.trim().length > 0) {
    return fromStorage.trim()
  }

  const fromUrl = new URLSearchParams(window.location.search).get('token')
  if (fromUrl && fromUrl.trim().length > 0) {
    localStorage.setItem('qo_auth_token', fromUrl.trim())
    return fromUrl.trim()
  }

  return null
}

function withTokenQuery(url: string, token: string): string {
  const absolute = new URL(url, window.location.origin)
  if (!absolute.searchParams.has('token')) {
    absolute.searchParams.set('token', token)
  }

  if (url.startsWith('http://') || url.startsWith('https://')) {
    return absolute.toString()
  }

  return `${absolute.pathname}${absolute.search}${absolute.hash}`
}

function bootstrapAuthTransport() {
  const originalFetch = window.fetch.bind(window)

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const token = readAuthToken()
    if (!token) {
      return originalFetch(input, init)
    }

    const headers = new Headers(init?.headers ?? {})
    if (!headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`)
    }

    return originalFetch(input, {
      ...init,
      headers,
    })
  }

  const NativeEventSource = window.EventSource
  class TokenEventSource extends NativeEventSource {
    constructor(url: string | URL, eventSourceInitDict?: EventSourceInit) {
      const token = readAuthToken()
      const normalized = typeof url === 'string' ? url : url.toString()
      const nextUrl = token ? withTokenQuery(normalized, token) : normalized
      super(nextUrl, eventSourceInitDict)
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).EventSource = TokenEventSource
}

bootstrapAuthTransport()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
