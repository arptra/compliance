const fallbackProtocol = window.location.protocol || 'http:'
const fallbackApiPort = import.meta.env.VITE_API_PORT || (window.location.port === '15173' ? '18000' : '8000')
const BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '' : `${fallbackProtocol}//${window.location.hostname}:${fallbackApiPort}`)
export const AUTH_TOKEN_KEY = 'complaints-dashboard-auth-token'

export function getAuthToken() {
  return window.localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setAuthToken(token: string | null) {
  if (token) window.localStorage.setItem(AUTH_TOKEN_KEY, token)
  else window.localStorage.removeItem(AUTH_TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export function apiUrl(path: string) {
  return `${BASE}${path}`
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: authHeaders() })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export async function apiGetWithProgress<T>(
  path: string,
  onProgress: (loadedBytes: number, totalBytes: number | null) => void,
): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: authHeaders() })
  if (!res.ok) throw new Error(await res.text())
  const totalBytes = Number(res.headers.get('Content-Length')) || null
  if (!res.body) {
    onProgress(totalBytes ?? 0, totalBytes)
    return res.json() as Promise<T>
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loadedBytes = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    if (value) {
      chunks.push(value)
      loadedBytes += value.byteLength
      onProgress(loadedBytes, totalBytes)
    }
  }

  const merged = new Uint8Array(loadedBytes)
  let offset = 0
  chunks.forEach((chunk) => {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  })
  return JSON.parse(new TextDecoder().decode(merged)) as T
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), { method: 'PATCH', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export async function apiPostForm<T>(path: string, form: FormData, options?: { signal?: AbortSignal }): Promise<T> {
  const res = await fetch(apiUrl(path), { method: 'POST', headers: authHeaders(), body: form, signal: options?.signal })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

function parseFilenameFromDisposition(contentDisposition: string | null) {
  if (!contentDisposition) return null
  const utf8Match = contentDisposition.match(/filename\\*=UTF-8''([^;]+)/i)
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1])
    } catch {
      return utf8Match[1]
    }
  }
  const basicMatch = contentDisposition.match(/filename="?([^"]+)"?/i)
  return basicMatch?.[1] ?? null
}

export async function apiPostBlob(
  path: string,
  body: unknown,
  options?: { onProgress?: (loadedBytes: number, totalBytes: number | null) => void },
): Promise<{ blob: Blob; filename: string | null }> {
  const res = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error(await res.text())
  const filename = parseFilenameFromDisposition(res.headers.get('Content-Disposition'))
  const totalBytes = Number(res.headers.get('Content-Length')) || null
  if (options?.onProgress && res.body) {
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let loadedBytes = 0
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (value) {
        chunks.push(value)
        loadedBytes += value.byteLength
        options.onProgress(loadedBytes, totalBytes)
      }
    }
    const merged = new Uint8Array(loadedBytes)
    let offset = 0
    chunks.forEach((chunk) => {
      merged.set(chunk, offset)
      offset += chunk.byteLength
    })
    return {
      blob: new Blob([merged.buffer]),
      filename,
    }
  }
  return {
    blob: await res.blob(),
    filename,
  }
}
