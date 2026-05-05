const BASE = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8000`

export function apiUrl(path: string) {
  return `${BASE}${path}`
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path))
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(apiUrl(path), { method: 'POST', body: form })
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

export async function apiPostBlob(path: string, body: unknown): Promise<{ blob: Blob; filename: string | null }> {
  const res = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error(await res.text())
  return {
    blob: await res.blob(),
    filename: parseFilenameFromDisposition(res.headers.get('Content-Disposition')),
  }
}
