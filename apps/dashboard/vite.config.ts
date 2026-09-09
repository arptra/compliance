import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { Agent } from 'node:https'
import { rootCertificates } from 'node:tls'

export default defineConfig(({ command }) => {
  const enabled = process.env.HTTPS_ENABLED || '0'
  if (enabled !== '0' && enabled !== '1') throw new Error('HTTPS_ENABLED must be 1 or 0')
  let https
  // Building static assets does not require access to the VM's private key.
  if (command === 'serve' && enabled === '1') {
    if (!process.env.TLS_CERT_FILE || !process.env.TLS_KEY_FILE) {
      throw new Error('HTTPS requires TLS_CERT_FILE (fullchain) and TLS_KEY_FILE')
    }
    https = {
      cert: readFileSync(process.env.TLS_CERT_FILE),
      key: readFileSync(process.env.TLS_KEY_FILE),
      passphrase: process.env.TLS_KEY_PASSWORD || undefined,
    }
  }
  const apiTarget = process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
  const proxyAgent = command === 'serve' && apiTarget.startsWith('https://') && process.env.TLS_CA_FILE
    ? new Agent({ ca: [...rootCertificates, readFileSync(process.env.TLS_CA_FILE, 'utf8')] })
    : undefined

  return {
    plugins: [react()],
    server: {
      https,
      host: process.env.VITE_HOST || '0.0.0.0',
      port: Number(process.env.VITE_PORT || 5173),
      strictPort: true,
      // Allows VM/public host access for dev UI when opening by external IP/domain.
      allowedHosts: true,
      origin: process.env.VITE_PUBLIC_ORIGIN || undefined,
      hmr: {
        host: process.env.VITE_HMR_HOST || undefined,
        clientPort: process.env.VITE_HMR_CLIENT_PORT ? Number(process.env.VITE_HMR_CLIENT_PORT) : undefined,
        protocol: (process.env.VITE_HMR_PROTOCOL || (https ? 'wss' : 'ws')) as 'ws' | 'wss',
      },
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          agent: proxyAgent,
        },
      },
    },
    preview: {
      https,
      host: process.env.VITE_HOST || '0.0.0.0',
      port: Number(process.env.VITE_PORT || 5173),
      strictPort: true,
      allowedHosts: true,
    },
  }
})
