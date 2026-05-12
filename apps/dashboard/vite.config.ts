import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.VITE_HOST || '0.0.0.0',
    port: Number(process.env.VITE_PORT || 5173),
    strictPort: true,
    // Allows VM/public host access for dev UI when opening by external IP/domain.
    allowedHosts: true,
    origin: process.env.VITE_PUBLIC_ORIGIN || undefined,
    hmr: {
      host: process.env.VITE_HMR_HOST || undefined,
      clientPort: process.env.VITE_HMR_CLIENT_PORT ? Number(process.env.VITE_HMR_CLIENT_PORT) : undefined,
      protocol: process.env.VITE_HMR_PROTOCOL as 'ws' | 'wss' | undefined,
    },
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: process.env.VITE_HOST || '0.0.0.0',
    port: Number(process.env.VITE_PORT || 5173),
    strictPort: true,
    allowedHosts: true,
  },
})
