import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI backend (uvicorn, port 7860) serves the built app; `npm run dev` proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:7860' } },
  build: {
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    modulePreload: { polyfill: false },
  },
})
