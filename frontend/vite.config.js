import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API calls to FastAPI so the browser sees a single origin.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } } },
});
