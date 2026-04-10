/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        // We'll rely on the default tailwind slate, cyan, emerald, and rose
        // The prompt specifically mentions text-cyan-600, dark:text-cyan-400
      }
    },
  },
  plugins: [],
}
