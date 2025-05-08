/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'apple-gray': {
          50: '#f5f5f7',
          100: '#e6e6e6',
          200: '#d2d2d7',
          300: '#86868b',
          400: '#6e6e73',
          500: '#1d1d1f',
        },
        'apple-blue': {
          500: '#0071e3',
          600: '#0077ed',
          700: '#006edb',
        }
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Text',
          'Segoe UI',
          'Roboto',
          'sans-serif'
        ]
      },
      boxShadow: {
        'apple': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'apple-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
      borderRadius: {
        'apple': '12px',
      }
    },
  },
  plugins: [],
} 