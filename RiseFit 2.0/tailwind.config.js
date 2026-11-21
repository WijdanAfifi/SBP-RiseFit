/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  // Web requires an explicit class strategy; the default `media` flag makes
  // react-native-css-interop throw when the color scheme is synced.
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: '#F6F7F2',
        surface: '#FFFFFF',
        ink: '#0F231D',
        inkSoft: '#364B43',
        muted: '#75857D',
        line: '#E6E8E0',
        forest: {
          DEFAULT: '#0F5B4F',
          dark: '#0A4339',
          deep: '#083630',
          light: '#14705F',
        },
        sage: {
          DEFAULT: '#A8C6B5',
          soft: '#E8F0E9',
          tint: '#F1F6F1',
          deep: '#7FA793',
        },
        orange: {
          DEFAULT: '#F4784B',
          deep: '#E4643A',
          soft: '#FDEDE5',
        },
        sky: {
          DEFAULT: '#3E71C7',
          soft: '#EDF2FB',
        },
      },
      borderRadius: {
        card: '22px',
        field: '14px',
        sheet: '28px',
      },
      fontSize: {
        display: ['34px', { lineHeight: '40px', letterSpacing: '-1px' }],
        h1: ['28px', { lineHeight: '34px', letterSpacing: '-0.7px' }],
        h2: ['22px', { lineHeight: '28px', letterSpacing: '-0.4px' }],
        h3: ['18px', { lineHeight: '24px', letterSpacing: '-0.2px' }],
        body: ['15px', { lineHeight: '23px' }],
        small: ['13px', { lineHeight: '19px' }],
        tiny: ['11px', { lineHeight: '15px' }],
        metric: ['34px', { lineHeight: '38px', letterSpacing: '-1.2px' }],
        metricSm: ['26px', { lineHeight: '30px', letterSpacing: '-0.8px' }],
      },
    },
  },
  plugins: [],
};
