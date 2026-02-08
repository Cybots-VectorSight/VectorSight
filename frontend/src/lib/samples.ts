export interface SampleSvg {
  id: string
  name: string
  svg: string
}

export const sampleSvgs: SampleSvg[] = [
  {
    id: "circle",
    name: "Circle",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" stroke-width="2"/>
</svg>`,
  },
  {
    id: "smiley",
    name: "Smiley",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" stroke-width="2"/>
  <circle cx="35" cy="38" r="5" fill="currentColor"/>
  <circle cx="65" cy="38" r="5" fill="currentColor"/>
  <path d="M 30 62 Q 50 80 70 62" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>`,
  },
  {
    id: "house",
    name: "House",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <polygon points="50,10 90,50 10,50" fill="none" stroke="currentColor" stroke-width="2"/>
  <rect x="20" y="50" width="60" height="40" fill="none" stroke="currentColor" stroke-width="2"/>
  <rect x="40" y="65" width="20" height="25" fill="none" stroke="currentColor" stroke-width="2"/>
</svg>`,
  },
  {
    id: "gear",
    name: "Settings Gear",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <circle cx="50" cy="50" r="15" fill="none" stroke="currentColor" stroke-width="2"/>
  <circle cx="50" cy="50" r="30" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="8 6"/>
  <line x1="50" y1="15" x2="50" y2="5" stroke="currentColor" stroke-width="3"/>
  <line x1="50" y1="85" x2="50" y2="95" stroke="currentColor" stroke-width="3"/>
  <line x1="15" y1="50" x2="5" y2="50" stroke="currentColor" stroke-width="3"/>
  <line x1="85" y1="50" x2="95" y2="50" stroke="currentColor" stroke-width="3"/>
</svg>`,
  },
  {
    id: "bar-chart",
    name: "Bar Chart",
    svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <line x1="10" y1="90" x2="90" y2="90" stroke="currentColor" stroke-width="2"/>
  <line x1="10" y1="10" x2="10" y2="90" stroke="currentColor" stroke-width="2"/>
  <rect x="20" y="50" width="12" height="40" fill="currentColor" opacity="0.7"/>
  <rect x="38" y="30" width="12" height="60" fill="currentColor" opacity="0.7"/>
  <rect x="56" y="45" width="12" height="45" fill="currentColor" opacity="0.7"/>
  <rect x="74" y="20" width="12" height="70" fill="currentColor" opacity="0.7"/>
</svg>`,
  },
]
