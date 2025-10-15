# StudAI Web App

Next.js frontend for the StudAI audio transcription application.

## Getting Started

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Run the development server:
```bash
npm run dev
# or
yarn dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- **Home Page**: Landing page at `/`
- **Record Page**: Audio transcription interface at `/record`
  - Audio file upload input
  - Transcribe button for processing

## Project Structure

```
app/
├── layout.tsx       # Root layout component
├── page.tsx         # Home page
└── record/
    └── page.tsx     # Audio transcription page
```

## Tech Stack

- Next.js 14
- React 18
- TypeScript 5
