/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Proxy Next.js /api/* to FastAPI dev server
    // In production, serve FastAPI separately and set NEXT_PUBLIC_API_BASE_URL
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_BASE_URL
          ? `${process.env.NEXT_PUBLIC_API_BASE_URL}/:path*`
          : 'http://localhost:8000/:path*',
      },
    ]
  },
}

module.exports = nextConfig
