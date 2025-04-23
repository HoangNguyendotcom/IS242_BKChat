import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'BkChat',
  description: 'Created by JamesNg',
  generator: 'JamesNg',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
