"use client";

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import { Eye } from "lucide-react"
import Image from "next/image"
import { useEffect, useState } from "react" // Add this import


export default function Home() {
  // Add this state for client-side rendering
  const [isClient, setIsClient] = useState(false)
  
  // Add this effect to set isClient to true after mounting
  useEffect(() => {
    setIsClient(true)
  }, [])

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col">
      {/* Main content */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg w-full max-w-4xl overflow-hidden relative">
          {/* Background waves */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -left-10 top-40 w-full h-96 bg-blue-200/30 rounded-full blur-xl transform rotate-12"></div>
            <div className="absolute -right-10 bottom-20 w-full h-96 bg-blue-300/30 rounded-full blur-xl transform -rotate-12"></div>
          </div>

          <div className="relative z-10 flex flex-col md:flex-row">
            {/* Left side with logo */}
            <div className="p-8 flex items-center justify-center md:w-1/4">
              <div className="w-72 h-72 relative">
                {/* Conditional rendering based on client state */}
                {isClient ? (
                  <Image
                    src="/images/bk-logo.png"
                    alt="BK Logo"
                    width={300}
                    height={300}
                    className="rounded-lg"
                  />
                ) : (
                  <div className="w-72 h-72 bg-gray-100 rounded-lg"></div> // Placeholder during SSR
                )}
              </div>
            </div>

            {/* Center login form */}
            <div className="p-8 md:w-2/4 border border-blue-100 rounded-lg mx-4 my-6 bg-white/80 backdrop-blur-sm">
              <div className="flex items-center justify-center mb-6">
                <div className="flex items-center">
                  <svg viewBox="0 0 24 24" className="w-8 h-8 mr-2">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" fill="black" />
                  </svg>
                  <h1 className="text-3xl font-bold text-red-500">BKchat</h1>
                </div>
              </div>

              <h2 className="text-xl font-bold mb-1">Welcome Back!</h2>
              <p className="text-slate-700 mb-6">Please log in to continue</p>

              <form className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="username" className="block text-sm font-medium">
                    Username
                  </label>
                  <Input id="username" placeholder="admin" />
                </div>

                <div className="space-y-2">
                  <label htmlFor="password" className="block text-sm font-medium">
                    Password
                  </label>
                  <div className="relative">
                    <Input id="password" type="password" placeholder="••••••••" />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400"
                      aria-label="Show password"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                <Button className="w-full bg-blue-500 hover:bg-blue-600" asChild>
                  <Link href="/main">Log In</Link>
                </Button>

                <div className="relative my-6">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-slate-300"></div>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-sm text-slate-600 mb-2">Don't have an account?</p>
                  <Button
                    variant="outline"
                    className="w-full bg-blue-100 border-blue-200 text-blue-700 hover:bg-blue-200"
                    asChild
                  >
                    <Link href="/signup">Sign Up</Link>
                  </Button>
                </div>
              </form>
            </div>

            {/* Right side with illustration */}
            <div className="hidden md:flex items-center justify-center p-8 md:w-1/4">
              <div className="w-full h-48 relative">
                <Image
                  src="/placeholder.svg?height=192&width=192"
                  alt="People collaborating"
                  width={192}
                  height={192}
                  className="object-contain"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}