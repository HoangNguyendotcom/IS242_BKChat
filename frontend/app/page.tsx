"use client";

import { Button } from "@/components/ui/button"
import Link from "next/link"
import Image from "next/image"
import { useEffect, useState } from "react"

export default function Home() {
  // Client-side rendering state
  const [isClient, setIsClient] = useState(false)
  const [apiStatus, setApiStatus] = useState({ connected: false, message: "" })
  
  // Set isClient to true after mounting
  useEffect(() => {
    setIsClient(true)
    
    // Check connection to Flask backend
    fetch('http://localhost:5000/')
      .then(response => {
        if (response.ok) {
          return response.json()
          .catch(() => {
            return { message: "BKchat API is running" };
          });
        } else {
          throw new Error('API connection failed');
        }
      })
      .then(data => {
        setApiStatus({ 
          connected: true, 
          message: data.message
        });
        console.log("Connected to Flask API:", data);
      })
      .catch(error => {
        console.error("Failed to connect to Flask API:", error);
        setApiStatus({
          connected: false,
          message: "API connection failed"
        });
      });
  }, []);

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

          <div className="relative z-10 flex flex-col items-center justify-center p-12 text-center">
            {/* Logo */}
            <div className="mb-8">
              <div className="flex items-center justify-center">
                <div className="w-24 h-24 relative mr-4">
                  {isClient ? (
                    <Image
                      src="/images/logo.png"
                      alt="BK Logo"
                      width={100}
                      height={100}
                      className="rounded-lg"
                    />
                  ) : (
                    <div className="w-24 h-24 bg-gray-100 rounded-lg"></div>
                  )}
                </div>
                <h1 className="text-6xl font-bold text-red-500">BKchat</h1>
              </div>
            </div>

            {/* Welcome Heading */}
            <h2 className="text-4xl font-bold mb-6 text-blue-700">Welcome to BKChat!!!</h2>
            
            {/* API Status indicator */}
            <div className="mb-4">
              <span className={`inline-block w-3 h-3 rounded-full mr-2 ${apiStatus.connected ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm text-gray-600">
                {apiStatus.connected ? 'Backend connected' : 'Backend disconnected'}
              </span>
            </div>
            
            {/* Larger BK image */}
            <div className="w-56 h-56 relative mb-10">
              {isClient ? (
                <Image
                  src="/images/bk-logo.png"
                  alt="BK Logo"
                  width={300}
                  height={300}
                  className="rounded-lg"
                />
              ) : (
                <div className="w-56 h-56 bg-gray-100 rounded-lg"></div>
              )}
            </div>

            <Button 
              className="w-64 h-14 text-2xl bg-blue-500 hover:bg-blue-600" 
              asChild
            >
              <Link href="/login">Start Chatting...</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
