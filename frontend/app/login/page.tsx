"use client";

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import { Eye, EyeOff } from "lucide-react"
import Image from "next/image"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

export default function Home() {
  // Client-side rendering state
  const [isClient, setIsClient] = useState(false)
  
  // Form state
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  const router = useRouter();

  // Set isClient to true after mounting
  useEffect(() => {
    setIsClient(true)
  }, [])

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch('http://localhost:5000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        // Login successful
        console.log("Login successful:", data);
        // Save user data or token if needed
        localStorage.setItem('token', data.access_token);

        // Redirect to main page
        router.push('/main');
      } else {
        // Login failed
        setError(data.message || "Login failed. Please check your credentials.");
      }
    } catch (error: any) {
      console.error("Login error:", error);
      setError("Connection error:", error);
    } finally {
      setLoading(false);
    }
  };

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword)
  }

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
            <div className="p-8 flex items-center justify-center md:w-2/5">
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
                  <div className="w-18 h-18 relative">
                {/* Conditional rendering based on client state */}
                {isClient ? (
                  <Image
                    src="/images/logo.png"
                    alt="BK Logo"
                    width={50}
                    height={50}
                    className="rounded-lg"
                  />
                ) : (
                  <div className="w-18 h-18 bg-gray-100 rounded-lg"></div> // Placeholder during SSR
                )}
                  </div>
                  <h1 className="text-5xl font-bold text-red-500">BKchat</h1>
                </div>
              </div>

              <h2 className="text-xl font-bold mb-1">Welcome Homie!</h2>
              <p className="text-slate-700 mb-6">Please log in to continue</p>

              {error && (
                <div className="p-3 mb-4 bg-red-100 text-red-700 rounded-md text-sm">
                  {error}
                </div>
              )}

              <form className="space-y-4" onSubmit={handleSubmit}>
                <div className="space-y-2">
                  <label htmlFor="username" className="block text-sm font-medium">
                    Username
                  </label>
                  <Input 
                    id="username" 
                    placeholder="admin" 
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="password" className="block text-sm font-medium">
                    Password
                  </label>
                  <div className="relative">
                    <Input 
                      id="password" 
                      type={showPassword ? "text" : "password"} 
                      placeholder="••••••••" 
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400"
                      onClick={togglePasswordVisibility}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <Button 
                  className="w-full bg-blue-500 hover:bg-blue-600" 
                  type="submit"
                  disabled={loading}
                >
                  {loading ? "Logging in..." : "Log In"}
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
          </div>
        </div>
      </div>
    </div>
  )
}
