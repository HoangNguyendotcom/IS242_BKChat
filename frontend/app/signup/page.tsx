"use client";

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import { Eye, EyeOff } from "lucide-react"
import Image from "next/image"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

export default function SignupPage() {
  // Client-side rendering state
  const [isClient, setIsClient] = useState(false);

  // Form data state
  const [formData, setFormData] = useState({
    name: '',
    username: '',
    email: '',
    password: ''
  });

  // Form validation and UI states
  const [errors, setErrors] = useState<any>({});
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const router = useRouter();

  // Set isClient to true after mounting
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Handle input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target;
    setFormData((prev: any) => ({
      ...prev,
      [id]: value
    }));

    // Clear errors when typing
    if (errors[id]) {
      setErrors((prev: any) => ({
        ...prev,
        [id]: ''
      }));
    }
  };

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(prev => !prev);
  };

  // Validate form data
  const validateForm = () => {
    const newErrors: any = {};

    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    } else if (formData.name.length < 2) {
      newErrors.name = 'Name must be at least 2 characters';
    }

    // Username validation
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
    }

    // Email validation
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(formData.email)) {
        newErrors.email = 'Please enter a valid email address';
      }
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate form
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      // Send data to backend
      const response = await fetch('http://localhost:5000/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {
        // Successful signup
        router.push('/login'); // Redirect to login page after successful signup
      } else {
        // Failed signup
        setErrors((prev: any) => ({
          ...prev,
          submit: data.message || 'Signup failed. Please try again.'
        }));
      }
    } catch (error: any) {
      console.error('Error during signup:', error);
      setErrors((prev: any) => ({
        ...prev,
        submit: 'Network error. Please try again.'
      }));
    } finally {
      setIsLoading(false);
    }
  };

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
                {isClient ? (
                  <Image
                    src="/images/bk-logo.png"
                    alt="BK Logo"
                    width={300}
                    height={300}
                    className="rounded-lg"
                  />
                ) : (
                  <div className="w-72 h-72 bg-gray-100 rounded-lg"></div>
                )}
              </div>
            </div>

            {/* Center signup form */}
            <div className="p-8 md:w-2/4 border border-blue-100 rounded-lg mx-4 my-6 bg-white/80 backdrop-blur-sm">
              <div className="flex items-center justify-center mb-6">
                <div className="flex items-center">
                  <div className="w-18 h-18 relative">
                    {isClient ? (
                      <Image
                        src="/images/logo.png"
                        alt="BK Logo"
                        width={50}
                        height={50}
                        className="rounded-lg"
                      />
                    ) : (
                      <div className="w-18 h-18 bg-gray-100 rounded-lg"></div>
                    )}
                  </div>
                  <h1 className="text-5xl font-bold text-red-500">BKchat</h1>
                </div>
              </div>
              <h2 className="text-xl font-bold mb-1">Welcome Homie!</h2>
              <p className="text-slate-700 mb-6">Please sign up to join with us</p>

              <form className="space-y-4" onSubmit={handleSubmit}>
                {errors.submit && (
                  <div className="p-3 rounded bg-red-50 border border-red-200">
                    <p className="text-red-600 text-sm">{errors.submit}</p>
                  </div>
                )}
                
                <div className="space-y-2">
                  <label htmlFor="name" className="block text-sm font-medium">
                    Name
                  </label>
                  <Input 
                    id="name" 
                    placeholder="name" 
                    value={formData.name}
                    onChange={handleChange}
                    className={errors.name ? "border-red-500" : ""}
                  />
                  {errors.name && (
                    <p className="text-red-500 text-xs mt-1">{errors.name}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <label htmlFor="username" className="block text-sm font-medium">
                    Username
                  </label>
                  <Input 
                    id="username" 
                    placeholder="username" 
                    value={formData.username}
                    onChange={handleChange}
                    className={errors.username ? "border-red-500" : ""}
                  />
                  {errors.username && (
                    <p className="text-red-500 text-xs mt-1">{errors.username}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <label htmlFor="email" className="block text-sm font-medium">
                    Email
                  </label>
                  <Input 
                    id="email" 
                    type="email" 
                    placeholder="admin@hcmut.edu.vn" 
                    value={formData.email}
                    onChange={handleChange}
                    className={errors.email ? "border-red-500" : ""}
                  />
                  {errors.email && (
                    <p className="text-red-500 text-xs mt-1">{errors.email}</p>
                  )}
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
                      value={formData.password}
                      onChange={handleChange}
                      className={errors.password ? "border-red-500" : ""}
                    />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                      onClick={togglePasswordVisibility}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="text-red-500 text-xs mt-1">{errors.password}</p>
                  )}
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-blue-500 hover:bg-blue-600" 
                  disabled={isLoading}
                >
                  {isLoading ? "Signing Up..." : "Sign Up"}
                </Button>

                <div className="relative my-6">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-slate-300"></div>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-sm text-slate-600 mb-2">Already have an account?</p>
                  <Button
                    variant="outline"
                    className="w-full bg-blue-100 border-blue-200 text-blue-700 hover:bg-blue-200"
                    asChild
                  >
                    <Link href="/login">Log In</Link>
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
