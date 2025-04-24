"use client";

import { useEffect } from "react";

export default function Home() {
  useEffect(() => {
    // Redirect to Flask backend login page on component mount
    window.location.href = 'http://localhost:5000/';
  }, []);

  // Return minimal content - this will only briefly appear before redirect
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100">
      <p className="text-gray-500">Redirecting to BKChat...</p>
    </div>
  );
}