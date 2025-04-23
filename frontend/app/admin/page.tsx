"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import Image from "next/image"
import { Home, User, LayoutDashboard, Settings, Bell, LogOut, MoreHorizontal } from "lucide-react"

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<string | null>(null)
  const [activeOptionMenu, setActiveOptionMenu] = useState<number | null>(null)
  const router = useRouter()

  const menuItems = [
    { id: "USER", icon: User, label: "USER" },
    { id: "DASHBOARD", icon: LayoutDashboard, label: "DASHBOARD" },
    { id: "GENERAL", icon: Settings, label: "GENERAL" },
    { id: "NOTIFICATIONS", icon: Bell, label: "NOTIFICATIONS" },
  ]

  const handleLogout = () => {
    // In a real app, you would handle logout logic here (clear tokens, etc.)
    router.push("/")
  }

  // Friend list data for USER tab
  const friends = [
    { name: "Dac Hoang", totalMessages: 813, toxicMessages: 72, toxicRate: "8.8%" },
    { name: "Tuan Nam", totalMessages: 645, toxicMessages: 210, toxicRate: "32.6%" },
    { name: "Hoang Long", totalMessages: 404, toxicMessages: 60, toxicRate: "14.8%" },
    { name: "Le Phu", totalMessages: 100, toxicMessages: 40, toxicRate: "4%" },
    { name: "Tri Cuong", totalMessages: 38, toxicMessages: 14, toxicRate: "36.8%" },
  ]

  // ML model data for DASHBOARD tab
  const mlData = {
    totalMessages: 2076,
    toxicMessages: 396,
    userFeedback: {
      toxic: 50,
      notToxic: 10,
    },
    modelPerformance: {
      f1Score: "92.5%",
      recall: "88.5%",
      precision: "97.5%",
      userFeedbackRate: "2.9%",
    },
    feedbackAnalysis: {
      trueNegative: "78.5%",
      truePositive: "18.6%",
      falseNegative: "2.4%",
      falsePositive: "0.5%",
    },
  }

  const toggleOptionMenu = (index: number) => {
    if (activeOptionMenu === index) {
      setActiveOptionMenu(null)
    } else {
      setActiveOptionMenu(index)
    }
  }

  // Close option menu when clicking outside
  const handleClickOutside = () => {
    setActiveOptionMenu(null)
  }

  // Add this state for client-side rendering
    const [isClient, setIsClient] = useState(false)
    
    // Add this effect to set isClient to true after mounting
    useEffect(() => {
      setIsClient(true)
    }, [])

  return (
    <div className="flex h-screen bg-white" onClick={handleClickOutside}>
      {/* Left sidebar */}
      <div className="w-60 border-r flex flex-col bg-gray-50">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-12 h-12 relative">
                {/* Conditional rendering based on client state */}
                  {isClient ? (
                    <Image
                      src="/images/logo.png"
                      alt="BK Logo"
                      width={40}
                      height={40}
                      className="rounded-lg"
                    />
                    ) : (
              <div className="w-12 h-12 bg-gray-100 rounded-lg"></div> // Placeholder during SSR
                  )}
            </div>
            <span className="text-3xl font-bold text-red-500">BKChat</span>
          </div>
          <Link href="/main" className="text-gray-500">
            <Home className="h-5 w-5" />
          </Link>
        </div>

        {/* Administrator title */}
        <div className="p-4 font-bold text-xl">Administrator</div>

        {/* Menu items */}
        <div className="flex-1">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`w-full text-left px-4 py-3 flex items-center gap-3 ${
                activeTab === item.id ? "bg-gray-200" : "hover:bg-gray-100"
              }`}
              onClick={() => setActiveTab(item.id)}
            >
              <item.icon className="h-5 w-5 text-gray-700" />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </div>

        {/* Logout button */}
        <div className="p-4 mt-auto">
          <button
            className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-gray-100 rounded-md"
            onClick={handleLogout}
          >
            <LogOut className="h-5 w-5 text-gray-700" />
            <span className="font-medium">LOG OUT</span>
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1">
        {/* Top navigation */}
        <div className="p-4 border-b flex items-center">
          <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
            <Image src="/images/profile.png" alt="Profile" width={40} height={40} className="object-cover" />
          </div>
        </div>

        {/* Content based on active tab */}
        <div className="p-6">
          {activeTab === null && (
            <div className="flex flex-col items-center justify-center h-[calc(100vh-6rem)]">
              <div className="bg-white rounded-lg shadow-md p-8 max-w-md w-full">
                <div className="flex items-center space-x-4 mb-6">
                  <div className="w-20 h-20 rounded-full bg-gray-200 overflow-hidden">
                    <Image src="/images/profile.png" alt="Minh Trinh" width={80} height={80} className="object-cover" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Minh Trinh</h2>
                    <p className="text-gray-500">@tcminh.sdh241</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <p className="text-gray-600">
                    Welcome to the BKChat administration panel. Select an option from the sidebar to manage your
                    application.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === "USER" && (
            <div>
              <h2 className="text-xl font-bold mb-6">FRIEND LIST:</h2>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-4 py-2 text-left">Name</th>
                      <th className="border border-gray-300 px-4 py-2 text-center">Total Messages</th>
                      <th className="border border-gray-300 px-4 py-2 text-center">Toxic Messages</th>
                      <th className="border border-gray-300 px-4 py-2 text-center">Toxic Rate</th>
                      <th className="border border-gray-300 px-4 py-2 text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {friends.map((friend, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="border border-gray-300 px-4 py-2">{friend.name}</td>
                        <td className="border border-gray-300 px-4 py-2 text-center">{friend.totalMessages}</td>
                        <td className="border border-gray-300 px-4 py-2 text-center text-red-500">
                          {friend.toxicMessages}
                        </td>
                        <td className="border border-gray-300 px-4 py-2 text-center text-red-500">
                          {friend.toxicRate}
                        </td>
                        <td className="border border-gray-300 px-4 py-2 text-center relative">
                          <button
                            className="text-blue-500"
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleOptionMenu(index)
                            }}
                          >
                            <MoreHorizontal className="h-5 w-5 inline" />
                          </button>

                          {activeOptionMenu === index && (
                            <div className="absolute right-10 top-2 bg-white shadow-lg rounded-md border border-gray-200 z-10 w-40 py-1">
                              <div className="px-3 py-2 text-center font-medium border-b border-gray-100">OPTIONS</div>
                              <button className="w-full text-left px-3 py-2 hover:bg-gray-100 flex items-center gap-2">
                                <span>Report</span>
                              </button>
                              <button className="w-full text-left px-3 py-2 hover:bg-gray-100 flex items-center gap-2 text-red-500">
                                <span>Block</span>
                              </button>
                              <button className="w-full text-left px-3 py-2 hover:bg-gray-100 flex items-center gap-2">
                                <span>Unfriend</span>
                              </button>
                              <button className="w-full text-left px-3 py-2 hover:bg-gray-100 flex items-center gap-2">
                                <span>Uncheck</span>
                              </button>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex justify-center mt-4">
                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-gray-300 rounded">1</button>
                  <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">2</button>
                  <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">3</button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "DASHBOARD" && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold">ML_model analysis:</h2>
                <button className="px-4 py-1 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50">
                  Report
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-sm text-gray-500 mb-2">Total messages:</p>
                  <p className="text-3xl font-bold text-center">{mlData.totalMessages}</p>
                </div>
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-sm text-gray-500 mb-2">Toxic messages:</p>
                  <p className="text-3xl font-bold text-center">{mlData.toxicMessages}</p>
                </div>
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-sm text-gray-500 mb-2">User feedback:</p>
                  <div className="flex justify-center space-x-8">
                    <div className="text-center">
                      <p className="text-red-500 font-bold">Toxic</p>
                      <p className="text-xl font-bold">{mlData.userFeedback.toxic}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-green-500 font-bold">Not Toxic</p>
                      <p className="text-xl font-bold">{mlData.userFeedback.notToxic}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <h3 className="font-medium mb-4">Model Performance:</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>F1 score:</span>
                        <span>{mlData.modelPerformance.f1Score}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: mlData.modelPerformance.f1Score }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Recall:</span>
                        <span>{mlData.modelPerformance.recall}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: mlData.modelPerformance.recall }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Precision:</span>
                        <span>{mlData.modelPerformance.precision}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: mlData.modelPerformance.precision }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>User feedback rate:</span>
                        <span>{mlData.modelPerformance.userFeedbackRate}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-blue-500 h-2.5 rounded-full" style={{ width: "2.9%" }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-4 rounded-md shadow-sm">
                  <h3 className="font-medium mb-4">Users Feedback Analysis</h3>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-xl font-semibold text-center mb-4">Users Feedback Analysis</h3>
                    <div className="flex flex-wrap justify-center gap-4 mb-4">
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.trueNegative} TN</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-yellow-300 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.truePositive} TP</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.falseNegative} FN</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.falsePositive} FP</span>
                      </div>
                    </div>
                    <div className="relative w-64 h-64 mx-auto">
                      {/* SVG Pie Chart - Starting from top (90 degrees) */}
                      <svg viewBox="0 0 100 100" className="w-full h-full">
                        {/* Segments calculated with precise arc commands */}

                        {/* TN - Blue (78.5%) - 282.6 degrees */}
                        <path d="M 50 50 L 50 10 A 40 40 0 0 1 50 90 A 40 40 0 0 1 13.4 34.4 Z" fill="#3b82f6" />

                        {/* TP - Yellow (18.6%) - 66.96 degrees */}
                        <path d="M 50 50 L 13.4 34.4 A 40 40 0 0 1 26.4 13.6 Z" fill="#fde047" />

                        {/* FN - Green (2.4%) - 8.64 degrees */}
                        <path d="M 50 50 L 26.4 13.6 A 40 40 0 0 1 34.4 10.8 Z" fill="#22c55e" />

                        {/* FP - Red (0.5%) - 1.8 degrees */}
                        <path d="M 50 50 L 34.4 10.8 A 40 40 0 0 1 50 10 Z" fill="#ef4444" />

                        {/* White center circle */}
                        <circle cx="50" cy="50" r="25" fill="white" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "GENERAL" && (
            <div>
              <h2 className="text-xl font-semibold mb-4">General Settings</h2>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Language</label>
                    <select className="w-full p-2 border rounded-md">
                      <option>English</option>
                      <option>Vietnamese</option>
                      <option>French</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Theme</label>
                    <select className="w-full p-2 border rounded-md">
                      <option>Light</option>
                      <option>Dark</option>
                      <option>System</option>
                    </select>
                  </div>
                  <div className="flex items-center">
                    <input type="checkbox" id="sounds" className="mr-2" defaultChecked />
                    <label htmlFor="sounds" className="text-sm font-medium text-gray-700">
                      Enable notification sounds
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "NOTIFICATIONS" && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Notification Settings</h2>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">New Messages</h3>
                      <p className="text-sm text-gray-500">Get notified when you receive new messages</p>
                    </div>
                    <div className="relative inline-block w-12 h-6">
                      <input type="checkbox" id="new-messages" className="sr-only" defaultChecked />
                      <span className="block h-6 w-12 rounded-full bg-blue-600"></span>
                      <span className="absolute left-1 top-1 h-4 w-4 rounded-full bg-white transition-transform transform translate-x-6"></span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">Friend Requests</h3>
                      <p className="text-sm text-gray-500">Get notified when someone sends you a friend request</p>
                    </div>
                    <div className="relative inline-block w-12 h-6">
                      <input type="checkbox" id="friend-requests" className="sr-only" defaultChecked />
                      <span className="block h-6 w-12 rounded-full bg-blue-600"></span>
                      <span className="absolute left-1 top-1 h-4 w-4 rounded-full bg-white transition-transform transform translate-x-6"></span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">System Updates</h3>
                      <p className="text-sm text-gray-500">Get notified about system updates and maintenance</p>
                    </div>
                    <div className="relative inline-block w-12 h-6">
                      <input type="checkbox" id="system-updates" className="sr-only" />
                      <span className="block h-6 w-12 rounded-full bg-gray-300"></span>
                      <span className="absolute left-1 top-1 h-4 w-4 rounded-full bg-white transition-transform"></span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
