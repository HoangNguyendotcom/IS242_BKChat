"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Home, User, LayoutDashboard, Settings, Bell, LogOut, MoreHorizontal } from "lucide-react"

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState("DASHBOARD")
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

  return (
    <div className="flex h-screen bg-white">
      {/* Left sidebar */}
      <div className="w-60 border-r flex flex-col bg-gray-50">
        {/* Header */}
        <div className="p-4 border-b flex items-center">
          <div className="flex items-center">
            <div className="w-8 h-8 relative mr-2">
              <svg viewBox="0 0 100 100" className="w-full h-full text-blue-600">
                <polygon points="50,10 90,30 90,70 50,90 10,70 10,30" fill="currentColor" />
                <text x="50" y="55" textAnchor="middle" fill="white" fontSize="24" fontWeight="bold">
                  BK
                </text>
              </svg>
            </div>
            <span className="font-bold text-lg">BKChat</span>
          </div>
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
          <Link href="/main" className="text-gray-500">
            <Home className="h-5 w-5" />
          </Link>
        </div>

        {/* Content based on active tab */}
        <div className="p-6">
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
                        <td className="border border-gray-300 px-4 py-2 text-center">
                          <button className="text-blue-500">
                            <MoreHorizontal className="h-5 w-5 inline" />
                          </button>
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
                  <div className="flex justify-center">
                    <div className="relative w-48 h-48">
                      {/* This is a simplified pie chart representation */}
                      <div className="absolute inset-0 rounded-full overflow-hidden">
                        <div
                          className="absolute inset-0 bg-blue-500"
                          style={{ clipPath: "polygon(50% 50%, 0 0, 0 100%, 100% 100%, 100% 0)" }}
                        ></div>
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="bg-white rounded-full w-16 h-16"></div>
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-4">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-blue-500 mr-2"></div>
                      <span className="text-xs">{mlData.feedbackAnalysis.trueNegative} TN</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-yellow-400 mr-2"></div>
                      <span className="text-xs">{mlData.feedbackAnalysis.truePositive} TP</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-red-500 mr-2"></div>
                      <span className="text-xs">{mlData.feedbackAnalysis.falseNegative} FN</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-orange-500 mr-2"></div>
                      <span className="text-xs">{mlData.feedbackAnalysis.falsePositive} FP</span>
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
