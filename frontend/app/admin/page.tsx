"use client"

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Image from "next/image"
import { Home, User, LayoutDashboard, Settings, Bell, LogOut, MoreHorizontal } from "lucide-react"

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<string | null>(null)
  const [activeOptionMenu, setActiveOptionMenu] = useState<number | null>(null)
  const [friends, setFriends] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<{ name: string; username: string }>({ name: '', username: '' })
  const [lastUpdateTime, setLastUpdateTime] = useState<number>(0)
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
  function roundNumber(num: number, decimalPlaces: number): number {
    const factor = Math.pow(10, decimalPlaces);
    return Math.round(num * factor) / factor;
  }

  // Fetch friends data
  useEffect(() => {
    const fetchFriends = async () => {
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          throw new Error('No token found')
        }

        // 1. Get current user data from localStorage
        const userData = localStorage.getItem('user')
        if (!userData) {
          throw new Error('No user data found')
        }
        const currentUser = JSON.parse(userData)
        setCurrentUser({
          name: currentUser.name,
          username: currentUser.username
        })

        // 2. Get all users to get friend names
        const usersResponse = await fetch('http://localhost:5000/api/auth/users', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!usersResponse.ok) {
          throw new Error('Failed to fetch users')
        }

        const usersData = await usersResponse.json()
        
        // Create a mapping of usernames to their full names
        const userNameMapping = usersData.users.reduce((acc: any, user: any) => {
          acc[user.username] = user.name
          return acc
        }, {})

        // 3. Get fresh contacts data
        const contactsResponse = await fetch('http://localhost:5000/api/chat/get_contacts_and_conversations', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!contactsResponse.ok) {
          throw new Error('Failed to fetch contacts')
        }

        const contactsData = await contactsResponse.json()
        console.log('Contacts data:', contactsData)

        interface FriendData {
          name: string
          friendTotalMessages: number
          friendToxicMessages: number
          toxicRate: string
        }

        // Transform the friends data using the current user's friends object
        const formattedFriends = Object.entries(currentUser.friends || {}).map(([username, data]: [string, any]): FriendData => {
          return {
            name: userNameMapping[username] || username,
            friendTotalMessages: data.messageCounter || 0,
            friendToxicMessages: data.toxicCounter || 0,
            toxicRate: data.messageCounter > 0 
              ? ((data.toxicCounter / data.messageCounter) * 100).toFixed(1) + "%" 
              : "0%"
          }
        }).sort((a: FriendData, b: FriendData) => b.friendTotalMessages - a.friendTotalMessages)

        console.log('Current user friends:', currentUser.friends)
        console.log('User name mapping:', userNameMapping)
        console.log('Formatted friends:', formattedFriends)

        setFriends(formattedFriends)
        setLastUpdateTime(Date.now())
      } catch (error) {
        console.error('Error fetching friends:', error)
      } finally {
        setIsLoading(false)
      }
    }

    // Call fetchFriends when component mounts
    fetchFriends()
  }, [])

  // Calculate totals from fetched data
  // const totalMessages = friends.reduce((sum, friend) => sum + friend.totalMessages, 0)
  // const toxicMessages = friends.reduce((sum, friend) => sum + friend.toxicMessages, 0)

  // Define interfaces for ML data
  interface ModelPerformance {
    f1Score: number;
    recall: number;
    precision: number;
    userFeedbackRate: number;
  }

  interface FeedbackAnalysis {
    truePositive: number;
    falseNegative: number;
    falsePositive: number;
    trueNegative: number;
  }

  interface MlData {
    systemTotalMessages: number;
    systemToxicMessages: number;
    userFeedback: {
      toxic: number;
      notToxic: number;
    };
    modelPerformance: ModelPerformance;
    feedbackAnalysis: FeedbackAnalysis;
  }

  // ML model data for DASHBOARD tab
  const [mlData, setMlData] = useState<MlData>({
    systemTotalMessages: 0,
    systemToxicMessages: 0,
    userFeedback: {
      toxic: 0,
      notToxic: 0,
    },
    modelPerformance: {
      f1Score: 0,
      recall: 0,
      precision: 0,
      userFeedbackRate: 0,
    },
    feedbackAnalysis: {
      truePositive: 0,
      falseNegative: 0,
      falsePositive: 0,
      trueNegative: 0,
    },
  });

  // Separate useEffect for fetching feedback data
  useEffect(() => {
    const fetchFeedbackData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No token found');
        }

        const feedbackResponse = await fetch('http://localhost:5000/api/settings/get-feedback-counts', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!feedbackResponse.ok) {
          throw new Error('Failed to fetch feedback counts');
        }

        const feedbackData = await feedbackResponse.json();
        console.log('Feedback data:', feedbackData); // Debug log
        
        setMlData(prevData => {
          console.log('Previous ML data:', prevData); // Debug log
          const newData = {
            ...prevData,
            userFeedback: {
              toxic: feedbackData.wrong_toxic_count || 0,
              notToxic: feedbackData.wrong_not_toxic_count || 0,
            }
          };
          console.log('New ML data:', newData); // Debug log
          return newData;
        });
      } catch (error) {
        console.error('Error fetching feedback data:', error);
      }
    };

    fetchFeedbackData();
  }, []); // No dependencies needed for feedback data

  // Separate useEffect for fetching total messages count
  useEffect(() => {
    const fetchTotalMessages = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No token found');
        }

        const response = await fetch('http://localhost:5000/api/settings/get-total-messages', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error('Failed to fetch total messages count');
        }

        const data = await response.json();
        console.log('Total messages data:', data); // Debug log
        
        setMlData(prevData => ({
          ...prevData,
          systemTotalMessages: data.total_messages || 0,
          systemToxicMessages: data.toxic_messages || 0
        }));
      } catch (error) {
        console.error('Error fetching total messages count:', error);
      }
    };

    fetchTotalMessages();
  }, []); // No dependencies needed for total messages count

  // Keep the existing useEffect for metrics calculations
  useEffect(() => {
    if (mlData.systemTotalMessages === 0) return;

    const total = mlData.systemTotalMessages;
    const totalFeedback = mlData.userFeedback.toxic + mlData.userFeedback.notToxic;
    const userFeedbackRate = total > 0 ? (totalFeedback / total) * 100 : 0;

    const falsePositive = mlData.userFeedback.notToxic;
    const falseNegative = mlData.userFeedback.toxic;
    const truePositive = mlData.systemToxicMessages - falsePositive;
    const trueNegative = total - mlData.systemToxicMessages - falseNegative;

    setMlData(prevData => ({
      ...prevData,
      modelPerformance: {
        f1Score: roundNumber(truePositive > 0 ? (2 * truePositive / (2 * truePositive + falsePositive + falseNegative)) * 100 : 0, 2),
        recall: roundNumber(truePositive > 0 ? (truePositive / (truePositive + falseNegative)) * 100 : 0, 2),
        precision: roundNumber(truePositive > 0 ? (truePositive / (truePositive + falsePositive)) * 100 : 0, 2),
        userFeedbackRate: roundNumber(userFeedbackRate, 2),
      },
      feedbackAnalysis: {
        truePositive: roundNumber(total > 0 ? (truePositive / total) * 100 : 0, 2),
        falseNegative: roundNumber(total > 0 ? (falseNegative / total) * 100 : 0, 2),
        falsePositive: roundNumber(total > 0 ? (falsePositive / total) * 100 : 0, 2),
        trueNegative: roundNumber(total > 0 ? (trueNegative / total) * 100 : 0, 2),
      }
    }));
  }, [mlData.systemTotalMessages, mlData.systemToxicMessages, mlData.userFeedback.toxic, mlData.userFeedback.notToxic]);

  // Find your toggleOptionMenu function and replace it with this version
  const toggleOptionMenu = (index: number) => {
    if (activeOptionMenu === index) {
      setActiveOptionMenu(null)
    } else {
      setActiveOptionMenu(index)
      // Position the dropdown after it's rendered
      setTimeout(() => {
        const button = document.querySelectorAll('.text-blue-500')[index] as HTMLElement;
        const dropdown = document.querySelector('.fixed.bg-white.shadow-lg') as HTMLElement;
      
        if (button && dropdown) {
          const rect = button.getBoundingClientRect();
          dropdown.style.top = `${rect.bottom + window.scrollY + 5}px`;
          dropdown.style.left = `${rect.left + window.scrollX - dropdown.offsetWidth + button.offsetWidth}px`;
        }
      }, 0);
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

        {/* Content based on active tab */}
        <div className="p-6">
        {activeTab === null && (
  <div className="flex flex-col items-start justify-start h-[calc(100vh-6rem)] p-6"> {/* Changed from center to start */}
    <div className="bg-white rounded-lg shadow-md p-10 w-full max-w-2xl"> {/* Increased padding and max width */}
      <div className="flex items-center space-x-8 mb-8"> {/* Increased spacing */}
        <div className="w-32 h-32 rounded-full bg-gray-200 overflow-hidden"> {/* Increased image size */}
          <Image src="/images/profile.png" alt={currentUser.name} width={128} height={128} className="object-cover" />
        </div>
        <div>
          <h2 className="text-4xl font-bold">{currentUser.name}</h2> {/* Larger text */}
          <p className="text-gray-500 text-xl">@{currentUser.username}</p> {/* Larger text */}
        </div>
      </div>
      <div className="space-y-4">
        <p className="text-gray-600 text-xl"> {/* Larger text */}
          Select an option from the sidebar to manage your
          application.
        </p>
      </div>
    </div>
  </div>
)}

          {activeTab === "USER" && (
            <div>
              <h2 className="text-xl font-bold mb-6">FRIEND LIST:</h2>
              {isLoading ? (
                <div className="flex justify-center items-center h-64">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
                </div>
              ) : (
                <div className="relative overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-gray-100">
                        <th className="border border-gray-300 px-4 py-2 text-left">Name</th>
                        <th className="border border-gray-300 px-4 py-2 text-center">Friend Total Messages</th>
                        <th className="border border-gray-300 px-4 py-2 text-center">Friend Toxic Messages</th>
                        <th className="border border-gray-300 px-4 py-2 text-center">Toxic Rate</th>
                        <th className="border border-gray-300 px-4 py-2 text-center">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {friends.map((friend, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="border border-gray-300 px-4 py-2">{friend.name}</td>
                          <td className="border border-gray-300 px-4 py-2 text-center">{friend.friendTotalMessages}</td>
                          <td className="border border-gray-300 px-4 py-2 text-center text-red-500">
                            {friend.friendToxicMessages}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 text-center text-red-500">
                            {friend.toxicRate}
                          </td>
                          <td className="border border-gray-300 px-4 py-2 text-center">
                            <div className="relative"> {/* Keep this relative container */}
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
                                  <div className="fixed bg-white shadow-lg rounded-md border border-gray-200 w-40 py-1 z-50"> {/* Changed from absolute to fixed for better positioning */}
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
                                  <button className="w-full text-left px-3 py-2 hover:bg-gray-100 flex items-center gap-2 text-green-500">
                                    <span>Uncheck</span>
                                  </button>
                                </div>
                                )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

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
                <h2 className="text-3xl font-bold">ML model analysis:</h2>
                <button className="px-4 py-1 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50">
                  Report
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-xl text-gray-500 mb-2">System Total Messages:</p>
                  <p className="text-5xl font-bold text-center">{mlData.systemTotalMessages}</p>
                </div>
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-xl text-gray-500 mb-2">System Toxic Messages:</p>
                  <p className="text-5xl font-bold text-center">{mlData.systemToxicMessages}</p>
                </div>
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <p className="text-xl text-gray-500 mb-2">User feedback:</p>
                  <div className="flex justify-center space-x-8">
                    <div className="text-center">
                      <p className="text-3xl text-red-500 font-bold">Toxic</p>
                      <p className="text-3xl font-bold">{mlData.userFeedback.toxic}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl text-green-500 font-bold">Not Toxic</p>
                      <p className="text-3xl font-bold">{mlData.userFeedback.notToxic}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-md shadow-sm">
                  <h3 className="text-3xl font-bold mb-4">Model Performance:</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>F1 score:</span>
                        <span>{mlData.modelPerformance.f1Score}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: `${Math.min(100, Number(mlData.modelPerformance.f1Score))}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Recall:</span>
                        <span>{mlData.modelPerformance.recall}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: `${Math.min(100, Number(mlData.modelPerformance.recall))}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Precision:</span>
                        <span>{mlData.modelPerformance.precision}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-500 h-2.5 rounded-full"
                          style={{ width: `${Math.min(100, Number(mlData.modelPerformance.precision))}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>User feedback rate:</span>
                        <span>{mlData.modelPerformance.userFeedbackRate}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-500 h-2.5 rounded-full" 
                          style={{ width: `${Math.min(100, Number(mlData.modelPerformance.userFeedbackRate))}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-4 rounded-md shadow-sm">
                  <h3 className="text-3xl font-bold mb-4">Users Feedback Analysis</h3>
                    <div className="flex flex-wrap justify-center gap-4 mb-4">
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.trueNegative}% TN</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-yellow-300 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.truePositive}% TP</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.falseNegative}% FN</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
                        <span className="text-sm font-medium">{mlData.feedbackAnalysis.falsePositive}% FP</span>
                      </div>
                    </div>
                    <div className="relative w-72 h-72 mx-auto">
                      {/* Fixed SVG Pie Chart */}
                      <svg viewBox="0 0 100 100" className="w-full h-full">
                        {(() => {
                          // Use the numeric values directly
                          const values = {
                            trueNegative: mlData.feedbackAnalysis.trueNegative,
                            truePositive: mlData.feedbackAnalysis.truePositive,
                            falseNegative: mlData.feedbackAnalysis.falseNegative,
                            falsePositive: mlData.feedbackAnalysis.falsePositive
                          };
                          
                          // Calculate total for percentages
                          const total = Object.values(values).reduce((sum, val) => sum + val, 0);
                          
                          // If total is 0, show empty chart
                          if (total === 0) {
                            return (
                              <circle cx="50" cy="50" r="40" fill="#e5e7eb" />
                            );
                          }
                          
                          // Define colors for each segment
                          const colors = {
                            trueNegative: "#3b82f6", // Blue
                            truePositive: "#fde047", // Yellow
                            falseNegative: "#22c55e", // Green
                            falsePositive: "#ef4444"  // Red
                          };
                          
                          // Start angle from the top (270 degrees / -90 degrees in radians)
                          let startAngle = -Math.PI / 2;
                          const paths: React.JSX.Element[] = [];
                          const radius = 40;
                          const centerX = 50;
                          const centerY = 50;
                          
                          // Helper function to create pie segment
                          const createSegment = (value: number, color: string, index: number) => {
                            if (value === 0) return null;
                            
                            // Calculate angle
                            const angle = (value / 100) * (2 * Math.PI);
                            const endAngle = startAngle + angle;
                            
                            // Calculate start and end points
                            const startX = centerX + radius * Math.cos(startAngle);
                            const startY = centerY + radius * Math.sin(startAngle);
                            const endX = centerX + radius * Math.cos(endAngle);
                            const endY = centerY + radius * Math.sin(endAngle);
                            
                            // Large arc flag is 1 if angle > π
                            const largeArcFlag = angle > Math.PI ? 1 : 0;
                            
                            // Create SVG path
                            const path = (
                              <path 
                                key={index}
                                d={`M ${centerX} ${centerY} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY} Z`}
                                fill={color}
                              />
                            );
                            
                            // Update startAngle for next segment
                            startAngle = endAngle;
                            
                            return path;
                          };
                          
                          // Create segments in order
                          const segments = [
                            { value: values.trueNegative, color: colors.trueNegative },
                            { value: values.truePositive, color: colors.truePositive },
                            { value: values.falseNegative, color: colors.falseNegative },
                            { value: values.falsePositive, color: colors.falsePositive }
                          ];
                          
                          // Generate paths for each segment
                          segments.forEach((segment, index) => {
                            const path = createSegment(segment.value, segment.color, index);
                            if (path) paths.push(path);
                          });
                          
                          return paths;
                        })()}
                      </svg>
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