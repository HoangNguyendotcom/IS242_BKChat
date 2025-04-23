"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, MoreVertical, Home, ImageIcon, Smile, Send } from "lucide-react"
import Image from "next/image"
import Link from "next/link"

interface Contact {
  id: string
  name: string
  username: string
  avatar: string
  lastMessage: string
  date: string
  reacted?: string
  reactedWith?: string
}

interface Message {
  id: string
  senderId: string
  text: string
  timestamp: string
  isEmoji?: boolean
  isToxic?: boolean
  userFeedback?: "toxic" | "not_toxic" | null
}

export default function MainPage() {
  const [selectedContact, setSelectedContact] = useState<string | null>(null) // Start with no contact selected
  const [messageInput, setMessageInput] = useState("")
  const optionsMenuRef = useRef<HTMLDivElement>(null);
  const [optionMenu, setOptionMenu] = useState<{
    visible: boolean
    messageId: string | null
    position: { top: number; left: number }
  }>({
    visible: false,
    messageId: null,
    position: { top: 0, left: 0 },
  })
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const contacts: Contact[] = [
    {
      id: "1",
      name: "Dac Hoang",
      username: "@ndhoang.sdh241",
      avatar: "/avatars/avatar1.jpg",
      lastMessage: "Dac Hoang reacted with",
      date: "Mar 23",
      reacted: "❤️",
    },
    {
      id: "2",
      name: "Tri Cuong",
      username: "@ntcuong.sdh241",
      avatar: "/avatars/avatar2.png",
      lastMessage: "Anh ăn cơm chưa?",
      date: "Mar 24",
    },
    {
      id: "3",
      name: "Tuan Nam",
      username: "@ntnam.sdh24",
      avatar: "/avatars/avatar3.avif",
      lastMessage: "Làm j đấy?",
      date: "Mar 25",
    },
    {
      id: "4",
      name: "Hoang Long",
      username: "@nbhlong.sdh24",
      avatar: "/avatars/avatar4.png",
      lastMessage: "Hello mrPip",
      date: "Mar 25",
    },
    {
      id: "5",
      name: "Le Phu",
      username: "@tlphu.sdh24",
      avatar: "/avatars/avatar5.png",
      lastMessage: "Oke em",
      date: "Mar 25",
    },
    {
      id: "6",
      name: "Hoang Minh",
      username: "@vhminh.sdh24",
      avatar: "/avatars/avatar2.png",
      lastMessage: "Hoang Minh reacted with",
      date: "Mar 25",
      reactedWith: "👍",
    },
  ]

  // Sample conversation with Dac Hoang
  const [conversations, setConversations] = useState<Record<string, Message[]>>({
    "1": [
      {
        id: "1",
        senderId: "1", // Dac Hoang
        text: "Ê m",
        timestamp: "Sat 5:10 AM",
        isToxic: false,
      },
      {
        id: "2",
        senderId: "1", // Dac Hoang
        text: "Đang làm gì đó?",
        timestamp: "Sat 5:10 AM",
        isToxic: false,
      },
      {
        id: "3",
        senderId: "1", // Dac Hoang
        text: "😊 😊 😊",
        timestamp: "Sat 5:10 AM",
        isEmoji: true,
        isToxic: false,
      },
      {
        id: "4",
        senderId: "current-user", // Current user
        text: "Có gì nói lẹ đi",
        timestamp: "Sat 5:15 PM",
        isToxic: false,
      },
      {
        id: "5",
        senderId: "current-user", // Current user
        text: "Đang bận lắm",
        timestamp: "Sat 5:15 PM",
        isToxic: false,
      },
      {
        id: "6",
        senderId: "current-user", // Current user
        text: "👍 👍",
        timestamp: "Sat 5:15 PM",
        isEmoji: true,
        isToxic: false,
      },
      {
        id: "7",
        senderId: "current-user", // Current user
        text: "Thêm m đó",
        timestamp: "Sat 5:17 PM",
        isToxic: true,
      },
      {
        id: "8",
        senderId: "1", // Dac Hoang
        text: "Bị điên hả, nhắn hỏi thăm thôi",
        timestamp: "Sat 5:17 PM",
        isToxic: true,
      },
    ],
  })

  const selectedContactData = contacts.find((contact) => contact.id === selectedContact)
  const currentConversation = selectedContact ? conversations[selectedContact] || [] : []

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [currentConversation])

  // Close option menu when clicking outside
  // MODIFIED: Added check to prevent closing when clicking on the menu itself
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      // Don't close if clicking on the menu itself or the button
      if ((e.target as Element).closest('.options-menu') || 
          (e.target as Element).closest('.options-toggle-button')) {
        return;
      }
      
      setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } })
    }

    document.addEventListener("click", handleClickOutside)

    return () => {
      document.removeEventListener("click", handleClickOutside)
    }
  }, [])

  const handleSendMessage = () => {
    if (messageInput.trim() && selectedContact) {
      const newMessage: Message = {
        id: `${Date.now()}`,
        senderId: "current-user",
        text: messageInput,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isToxic: false, // Default value, in a real app this would be determined by the ML model
      }

      setConversations((prev) => ({
        ...prev,
        [selectedContact]: [...(prev[selectedContact] || []), newMessage],
      }))

      setMessageInput("")
    }
  }

  // MODIFIED: Updated to position the menu based on screen boundaries
  const handleMessageOptions = (e: React.MouseEvent, messageId: string) => {
    e.stopPropagation();
    e.preventDefault(); 
    
    // Critical fix: Stop the event from propagating to document level
    // which would trigger the handleClickOutside function
    e.nativeEvent.stopImmediatePropagation();
    
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    
    // Get viewport dimensions
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    
    // Initial position
    let top = rect.bottom + window.scrollY;
    let left = rect.left + window.scrollX;
    
    // First show the menu with initial position
    setOptionMenu({
      visible: true,
      messageId,
      position: { top, left },
    });
    
    // Then check and adjust position after it's rendered
    setTimeout(() => {
      if (optionsMenuRef.current) {
        const menuRect = optionsMenuRef.current.getBoundingClientRect();
        
        // Check if menu would be off the bottom of the screen
        if (top + menuRect.height > window.scrollY + viewportHeight) {
          // Position menu above the button instead
          top = rect.top + window.scrollY - menuRect.height;
        }
        
        // Check if menu would be off the right of the screen
        if (left + menuRect.width > window.scrollX + viewportWidth) {
          left = window.scrollX + viewportWidth - menuRect.width - 10; // 10px margin
        }
        
        // Check if menu would be off the left of the screen
        if (left < window.scrollX) {
          left = window.scrollX + 10; // 10px margin
        }
        
        // Update with adjusted position
        setOptionMenu({
          visible: true,
          messageId,
          position: { top, left },
        });
      }
    }, 0);
  }

  const handleToxicFeedback = (messageId: string, isToxic: boolean) => {
    if (selectedContact) {
      setConversations((prev) => {
        const updatedConversation = prev[selectedContact].map((message) =>
          message.id === messageId ? { ...message, userFeedback: isToxic ? "toxic" : "not_toxic" } as Message : message,
        )

        return {
          ...prev,
          [selectedContact]: updatedConversation,
        }
      })
    }

    setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } })
  }
  
  const handleDeleteMessage = (messageId: string) => {
    if (selectedContact) {
      setConversations((prev) => {
        const updatedConversation = prev[selectedContact].filter((message) => message.id !== messageId)

        return {
          ...prev,
          [selectedContact]: updatedConversation,
        }
      })
    }

    setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } })
  }
  
  // Add this state for client-side rendering
  const [isClient, setIsClient] = useState(false)
      
  // Add this effect to set isClient to true after mounting
  useEffect(() => {
    setIsClient(true)
  }, [])

  return (
    <div className="flex h-screen bg-slate-100 p-4">
      {/* Left sidebar */}
      <div className="w-80 border-r flex flex-col">
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
          <Link 
            href="/main" 
            className="text-gray-500" 
            onClick={() => setSelectedContact(null)}
          >
            <Home className="h-5 w-5" />
          </Link>
        </div>

        {/* Messages header */}
        <div className="p-4 font-medium">Messages</div>

        {/* Search */}
        <div className="px-4 pb-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search people or messages"
              className="pl-9 bg-gray-100 border-0 focus-visible:ring-0 text-sm"
            />
          </div>
        </div>

        {/* Contacts list */}
        <div className="flex-1 overflow-auto">
          {contacts.map((contact) => (
            <button
              key={contact.id}
              className={`w-full text-left p-3 hover:bg-gray-50 flex items-start gap-3 ${
                selectedContact === contact.id ? "bg-gray-100" : ""
              }`}
              onClick={() => setSelectedContact(contact.id)}
            >
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
                  <Image src={contact.avatar || "/avatars/avatar.jpg"} alt={contact.name} width={40} height={40} />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between">
                  <p className="font-medium text-sm truncate">{contact.name}</p>
                  <span className="text-xs text-gray-500">{contact.date}</span>
                </div>
                <p className="text-xs text-gray-500">{contact.username}</p>
                <p className="text-xs text-gray-600 truncate flex items-center">
                  {contact.lastMessage} {contact.reacted && <span className="ml-1">{contact.reacted}</span>}
                  {contact.reactedWith && <span className="ml-1">{contact.reactedWith}</span>}
                </p>
              </div>
            </button>
          ))}
        </div>

        {/* User profile */}
        <div className="p-3 border-t flex items-center justify-between">
          <Link href="/admin" className="flex items-center gap-3">
            <div className="w-12 h-12 relative">
              {/* Conditional rendering based on client state */}
              {isClient ? (
                <Image
                  src="/images/profile.png"
                  alt="profile"
                  width={40}
                  height={40}
                  className="rounded-lg"
                />
              ) : (
                <div className="w-12 h-12 bg-gray-100 rounded-lg"></div> // Placeholder during SSR
              )}
            </div>
            <div>
              <p className="text-sm font-medium">Cong Minh</p>
              <p className="text-xs text-gray-500">@tcminh.sdh24</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Main content */}
      {selectedContactData ? (
        <div className="flex-1 flex flex-col">
          {/* Chat header */}
          <div className="flex items-center p-4 border-b">
            <div className="flex-1 flex items-center">
              <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden mr-3">
                <Image
                  src={selectedContactData.avatar || "/placeholder.svg"}
                  alt={selectedContactData.name}
                  width={40}
                  height={40}
                />
              </div>
              <div>
                <p className="font-medium">{selectedContactData.name}</p>
                <p className="text-xs text-gray-500">{selectedContactData.username}</p>
              </div>
            </div>
          </div>

          {/* Chat messages */}
          <div className="flex-1 overflow-auto p-4">
            <div className="space-y-4">
              {currentConversation.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.senderId === "current-user" ? "justify-end" : "justify-start"}`}
                >
                  {message.senderId !== "current-user" && (
                    <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden mr-2 flex-shrink-0">
                      <Image
                        src={selectedContactData.avatar || "/avatar/avatar1.jpg"}
                        alt={selectedContactData.name}
                        width={32}
                        height={32}
                      />
                    </div>
                  )}
                  <div className="flex flex-col relative group">
                    <div
                      className={`rounded-lg px-4 py-2 max-w-xs flex items-center relative ${
                        message.senderId === "current-user" ? "bg-blue-200 text-blue-900" : "bg-gray-100 text-gray-900"
                      } ${message.isEmoji ? "text-2xl bg-transparent px-0" : ""}`}
                    >
                      {/* For current user messages, show options button on the left */}
                      {message.senderId === "current-user" && (
                        <button
                          className="mr-2 text-gray-500 hover:text-gray-700 z-10 options-toggle-button"
                          onClick={(e) => handleMessageOptions(e, message.id)}
                        >
                          <MoreVertical className="h-4 w-4" />
                        </button>
                      )}
                      
                      <span className="flex-1">{message.text}</span>
                      
                      {/* For other user messages, show options button on the right */}
                      {message.senderId !== "current-user" && (
                        <button
                          className="ml-2 text-gray-500 hover:text-gray-700 z-10 options-toggle-button"
                          onClick={(e) => handleMessageOptions(e, message.id)}
                        >
                          <MoreVertical className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                    <div className="flex items-center mt-1">
                      <span className="text-xs text-gray-500">{message.timestamp}</span>
                      {message.userFeedback && (
                        <span
                          className={`ml-2 text-xs ${message.userFeedback === "toxic" ? "text-red-500" : "text-green-500"}`}
                          >
                          • {message.userFeedback === "toxic" ? "Marked as toxic" : "Marked as not toxic"}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Message options menu */}
          {optionMenu.visible && (
            <div
              ref={optionsMenuRef}
              className="fixed bg-white shadow-md rounded-md py-1 z-50 border border-gray-200 options-menu"
              style={{ 
                top: optionMenu.position.top, 
                left: optionMenu.position.left,
                minWidth: '200px' 
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="px-4 py-2 text-center font-medium border-b border-gray-100">OPTION</div>
              {/* Find the current message to determine its feedback state */}
              {optionMenu.messageId && currentConversation.find(msg => msg.id === optionMenu.messageId)?.userFeedback === "toxic" ? (
                <button
                  className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-green-500"
                  onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, false)}
                >
                  ... Not a toxic message
                </button>
              ) : (
                <button
                  className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-500"
                  onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, true)}
                >
                  ... Toxic message
                </button>
              )}
              <button
                className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100"
                onClick={() => optionMenu.messageId && handleDeleteMessage(optionMenu.messageId)}
              >
                Delete message
              </button>
            </div>
          )}

          {/* Message input */}
          <div className="border-t p-3 flex items-center">
            <button className="p-2 text-gray-500">
              <ImageIcon className="h-5 w-5" />
            </button>
            <div className="flex-1 mx-2">
              <Input
                placeholder="Nói bậy bạ là t chém m luôn đó"
                className="border-0 focus-visible:ring-0"
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
              />
            </div>
            <button className="p-2 text-gray-500">
              <Smile className="h-5 w-5" />
            </button>
            <button className="p-2 text-green-500" onClick={handleSendMessage}>
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center p-6">
          <div className="text-center">
            <h2 className="text-2xl font-semibold mb-3">You don't have a conversation selected.</h2>
            <p className="text-lg text-gray-500 mb-8">Choose one from your existing conversation, or start a new one.</p>
            <Button className="bg-blue-500 hover:bg-blue-600 text-lg py-2 px-4">New Conversation</Button>
          </div>
        </div>
      )}

    </div>
  )
}