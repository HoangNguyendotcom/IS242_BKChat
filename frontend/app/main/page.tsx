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
  const [selectedContact, setSelectedContact] = useState<string | null>("1") // Default to first contact
  const [messageInput, setMessageInput] = useState("")
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
      username: "@dachoang",
      avatar: "/placeholder.svg?height=40&width=40",
      lastMessage: "Dac Hoang reacted with",
      date: "Mar 23",
      reacted: "❤️",
    },
    {
      id: "2",
      name: "Tri Cuong",
      username: "@tricuong",
      avatar: "/placeholder.svg?height=40&width=40",
      lastMessage: "Anh ăn cơm chưa?",
      date: "Mar 24",
    },
    {
      id: "3",
      name: "Tuan Nam",
      username: "@tuannam",
      avatar: "/placeholder.svg?height=40&width=40",
      lastMessage: "Làm j đấy?",
      date: "Mar 25",
    },
    {
      id: "4",
      name: "Hoang Long",
      username: "@hbhlong",
      avatar: "/placeholder.svg?height=40&width=40",
      lastMessage: "Còn nhớ mua cà cây Minh?",
      date: "Mar 25",
    },
    {
      id: "5",
      name: "Le Phu",
      username: "@lephu",
      avatar: "/placeholder.svg?height=40&width=40",
      lastMessage: "Oke em",
      date: "Mar 25",
    },
    {
      id: "6",
      name: "Hoang Minh",
      username: "@hminh",
      avatar: "/placeholder.svg?height=40&width=40",
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
  useEffect(() => {
    const handleClickOutside = () => {
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

  const handleMessageOptions = (e: React.MouseEvent, messageId: string) => {
    e.stopPropagation() // Prevent the click from closing the menu

    const rect = (e.target as HTMLElement).getBoundingClientRect()

    setOptionMenu({
      visible: true,
      messageId,
      position: {
        top: rect.bottom + window.scrollY,
        left: rect.left + window.scrollX,
      },
    })
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

  return (
    <div className="flex h-screen bg-white">
      {/* Left sidebar */}
      <div className="w-80 border-r flex flex-col">
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
                  <Image src={contact.avatar || "/placeholder.svg"} alt={contact.name} width={40} height={40} />
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
            <div className="w-9 h-9 rounded-full bg-gray-200 overflow-hidden">
              <Image src="/placeholder.svg?height=36&width=36" alt="User" width={36} height={36} />
            </div>
            <div>
              <p className="text-sm font-medium">Minh Trinh</p>
              <p className="text-xs text-gray-500">@minhtrinhk241</p>
            </div>
          </Link>
          <button>
            <MoreVertical className="h-5 w-5 text-gray-500" />
          </button>
        </div>
      </div>

      {/* Main content */}
      {selectedContactData ? (
        <div className="flex-1 flex flex-col">
          {/* Chat header */}
          <div className="flex items-center p-4 border-b">
            <div className="flex-1 flex items-center">
              <Link href="/main" className="mr-4">
                <Home className="h-5 w-5 text-gray-500" />
              </Link>
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
                        src={selectedContactData.avatar || "/placeholder.svg"}
                        alt={selectedContactData.name}
                        width={32}
                        height={32}
                      />
                    </div>
                  )}
                  <div className="flex flex-col relative group">
                    <div
                      className={`rounded-lg px-4 py-2 max-w-xs ${
                        message.senderId === "current-user" ? "bg-blue-200 text-blue-900" : "bg-gray-100 text-gray-900"
                      } ${message.isEmoji ? "text-2xl bg-transparent px-0" : ""}`}
                    >
                      {message.text}
                      <button
                        className="absolute right-0 top-0 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => handleMessageOptions(e, message.id)}
                      >
                        <MoreVertical className="h-4 w-4 text-gray-500" />
                      </button>
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
            <h2 className="text-xl font-semibold mb-2">You don't have a message selected.</h2>
            <p className="text-gray-500 mb-6">Choose one from your existing messages, or start a new one.</p>
            <Button className="bg-blue-500 hover:bg-blue-600">New Message</Button>
          </div>
        </div>
      )}

      {/* Message options menu */}
      {optionMenu.visible && (
        <div
          className="absolute bg-white shadow-md rounded-md py-1 z-50 border border-gray-200"
          style={{ top: optionMenu.position.top, left: optionMenu.position.left }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="px-4 py-2 text-center font-medium border-b border-gray-100">OPTION</div>
          <button
            className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-blue-500"
            onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, false)}
          >
            Not a toxic message
          </button>
          <button
            className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-500"
            onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, true)}
          >
            Toxic message
          </button>
          <button
            className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100"
            onClick={() => optionMenu.messageId && handleDeleteMessage(optionMenu.messageId)}
          >
            Delete message
          </button>
        </div>
      )}
    </div>
  )
}
