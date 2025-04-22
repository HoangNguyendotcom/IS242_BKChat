"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Send, PlusCircle, Paperclip, Smile } from "lucide-react"

interface Message {
  id: number
  content: string
  sender: "user" | "other"
  timestamp: Date
}

interface Conversation {
  id: number
  name: string
  avatar: string
  lastMessage: string
  lastMessageTime: string
  unread: number
  online: boolean
}

export default function MessagesPage() {
  const [activeConversation, setActiveConversation] = useState<number>(1)
  const [messageInput, setMessageInput] = useState("")
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, content: "Hey there! How are you doing?", sender: "other", timestamp: new Date(Date.now() - 3600000) },
    {
      id: 2,
      content: "I'm good, thanks for asking! How about you?",
      sender: "user",
      timestamp: new Date(Date.now() - 3000000),
    },
    {
      id: 3,
      content: "I'm doing well. Just working on some new features for the app.",
      sender: "other",
      timestamp: new Date(Date.now() - 2400000),
    },
    {
      id: 4,
      content: "That sounds interesting! What kind of features?",
      sender: "user",
      timestamp: new Date(Date.now() - 1800000),
    },
    {
      id: 5,
      content: "We're adding real-time messaging and improved friend management.",
      sender: "other",
      timestamp: new Date(Date.now() - 1200000),
    },
    {
      id: 6,
      content: "That's awesome! Can't wait to try it out.",
      sender: "user",
      timestamp: new Date(Date.now() - 600000),
    },
  ])

  const conversations: Conversation[] = [
    {
      id: 1,
      name: "Jane Smith",
      avatar: "",
      lastMessage: "That's awesome! Can't wait to try it out.",
      lastMessageTime: "10m",
      unread: 0,
      online: true,
    },
    {
      id: 2,
      name: "Alex Johnson",
      avatar: "",
      lastMessage: "Are we still meeting tomorrow?",
      lastMessageTime: "1h",
      unread: 3,
      online: true,
    },
    {
      id: 3,
      name: "Sam Wilson",
      avatar: "",
      lastMessage: "Thanks for the help!",
      lastMessageTime: "3h",
      unread: 0,
      online: false,
    },
    {
      id: 4,
      name: "Taylor Brown",
      avatar: "",
      lastMessage: "I'll send you the files later.",
      lastMessageTime: "1d",
      unread: 0,
      online: false,
    },
    {
      id: 5,
      name: "Morgan Lee",
      avatar: "",
      lastMessage: "Let me know when you're free.",
      lastMessageTime: "2d",
      unread: 0,
      online: true,
    },
  ]

  const handleSendMessage = () => {
    if (messageInput.trim()) {
      const newMessage: Message = {
        id: messages.length + 1,
        content: messageInput,
        sender: "user",
        timestamp: new Date(),
      }
      setMessages([...messages, newMessage])
      setMessageInput("")

      // Simulate reply after 1 second
      setTimeout(() => {
        const reply: Message = {
          id: messages.length + 2,
          content: "Thanks for your message! I'll get back to you soon.",
          sender: "other",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, reply])
      }, 1000)
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)] md:h-screen">
      {/* Conversation List */}
      <div className="hidden md:flex w-80 flex-col border-r dark:border-slate-800">
        <div className="p-4 border-b dark:border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold">Messages</h2>
            <Button variant="ghost" size="icon">
              <PlusCircle className="h-5 w-5" />
              <span className="sr-only">New conversation</span>
            </Button>
          </div>
          <div className="relative">
            <Input placeholder="Search conversations..." className="pl-8" />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-500 dark:text-slate-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>
        <div className="flex-1 overflow-auto">
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              className={`flex items-center gap-3 w-full p-4 text-left hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors ${
                activeConversation === conversation.id ? "bg-slate-100 dark:bg-slate-800" : ""
              }`}
              onClick={() => setActiveConversation(conversation.id)}
            >
              <div className="relative">
                <Avatar>
                  <AvatarImage
                    src={conversation.avatar || `/placeholder.svg?height=40&width=40`}
                    alt={conversation.name}
                  />
                  <AvatarFallback>
                    {conversation.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </AvatarFallback>
                </Avatar>
                {conversation.online && (
                  <span className="absolute bottom-0 right-0 h-3 w-3 rounded-full bg-green-500 border-2 border-white dark:border-slate-950"></span>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-baseline">
                  <p className="font-medium truncate">{conversation.name}</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
                    {conversation.lastMessageTime}
                  </p>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400 truncate">{conversation.lastMessage}</p>
              </div>
              {conversation.unread > 0 && (
                <div className="flex-shrink-0 h-5 w-5 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-xs text-white font-medium">{conversation.unread}</span>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="flex items-center p-4 border-b dark:border-slate-800">
          <Avatar>
            <AvatarImage src={`/placeholder.svg?height=40&width=40`} alt="Jane Smith" />
            <AvatarFallback>JS</AvatarFallback>
          </Avatar>
          <div className="ml-3">
            <p className="font-medium">Jane Smith</p>
            <p className="text-xs text-green-600 dark:text-green-400">Online</p>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.sender === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-slate-200 dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                }`}
              >
                <p>{message.content}</p>
                <p className="text-xs mt-1 opacity-70">{formatTime(message.timestamp)}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Message Input */}
        <div className="p-4 border-t dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon">
              <Paperclip className="h-5 w-5" />
              <span className="sr-only">Attach file</span>
            </Button>
            <Input
              placeholder="Type a message..."
              value={messageInput}
              onChange={(e) => setMessageInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              className="flex-1"
            />
            <Button variant="ghost" size="icon">
              <Smile className="h-5 w-5" />
              <span className="sr-only">Add emoji</span>
            </Button>
            <Button size="icon" onClick={handleSendMessage}>
              <Send className="h-5 w-5" />
              <span className="sr-only">Send message</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
