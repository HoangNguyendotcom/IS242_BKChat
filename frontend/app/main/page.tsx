"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, MoreVertical, Home, ImageIcon, Smile, Send, X } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { useRouter } from "next/navigation"

interface Contact {
  id: string
  name: string
  username: string
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
  const [selectedContact, setSelectedContact] = useState<string | null>(null);
  const [messageInput, setMessageInput] = useState("");
  const optionsMenuRef = useRef<HTMLDivElement>(null);
  const [optionMenu, setOptionMenu] = useState<{
    visible: boolean;
    messageId: string | null;
    position: { top: number; left: number };
  }>({
    visible: false,
    messageId: null,
    position: { top: 0, left: 0 }
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const defaultAvatar = "/avatars/avatar.jpeg";

  // New search functionality
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchDropdownOpen, setIsSearchDropdownOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchDropdownRef = useRef<HTMLDivElement>(null);

  const [contacts, setContacts] = useState<Contact[]>([]);
  const [conversations, setConversations] = useState<Record<string, Message[]>>({});

  // Get user function that checks localStorage
  const getUser = () => {
    // First try localStorage
    const user = localStorage.getItem('user');
    
    if (user) {
      return JSON.parse(user);
    }
    
    return null;
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const user = getUser();
        
        if (!user) {
          console.error('User not found in localStorage');
          // Redirect to login page if no user found
          router.push('/');
          return;
        }
        
        console.log('Found user, fetching data...');
        
        const response = await fetch('http://localhost:5000/api/chat/get_contacts_and_conversations');
        
        if (!response.ok) {
          // Get the error details from the response
          const errorText = await response.text();
          console.error(`API Error: ${response.status}`, errorText);

          throw new Error(`API request failed with status ${response.status}`);
        }
        
        const data = await response.json();
        setContacts(data.contacts);
        setConversations(data.conversations);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [router]);

  // Function to filter contacts based on search query
  const getFilteredContacts = () => {
    if (!searchQuery.trim()) return contacts;
    
    return contacts.filter(contact => 
      contact.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
      contact.username.toLowerCase().includes(searchQuery.toLowerCase())
    );
  };

  // Handle search input change
  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setIsSearchDropdownOpen(true);
  };

  // Handle selection from search dropdown
  const handleSelectContact = (contactId: string) => {
    setSelectedContact(contactId);
    setIsSearchDropdownOpen(false);
    setSearchQuery("");
    
    // Optional: Focus back on input after selection
    if (searchInputRef.current) {
      searchInputRef.current.blur();
    }
  };

  // Setup click outside listeners for search dropdown
  useEffect(() => {
    // Handler for click outside
    const handleClickOutside = (event: MouseEvent) => {
      // If click is outside both the input and dropdown
      const clickedElement = event.target as Node;
      const isOutsideInput = searchInputRef.current && !searchInputRef.current.contains(clickedElement);
      const isOutsideDropdown = searchDropdownRef.current && !searchDropdownRef.current.contains(clickedElement);
      
      if (isOutsideInput && isOutsideDropdown && isSearchDropdownOpen) {
        setIsSearchDropdownOpen(false);
      }
    };
    
    // Add event listener
    document.addEventListener('mousedown', handleClickOutside);
    
    // Clean up
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isSearchDropdownOpen]);

  // Get selected contact data and conversation
  const selectedContactData = contacts.find((contact) => contact.id === selectedContact)
  const currentConversation = selectedContact ? conversations[selectedContact] || [] : []

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [currentConversation])

  // Close option menu when clicking outside
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

  const handleSendMessage = async () => {
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

      // Here you could also send the message to your backend
      // const token = getToken();
      // if (token) {
      //   try {
      //     await fetch('http://localhost:5000/api/chat/send_message', {
      //       method: 'POST',
      //       headers: {
      //         'Authorization': `Bearer ${token}`,
      //         'Content-Type': 'application/json',
      //       },
      //       body: JSON.stringify({
      //         contactId: selectedContact,
      //         message: messageInput
      //       }),
      //     });
      //   } catch (error) {
      //     console.error('Error sending message:', error);
      //   }
      // }

      setMessageInput("")
    }
  }

  // Updated to position the menu based on screen boundaries
  const handleMessageOptions = (e: React.MouseEvent, messageId: string) => {
    e.stopPropagation();
    e.preventDefault(); 
    
    // Critical fix: Stop the event from propagating to document level
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
        if (left < window.scrollY) {
          left = window.scrollY + 10; // 10px margin
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
    <div className="flex h-screen bg-slate-100 p-4 ">
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

        {/* Enhanced Search with dropdown */}
        <div className="px-4 pb-2 relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              ref={searchInputRef}
              placeholder="Search people or messages"
              className="pl-9 bg-gray-100 border-0 focus-visible:ring-0 text-sm"
              value={searchQuery}
              onChange={handleSearchInputChange}
              onFocus={() => setIsSearchDropdownOpen(true)}
            />
            {searchQuery && (
              <button 
                className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400"
                onClick={() => {
                  setSearchQuery("");
                  setIsSearchDropdownOpen(true);
                  if (searchInputRef.current) searchInputRef.current.focus();
                }}
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          
          {/* Search dropdown */}
          {isSearchDropdownOpen && (
            <div 
              ref={searchDropdownRef}
              className="absolute z-10 mt-1 w-full bg-white rounded-md shadow-lg max-h-60 overflow-auto"
            >
              {getFilteredContacts().length > 0 ? (
                getFilteredContacts().map((contact) => (
                  <button
                    key={contact.id}
                    className="w-full text-left p-3 hover:bg-gray-50 flex items-start gap-3 border-b border-gray-100"
                    onClick={() => handleSelectContact(contact.id)}
                  >
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden">
                        <Image src={defaultAvatar} alt={contact.name} width={32} height={32} />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm text-gray-800">{contact.name}</p>
                      <p className="text-xs text-gray-500">{contact.username}</p>
                    </div>
                  </button>
                ))
              ) : (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No results found
                </div>
              )}
            </div>
          )}
        </div>

        {/* Contacts list with darker selection background */}
        <div className="flex-1 overflow-auto ">
          {contacts.map((contact) => (
            <button
              key={contact.id}
              className={`w-full text-left p-3 hover:bg-gray-50 flex items-start gap-3 ${
                selectedContact === contact.id ? "bg-gray-200" : ""
              }`}
              onClick={() => setSelectedContact(contact.id)}
            >
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
                  <Image src={"/avatars/avatar.jpg"} alt={contact.name} width={40} height={40} />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between">
                  <p className="font-medium text-sm truncate text-gray-800 ">{contact.name}</p>
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
          {isClient && getUser() ? (
            <Link href="/admin" className="flex items-center gap-3">
              <div className="w-12 h-12 relative">
                <Image
                  src="/images/profile.png"
                  alt="profile"
                  width={40}
                  height={40}
                  className="rounded-lg"
                />
              </div>
              <div>
                <p className="text-sm font-medium">{getUser().username}</p>
                <p className="text-xs text-gray-500">@{getUser().username}</p>
              </div>
            </Link>
          ) : (
            <div>Loading...</div>
          )}
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
                  src={defaultAvatar}
                  alt={selectedContactData.name}
                  width={40}
                  height={40}
                />
              </div>
              <div>
                <p className="font-medium text-gray-800">{selectedContactData.name}</p>
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
                        src={defaultAvatar}
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
              placeholder="Input something..."
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
