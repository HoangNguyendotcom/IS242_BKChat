"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, MoreVertical, Home, ImageIcon, Smile, Send, X, UserPlus } from "lucide-react"
import Image from "next/image"
import Link from "next/link"

interface Contact {
  id: string
  name: string
  username: string
  avatar: string
  lastMessage: string
  date: string
  reacted?: boolean
  reactedWith?: string
}

interface Message {
  id: string
  senderId: string
  text: string
  timestamp: string
  isEmoji?: boolean
  isToxic?: boolean
  userFeedback?: "Toxic" | "Not Toxic" | null
  isHidden?: boolean
}

interface CurrentUser {
  _id: string
  name: string
  username: string
  avatar?: string
}

interface User {
  _id: string
  name: string
  username: string
  avatar?: string
  isFriend?: boolean
}

export default function MainPage() {
  const [selectedContact, setSelectedContact] = useState<string | null>(null)
  const [messageInput, setMessageInput] = useState("")
  const [contacts, setContacts] = useState<Contact[]>([])
  const [allUsers, setAllUsers] = useState<User[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [messages, setMessages] = useState<Message[]>([])
  const [isMessagesLoading, setIsMessagesLoading] = useState(false)
  const [toxicWarningMessage, setToxicWarningMessage] = useState<string | null>(null)
  const [pendingToxicMessage, setPendingToxicMessage] = useState<{text: string} | null>(null)
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
  
  // New search functionality
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearchDropdownOpen, setIsSearchDropdownOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const searchDropdownRef = useRef<HTMLDivElement>(null)

  const [conversations, setConversations] = useState<Record<string, Message[]>>({})
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null)

  // Fetch contacts when component mounts
  useEffect(() => {
    const fetchContacts = async () => {
      try {
        // Check if we're in a browser environment
        if (typeof window === 'undefined') {
          console.error('Not in browser environment')
          return
        }

        const token = localStorage.getItem('token')
        console.log('Token from localStorage:', token) // Debug log
        
        if (!token) {
          console.error('No token found in localStorage')
          return
        }

        // Verify token format
        const tokenParts = token.split('.')
        if (tokenParts.length !== 3) {
          console.error('Invalid token format')
          return
        }

        console.log('Fetching contacts with token...') // Debug log
        const response = await fetch('http://localhost:5000/api/chat/get_contacts_and_conversations', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
        
        console.log('Response status:', response.status) // Debug log
        if (!response.ok) {
          const errorData = await response.json()
          console.error('Error response:', errorData) // Debug log
          throw new Error('Failed to fetch contacts')
        }
        
        const data = await response.json()
        console.log('Response data:', data) // Debug log
        
        if (data.contacts) {
          const formattedContacts = data.contacts.map((contact: any) => ({
            id: contact._id,
            name: contact.name,
            username: contact.username,
            avatar: contact.avatar || "/avatars/avatar.jpeg",
            lastMessage: contact.lastMessage || '',
            date: contact.date || '',
            reacted: contact.reacted || undefined,
            reactedWith: contact.reactedWith || undefined
          }))
          
          setContacts(formattedContacts)
        }
      } catch (error) {
        console.error('Error fetching contacts:', error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchContacts()
  }, [])

  // Fetch current user data
  useEffect(() => {
    const fetchCurrentUser = async () => {
      try {
        // Get user data from localStorage
        const userData = localStorage.getItem('user')
        if (!userData) {
          throw new Error('No user data found')
        }

        const user = JSON.parse(userData)
        setCurrentUser({
          _id: user._id,
          name: user.name,
          username: user.username
        })
        
      } catch (error) {
        console.error('Error getting current user:', error)
      }
    }

    fetchCurrentUser()
  }, [])

  // Fetch messages when a contact is selected
  useEffect(() => {
    const fetchMessages = async () => {
      if (!selectedContact) return
      
      setIsMessagesLoading(true)
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          console.error('No token found')
          return
        }

        const response = await fetch(`http://localhost:5000/api/chat/messages/${selectedContact}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
        
        if (!response.ok) {
          throw new Error('Failed to fetch messages')
        }
        
        const data = await response.json()
        
        if (data.messages) {
          const formattedMessages = data.messages.map((msg: any) => ({
            id: msg._id,
            senderId: msg.senderId,
            text: msg.text,
            timestamp: new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            isEmoji: msg.isEmoji,
            isToxic: msg.isToxic,
            userFeedback: msg.userFeedback,
            isHidden: msg.isToxic || msg.userFeedback === "Toxic"
          }))
          setMessages(formattedMessages)
        }
      } catch (error) {
        console.error('Error fetching messages:', error)
      } finally {
        setIsMessagesLoading(false)
      }
    }

    fetchMessages()
  }, [selectedContact])

  // Fetch all users when component mounts
  useEffect(() => {
    const fetchAllUsers = async () => {
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          console.error('No token found')
          return
        }

        const response = await fetch('http://localhost:5000/api/auth/users', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        
        if (!response.ok) {
          throw new Error('Failed to fetch users')
        }
        
        const data = await response.json()
        
        if (data.users) {
          const formattedUsers = data.users.map((user: any) => ({
            _id: user._id,
            name: user.name || user.username,  // Fallback to username if name is not available
            username: user.username,
            avatar: user.avatar || "/avatars/avatar.jpeg",
            isFriend: contacts.some(contact => contact.id === user._id)
          }))
          setAllUsers(formattedUsers)
        }
      } catch (error) {
        console.error('Error fetching users:', error)
        // Set empty array to prevent continuous loading state
        setAllUsers([])
      }
    }

    fetchAllUsers()
  }, [contacts])

  // Function to filter users based on search query
  const getFilteredUsers = () => {
    if (!searchQuery.trim()) return allUsers;
    
    return allUsers.filter(user => 
      user.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
      user.username.toLowerCase().includes(searchQuery.toLowerCase())
    );
  };

  // Handle search input change
  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setIsSearchDropdownOpen(true);
  };

  // Handle selection from search dropdown
  const handleSelectUser = async (userId: string) => {
    const selectedUser = allUsers.find(user => user._id === userId);
    if (!selectedUser) return;

    setSelectedContact(userId);
    setIsSearchDropdownOpen(false);
    setSearchQuery("");
    
    if (searchInputRef.current) {
      searchInputRef.current.blur();
    }
  };

  // Add new function to handle adding friend
  const handleAddFriend = async (userId: string) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.error('No token found');
        return;
      }

      // Get current user's username from localStorage
      const userData = localStorage.getItem('user');
      if (!userData) {
        console.error('No user data found');
        return;
      }
      const currentUser = JSON.parse(userData);

      const response = await fetch('http://localhost:5000/api/friends/add_friend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          username: currentUser.username,
          friend_username: allUsers.find(user => user._id === userId)?.username
        })
      });

      if (!response.ok) {
        throw new Error('Failed to add friend');
      }

      // Add the new contact to the contacts list
      const newFriend = allUsers.find(user => user._id === userId);
      if (newFriend) {
        const newContact: Contact = {
          id: newFriend._id,
          name: newFriend.name,
          username: newFriend.username,
          avatar: newFriend.avatar || "/avatars/avatar.jpeg",
          lastMessage: '',
          date: new Date().toISOString()
        };
        setContacts(prev => [...prev, newContact]);
        
        // Update user's friend status
        setAllUsers(prev => 
          prev.map(user => 
            user._id === userId ? { ...user, isFriend: true } : user
          )
        );
      }
    } catch (error) {
      console.error('Error adding friend:', error);
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

  const checkToxicity = async (text: string): Promise<boolean> => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        console.error('No token found')
        return false
      }

      // Use the existing messages endpoint with a special flag to only check toxicity
      const response = await fetch('http://localhost:5000/api/chat/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          receiverId: selectedContact,
          text: text,
          isEmoji: false,
          checkOnly: true,
          checkToxicity: true // Add this flag to check toxicity
        })
      })

      if (!response.ok) {
        throw new Error('Failed to check toxicity')
      }

      const data = await response.json()
      return data.isToxic
    } catch (error) {
      console.error('Error checking toxicity:', error)
      return false
    }
  }

  const handleMessageSubmit = async () => {
    if (!messageInput.trim() || !selectedContact) return

    try {
      // First check toxicity
      const isToxic = await checkToxicity(messageInput)
      
      if (isToxic) {
        // If toxic, store the message and show warning
        setPendingToxicMessage({ text: messageInput })
        setToxicWarningMessage(messageInput)
        return
      }

      // If not toxic, send immediately
      await sendMessage(messageInput, false)
    } catch (error) {
      console.error('Error handling message submit:', error)
    }
  }

  const sendMessage = async (text: string, isToxic: boolean) => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        console.error('No token found')
        return
      }

      const response = await fetch('http://localhost:5000/api/chat/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          receiverId: selectedContact,
          text: text,
          isEmoji: false,
          checkOnly: false, // This is an actual send
          checkToxicity: isToxic // Add this flag to check toxicity if needed
        })
      })

      if (!response.ok) {
        throw new Error('Failed to send message')
      }

      const data = await response.json()

      // Add the new message to the local state
      const newMessage = {
        id: data.messageId,
        senderId: currentUser?._id || '',
        text: text,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isEmoji: false,
        isToxic: isToxic,
        userFeedback: null,
        isHidden: isToxic
      }

      setMessages(prev => [...prev, newMessage])
      setMessageInput("")
      setToxicWarningMessage(null)
      setPendingToxicMessage(null)
    } catch (error) {
      console.error('Error sending message:', error)
    }
  }

  const handleSendAnyway = async () => {
    if (pendingToxicMessage) {
      await sendMessage(pendingToxicMessage.text, true)
    }
  }

  // Updated to position the menu based on screen boundaries
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

  const handleToxicFeedback = async (messageId: string, isToxic: boolean) => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        console.error('No token found')
        return
      }

      const response = await fetch(`http://localhost:5000/api/chat/messages/${messageId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          userFeedback: isToxic ? "Toxic" : "Not Toxic"
        })
      })

      if (!response.ok) {
        throw new Error('Failed to update message feedback')
      }

      // Update local state after successful API call
      setMessages((prevMessages) => {
        return prevMessages.map((message) =>
          message.id === messageId 
            ? { 
                ...message, 
                userFeedback: isToxic ? "Toxic" : "Not Toxic",
                isHidden: isToxic
              } 
            : message
        )
      })

      // Close the option menu after setting feedback
      setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } })
    } catch (error) {
      console.error('Error updating message feedback:', error)
      alert('Failed to update message feedback. Please try again.')
    }
  }
  
  const handleDeleteMessage = async (messageId: string) => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        alert('You need to be logged in to delete messages')
        return
      }

      // First confirm with the user
      if (!window.confirm('Are you sure you want to delete this message?')) {
        return
      }

      // Call API to delete message
      const response = await fetch(`http://localhost:5000/api/chat/messages/${messageId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.message || `Failed to delete message (${response.status})`)
      }

      // If successful, update local state
      setMessages((prevMessages) => 
        prevMessages.filter((message) => message.id !== messageId)
      )

      // Close the option menu
      setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } })

    } catch (error) {
      console.error('Error deleting message:', error)
      alert(error instanceof Error ? error.message : 'Failed to delete message. Please try again.')
    }
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
              placeholder="Search users..."
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
          
          {/* Updated search dropdown */}
          {isSearchDropdownOpen && (
            <div 
              ref={searchDropdownRef}
              className="absolute z-10 mt-1 w-full bg-white rounded-md shadow-lg max-h-60 overflow-auto"
            >
              {getFilteredUsers().length > 0 ? (
                getFilteredUsers().map((user) => (
                  <button
                    key={user._id}
                    className="w-full text-left p-3 hover:bg-gray-50 flex items-start gap-3 border-b border-gray-100"
                    onClick={() => handleSelectUser(user._id)}
                  >
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden">
                        <Image src={user.avatar || "/avatars/avatar.jpeg"} alt={user.name} width={32} height={32} />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm text-gray-800">{user.name}</p>
                      <p className="text-xs text-gray-500">@{user.username}</p>
                      {user.isFriend && (
                        <span className="inline-block px-2 py-0.5 text-xs bg-blue-100 text-blue-800 rounded-full mt-1">
                          Friend
                        </span>
                      )}
                    </div>
                  </button>
                ))
              ) : (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No users found
                </div>
              )}
            </div>
          )}
        </div>

        {/* Contacts list with darker selection background */}
        <div className="flex-1 overflow-auto ">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            </div>
          ) : contacts.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500">
              No contacts found
            </div>
          ) : (
            contacts.map((contact) => (
              <button
                key={contact.id}
                className={`w-full text-left p-3 hover:bg-gray-50 flex items-start gap-3 ${
                  selectedContact === contact.id ? "bg-gray-200" : ""
                }`}
                onClick={() => setSelectedContact(contact.id)}
              >
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
                    <Image src={contact.avatar || "/avatars/avatar.jpeg"} alt={contact.name} width={40} height={40} />
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline justify-between">
                    <p className="font-medium text-sm truncate text-gray-800 ">{contact.name}</p>
                  </div>
                  <p className="text-xs text-gray-500">{contact.username}</p>
                </div>
              </button>
            ))
          )}
        </div>

        {/* User profile */}
        <div className="p-3 border-t flex items-center justify-between">
          <Link 
            href="/admin" 
            className="flex items-center gap-3"
            onClick={async (e) => {
              e.preventDefault();
              try {
                const token = localStorage.getItem('token');
                if (!token) {
                  console.error('No token found');
                  return;
                }

                // Update counters before redirecting
                const response = await fetch('http://localhost:5000/api/settings/update-counters', {
                  method: 'POST',
                  headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                  }
                });

                if (!response.ok) {
                  throw new Error('Failed to update counters');
                }

                // Redirect to admin page after successful update
                window.location.href = '/admin';
              } catch (error) {
                console.error('Error updating counters:', error);
                // Still redirect even if counter update fails
                window.location.href = '/admin';
              }
            }}
          >
            <div className="w-12 h-12 relative">
              {isClient && currentUser ? (
                <Image
                  src={currentUser.avatar || "/images/profile.png"}
                  alt="profile"
                  width={40}
                  height={40}
                  className="rounded-lg"
                />
              ) : (
                <div className="w-12 h-12 bg-gray-100 rounded-lg"></div>
              )}
            </div>
            <div>
              <p className="text-sm font-medium">{currentUser?.name || ''}</p>
              <p className="text-xs text-gray-500">@{currentUser?.username || ''}</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Main content */}
      {selectedContact ? (
        <div className="flex-1 flex flex-col">
          {/* Chat header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center">
              <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden mr-3">
                <Image
                  src={allUsers.find(u => u._id === selectedContact)?.avatar || "/placeholder.svg"}
                  alt={allUsers.find(u => u._id === selectedContact)?.name || "Contact"}
                  width={40}
                  height={40}
                />
              </div>
              <div>
                <p className="font-medium text-gray-800">
                  {allUsers.find(u => u._id === selectedContact)?.name}
                </p>
                <p className="text-xs text-gray-500">
                  @{allUsers.find(u => u._id === selectedContact)?.username}
                </p>
              </div>
            </div>
            {/* Add friend button if not already a friend */}
            {allUsers.find(u => u._id === selectedContact && !u.isFriend) && (
              <Button
                onClick={() => handleAddFriend(selectedContact)}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md flex items-center gap-2"
              >
                <UserPlus className="h-4 w-4" />
                Add Friend
              </Button>
            )}
          </div>

          {/* Chat messages */}
          <div className="flex-1 overflow-auto p-4">
            {isMessagesLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.senderId === currentUser?._id ? "justify-end" : "justify-start"}`}
                  >
                    <div className="flex flex-col group">
                      <div className="flex items-center gap-2">
                        {message.senderId !== currentUser?._id && (
                          <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden flex-shrink-0">
                            <Image
                              src={selectedContactData?.avatar || "/avatars/avatar.jpeg"}
                              alt={selectedContactData?.name || "Contact"}
                              width={32}
                              height={32}
                            />
                          </div>
                        )}

                        {/* Status and options for current user's messages */}
                        {message.senderId === currentUser?._id && (
                          <div className="flex items-center gap-2">
                            <button
                              className="text-gray-500 hover:text-gray-700 z-10 options-toggle-button"
                              onClick={(e) => handleMessageOptions(e, message.id)}
                            >
                              <MoreVertical className="h-4 w-4" />
                            </button>
                            {message.userFeedback && (
                              <span className={`text-sm ${message.userFeedback === "Toxic" ? "text-red-500" : "text-green-500"}`}>
                                Marked {message.userFeedback === "Toxic" ? "as Toxic" : "as Not Toxic"}
                              </span>
                            )}
                          </div>
                        )}

                        <div className={`flex-1 flex items-center gap-2 rounded-lg px-4 py-2 ${
                          message.senderId === currentUser?._id ? "bg-blue-200 text-blue-900" : "bg-gray-50 text-gray-900"
                        } ${message.isEmoji ? "text-2xl bg-transparent px-0" : ""}`}>
                          {((message.isToxic && message.userFeedback === null) || message.userFeedback === "Toxic") && message.isHidden ? (
                            <span className="text-yellow-800">Warning: This message contains toxic words</span>
                          ) : (
                            <span>{message.text}</span>
                          )}
                        </div>

                        {/* Status and options for received messages */}
                        {message.senderId !== currentUser?._id && (
                          <div className="flex items-center gap-2 text-sm">
                            {message.userFeedback && (
                              <span className={`${message.userFeedback === "Toxic" ? "text-red-500" : "text-green-500"}`}>
                                Marked {message.userFeedback === "Toxic" ? "as Toxic" : "as Not Toxic"}
                              </span>
                            )}
                            <button
                              className="text-gray-500 hover:text-gray-700 z-10 options-toggle-button"
                              onClick={(e) => handleMessageOptions(e, message.id)}
                            >
                              <MoreVertical className="h-4 w-4" />
                            </button>
                          </div>
                        )}

                        {message.senderId === currentUser?._id && (
                          <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden flex-shrink-0">
                            <Image
                              src={currentUser?.avatar || "/images/profile.png"}
                              alt="You"
                              width={32}
                              height={32}
                            />
                          </div>
                        )}
                      </div>
                      
                      {/* Timestamp below message */}
                      <div className={`flex text-xs text-gray-500 mt-1 ${
                        message.senderId === currentUser?._id ? "justify-end" : "justify-start ml-10"
                      }`}>
                        {message.timestamp}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
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
              {optionMenu.messageId && (
                <>
                  {/* Only show toxicity feedback options if the current user is the receiver */}
                  {messages.find(msg => msg.id === optionMenu.messageId)?.senderId !== currentUser?._id && (
                    <>
                      {/* Case 1: Show "Not a toxic message" when message is toxic with no feedback OR marked as Toxic */}
                      {((messages.find(msg => msg.id === optionMenu.messageId)?.isToxic && 
                         messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === null) || 
                        messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === "Toxic") && (
                        <button
                          className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-green-500"
                          onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, false)}
                        >
                          ... Not a toxic message
                        </button>
                      )}

                      {/* Case 2: Show "Toxic message" when message is not toxic with no feedback OR marked as Not Toxic */}
                      {((messages.find(msg => msg.id === optionMenu.messageId)?.isToxic === false && 
                         messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === null) || 
                        messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === "Not Toxic") && (
                        <button
                          className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-red-500"
                          onClick={() => optionMenu.messageId && handleToxicFeedback(optionMenu.messageId, true)}
                        >
                          ... Toxic message
                        </button>
                      )}
                    </>
                  )}

                  {/* Show Hide/Unhide for toxic messages or messages marked as Toxic - available to both sender and receiver */}
                  {((messages.find(msg => msg.id === optionMenu.messageId)?.isToxic && 
                     messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === null) || 
                    messages.find(msg => msg.id === optionMenu.messageId)?.userFeedback === "Toxic") && (
                    <button
                      className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100 text-blue-500"
                      onClick={() => {
                        if (!optionMenu.messageId) return;
                        setMessages(prevMessages => 
                          prevMessages.map(msg => 
                            msg.id === optionMenu.messageId 
                              ? { ...msg, isHidden: !msg.isHidden }
                              : msg
                          )
                        );
                        setOptionMenu({ visible: false, messageId: null, position: { top: 0, left: 0 } });
                      }}
                    >
                      {messages.find(msg => msg.id === optionMenu.messageId)?.isHidden 
                        ? "Show message" 
                        : "Hide message"}
                    </button>
                  )}

                  {/* Always show Delete option */}
                  <button
                    className="w-full text-left px-4 py-2 text-sm hover:bg-gray-100"
                    onClick={() => optionMenu.messageId && handleDeleteMessage(optionMenu.messageId)}
                  >
                    Delete message
                  </button>
                </>
              )}
            </div>
          )}

          {/* Message input */}
          <div className="border-t bg-white p-4">
            {toxicWarningMessage && (
              <div className="mb-4 flex items-center gap-2 bg-gray-200 p-3 rounded-lg">
                <div className="text-red-500">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <p className="text-gray-700">Warning: This message contains toxic words.</p>
                  <p className="text-gray-600">Do you really want to send it?</p>
                </div>
                <button
                  className="text-blue-600 hover:text-blue-800 font-medium"
                  onClick={handleSendAnyway}
                >
                  Send Anyway
                </button>
              </div>
            )}
            <div className="flex items-center gap-2">
              <button className="p-2 hover:bg-gray-100 rounded-full">
                <ImageIcon className="h-6 w-6 text-gray-500" />
              </button>
              <div className="flex-1">
                <Input
                  placeholder="Input something..."
                  className="border rounded-full focus-visible:ring-1 focus-visible:ring-blue-400 px-4 py-2"
                  value={messageInput}
                  onChange={(e) => {
                    setMessageInput(e.target.value)
                    // Clear toxic warning when user starts typing a new message
                    if (toxicWarningMessage) {
                      setToxicWarningMessage(null)
                      setPendingToxicMessage(null)
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleMessageSubmit()
                    }
                  }}
                />
              </div>
              <button className="p-2 hover:bg-gray-100 rounded-full">
                <Smile className="h-6 w-6 text-gray-500" />
              </button>
              <button 
                className="p-2 hover:bg-blue-50 rounded-full text-blue-500"
                onClick={(e) => {
                  e.preventDefault();
                  handleMessageSubmit();
                }}
              >
                <Send className="h-6 w-6" />
              </button>
            </div>
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