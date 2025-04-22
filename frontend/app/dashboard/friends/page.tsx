"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent } from "@/components/ui/card"
import { Search, UserPlus, Check, X } from "lucide-react"

interface Friend {
  id: number
  name: string
  username: string
  avatar: string
  online: boolean
  lastActive?: string
}

export default function FriendsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  const friends: Friend[] = [
    { id: 1, name: "Jane Smith", username: "janesmith", avatar: "", online: true },
    { id: 2, name: "Alex Johnson", username: "alexj", avatar: "", online: true },
    { id: 3, name: "Sam Wilson", username: "samwilson", avatar: "", online: false, lastActive: "3h ago" },
    { id: 4, name: "Taylor Brown", username: "taylorbrown", avatar: "", online: false, lastActive: "1d ago" },
    { id: 5, name: "Morgan Lee", username: "morganlee", avatar: "", online: true },
  ]

  const pendingRequests: Friend[] = [
    { id: 6, name: "Chris Davis", username: "chrisdavis", avatar: "", online: false, lastActive: "5h ago" },
    { id: 7, name: "Jordan Taylor", username: "jordant", avatar: "", online: true },
  ]

  const suggestions: Friend[] = [
    { id: 8, name: "Riley Johnson", username: "rileyj", avatar: "", online: true },
    { id: 9, name: "Casey Williams", username: "caseyw", avatar: "", online: false, lastActive: "2d ago" },
    { id: 10, name: "Jamie Garcia", username: "jamieg", avatar: "", online: true },
    { id: 11, name: "Quinn Martinez", username: "quinnm", avatar: "", online: false, lastActive: "1w ago" },
  ]

  const filteredFriends = friends.filter(
    (friend) =>
      friend.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      friend.username.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="container mx-auto p-4 md:p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Friends</h1>
        <p className="text-slate-600 dark:text-slate-400">Manage your connections</p>
      </div>

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500 dark:text-slate-400" />
          <Input
            placeholder="Search friends..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <Tabs defaultValue="all">
        <TabsList className="mb-6">
          <TabsTrigger value="all">All Friends ({friends.length})</TabsTrigger>
          <TabsTrigger value="online">Online ({friends.filter((f) => f.online).length})</TabsTrigger>
          <TabsTrigger value="pending">Pending ({pendingRequests.length})</TabsTrigger>
          <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4">
          {filteredFriends.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredFriends.map((friend) => (
                <Card key={friend.id}>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <div className="relative">
                        <Avatar>
                          <AvatarImage src={friend.avatar || `/placeholder.svg?height=40&width=40`} alt={friend.name} />
                          <AvatarFallback>
                            {friend.name
                              .split(" ")
                              .map((n) => n[0])
                              .join("")}
                          </AvatarFallback>
                        </Avatar>
                        {friend.online && (
                          <span className="absolute bottom-0 right-0 h-3 w-3 rounded-full bg-green-500 border-2 border-white dark:border-slate-950"></span>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium">{friend.name}</p>
                        <p className="text-sm text-slate-600 dark:text-slate-400">@{friend.username}</p>
                        <p className="text-xs mt-1">
                          {friend.online ? (
                            <span className="text-green-600 dark:text-green-400">Online</span>
                          ) : (
                            <span className="text-slate-500 dark:text-slate-400">Last active {friend.lastActive}</span>
                          )}
                        </p>
                      </div>
                      <Button variant="outline" size="sm">
                        Message
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-slate-600 dark:text-slate-400">No friends found matching "{searchQuery}"</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="online" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {friends
              .filter((f) => f.online)
              .map((friend) => (
                <Card key={friend.id}>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <div className="relative">
                        <Avatar>
                          <AvatarImage src={friend.avatar || `/placeholder.svg?height=40&width=40`} alt={friend.name} />
                          <AvatarFallback>
                            {friend.name
                              .split(" ")
                              .map((n) => n[0])
                              .join("")}
                          </AvatarFallback>
                        </Avatar>
                        <span className="absolute bottom-0 right-0 h-3 w-3 rounded-full bg-green-500 border-2 border-white dark:border-slate-950"></span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium">{friend.name}</p>
                        <p className="text-sm text-slate-600 dark:text-slate-400">@{friend.username}</p>
                        <p className="text-xs mt-1 text-green-600 dark:text-green-400">Online</p>
                      </div>
                      <Button variant="outline" size="sm">
                        Message
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
          </div>
        </TabsContent>

        <TabsContent value="pending" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {pendingRequests.map((request) => (
              <Card key={request.id}>
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    <Avatar>
                      <AvatarImage src={request.avatar || `/placeholder.svg?height=40&width=40`} alt={request.name} />
                      <AvatarFallback>
                        {request.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium">{request.name}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-400">@{request.username}</p>
                    </div>
                    <div className="flex gap-2">
                      <Button size="icon" variant="outline" className="h-8 w-8">
                        <Check className="h-4 w-4" />
                        <span className="sr-only">Accept</span>
                      </Button>
                      <Button size="icon" variant="outline" className="h-8 w-8">
                        <X className="h-4 w-4" />
                        <span className="sr-only">Decline</span>
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="suggestions" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {suggestions.map((suggestion) => (
              <Card key={suggestion.id}>
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    <Avatar>
                      <AvatarImage
                        src={suggestion.avatar || `/placeholder.svg?height=40&width=40`}
                        alt={suggestion.name}
                      />
                      <AvatarFallback>
                        {suggestion.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium">{suggestion.name}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-400">@{suggestion.username}</p>
                    </div>
                    <Button size="sm">
                      <UserPlus className="h-4 w-4 mr-2" />
                      Add
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
