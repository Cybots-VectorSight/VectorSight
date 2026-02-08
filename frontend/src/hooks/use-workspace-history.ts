"use client"

import { useState, useEffect, useCallback } from "react"

export interface HistoryEntry {
  id: string
  label: string
  query: string
  svgSnippet: string
  messageCount: number
  timestamp: number
}

const STORAGE_KEY = "vectorsight-history"
const MAX_ENTRIES = 20

function loadHistory(): HistoryEntry[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

export function useWorkspaceHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>([])

  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  const addEntry = useCallback(
    (label: string, svg: string, query?: string, messageCount?: number) => {
      setHistory((prev) => {
        const entry: HistoryEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          label,
          query: query ?? label,
          svgSnippet: svg.slice(0, 500),
          messageCount: messageCount ?? 1,
          timestamp: Date.now(),
        }
        const next = [entry, ...prev].slice(0, MAX_ENTRIES)
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
        return next
      })
    },
    []
  )

  const removeEntry = useCallback((id: string) => {
    setHistory((prev) => {
      const next = prev.filter((e) => e.id !== id)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }, [])

  const clearHistory = useCallback(() => {
    setHistory([])
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  return { history, addEntry, removeEntry, clearHistory }
}
