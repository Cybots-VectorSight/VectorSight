"use client"

import { useReducer, useCallback } from "react"
import type { WorkspaceState, SvgVersion, ChatMessage } from "@/lib/types"
import { generateId } from "@/lib/utils"

type Action =
  | { type: "LOAD_SVG"; svg: string }
  | { type: "ADD_VERSION"; version: SvgVersion }
  | { type: "SET_ACTIVE_VERSION"; id: string }
  | { type: "ADD_MESSAGE"; message: ChatMessage }
  | { type: "UPDATE_MESSAGE"; id: string; updates: Partial<ChatMessage> }
  | { type: "SET_BASELINE"; enabled: boolean }
  | { type: "SET_LOADING"; loading: boolean }

const initialState: WorkspaceState = {
  currentSvg: null,
  versions: [],
  activeVersionId: null,
  messages: [],
  baselineEnabled: false,
  isLoading: false,
}

function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case "LOAD_SVG": {
      const version: SvgVersion = {
        id: generateId(),
        svg: action.svg,
        label: "Original",
        timestamp: Date.now(),
      }
      return {
        ...state,
        currentSvg: action.svg,
        versions: [version],
        activeVersionId: version.id,
        messages: [],
      }
    }
    case "ADD_VERSION":
      return {
        ...state,
        currentSvg: action.version.svg,
        versions: [...state.versions, action.version],
        activeVersionId: action.version.id,
      }
    case "SET_ACTIVE_VERSION": {
      const version = state.versions.find((v) => v.id === action.id)
      return {
        ...state,
        activeVersionId: action.id,
        currentSvg: version?.svg ?? state.currentSvg,
      }
    }
    case "ADD_MESSAGE":
      return {
        ...state,
        messages: [...state.messages, action.message],
      }
    case "UPDATE_MESSAGE":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, ...action.updates } : m
        ),
      }
    case "SET_BASELINE":
      return { ...state, baselineEnabled: action.enabled }
    case "SET_LOADING":
      return { ...state, isLoading: action.loading }
    default:
      return state
  }
}

export function useWorkspace() {
  const [state, dispatch] = useReducer(reducer, initialState)

  const loadSvg = useCallback((svg: string) => {
    dispatch({ type: "LOAD_SVG", svg })
  }, [])

  const addVersion = useCallback((version: SvgVersion) => {
    dispatch({ type: "ADD_VERSION", version })
  }, [])

  const setActiveVersion = useCallback((id: string) => {
    dispatch({ type: "SET_ACTIVE_VERSION", id })
  }, [])

  const addMessage = useCallback((message: ChatMessage) => {
    dispatch({ type: "ADD_MESSAGE", message })
  }, [])

  const updateMessage = useCallback(
    (id: string, updates: Partial<ChatMessage>) => {
      dispatch({ type: "UPDATE_MESSAGE", id, updates })
    },
    []
  )

  const setBaseline = useCallback((enabled: boolean) => {
    dispatch({ type: "SET_BASELINE", enabled })
  }, [])

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: "SET_LOADING", loading })
  }, [])

  return {
    state,
    loadSvg,
    addVersion,
    setActiveVersion,
    addMessage,
    updateMessage,
    setBaseline,
    setLoading,
  }
}
