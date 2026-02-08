import { useQuery, useMutation } from "@tanstack/react-query"
import type {
  HealthResponse,
  AnalyzeResponse,
  ChatResponse,
  ModifyResponse,
  CreateResponse,
} from "@/lib/types"

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: "Request failed" }))
    throw new Error(error.error || error.message || `API error ${res.status}`)
  }
  return res.json()
}

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: () => apiFetch("/health"),
    retry: false,
    refetchInterval: 30000,
  })
}

export function useAnalyze() {
  return useMutation<AnalyzeResponse, Error, { svg: string }>({
    mutationFn: (data) =>
      apiFetch("/analyze", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  })
}

export function useChat() {
  return useMutation<
    ChatResponse,
    Error,
    { svg: string; question: string }
  >({
    mutationFn: (data) =>
      apiFetch("/chat", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  })
}

export function useModify() {
  return useMutation<
    ModifyResponse,
    Error,
    { svg: string; instruction: string }
  >({
    mutationFn: (data) =>
      apiFetch("/modify", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  })
}

export function useCreate() {
  return useMutation<CreateResponse, Error, { description: string }>({
    mutationFn: (data) =>
      apiFetch("/create", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  })
}
