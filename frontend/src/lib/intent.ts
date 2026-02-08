import type { Intent } from "@/lib/types"

const CHAT_PREFIXES = ["ask:", "question:", "explain:", "describe:", "what "]
const MODIFY_PREFIXES = ["edit:", "modify:", "change:", "make:", "add:", "remove:", "delete:", "update:"]
const QUESTION_WORDS = ["what", "why", "how", "where", "when", "which", "who", "is", "are", "does", "do", "can", "could", "would", "should", "tell"]

export function detectIntent(input: string): { intent: Intent; cleanInput: string } {
  const trimmed = input.trim()
  const lower = trimmed.toLowerCase()

  // Check explicit prefixes
  for (const prefix of CHAT_PREFIXES) {
    if (lower.startsWith(prefix)) {
      return {
        intent: "chat",
        cleanInput: trimmed.slice(prefix.length).trim(),
      }
    }
  }

  for (const prefix of MODIFY_PREFIXES) {
    if (lower.startsWith(prefix)) {
      return {
        intent: "modify",
        cleanInput: trimmed.slice(prefix.length).trim(),
      }
    }
  }

  // Question mark at end → chat
  if (trimmed.endsWith("?")) {
    return { intent: "chat", cleanInput: trimmed }
  }

  // Starts with a question word → chat
  const firstWord = lower.split(/\s+/)[0]
  if (QUESTION_WORDS.includes(firstWord)) {
    return { intent: "chat", cleanInput: trimmed }
  }

  // Default → modify
  return { intent: "modify", cleanInput: trimmed }
}
