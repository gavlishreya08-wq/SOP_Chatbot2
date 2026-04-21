export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo | null;
  followup?: string | null;
  image?: string | null;
  confidence?: "high" | "medium" | "low";
  suggestions?: string[];
  feedback?: "up" | "down" | null;
  hasMore?: boolean;
  nextOffset?: number | null;
  activeSop?: string | null;
  originalQuestion?: string | null;
}

export type LlmProvider = "gemini" | "groq";

export type AnswerMode =
  | "brief"
  | "detailed"
  | "checklist"
  | "step-by-step"
  | "only-responsibilities"
  | "only-objective";

export interface ModelOption {
  provider: LlmProvider;
  label: string;
  model: string;
  enabled: boolean;
}

export interface ProviderHealth {
  healthy: boolean;
  configured: boolean;
  model: string | null;
  error_count: number;
  last_error: string | null;
}

export interface HealthInfo {
  status: string;
  llm_provider: LlmProvider;
  model: string;
  available_models: ModelOption[];
  provider_status?: Record<string, ProviderHealth>;
}

export interface SourceInfo {
  title: string;
  filename: string;
  link: string | null;
  version: string;
  created_date: string;
  pages?: string[];
  citations?: CitationInfo[];
}

export interface CitationInfo {
  page?: string | null;
  section?: string | null;
}

export interface ChatRequest {
  message: string;
  history: { role: string; content: string }[];
  active_sop: string | null;
  stream: boolean;
  llm_provider?: LlmProvider;
  answer_mode?: AnswerMode;
  source_locked?: boolean;
  cursor_offset?: number;
  page_limit?: number;
}

export interface StreamEvent {
  type: "token" | "done" | "error" | "fallback";
  content?: string;
  sources?: SourceInfo | null;
  followup?: string | null;
  active_sop?: string | null;
  image?: string | null;
  full_answer?: string;
  confidence?: "high" | "medium" | "low";
  suggestions?: string[];
  has_more?: boolean;
  next_offset?: number | null;
}

export interface SopEntry {
  source: string;
  title: string;
}

export interface CompareRequest {
  question: string;
  sop_a: string;
  sop_b: string;
}

export interface CompareResult {
  answer: string;
  sources: SourceInfo | null;
  sources_b: SourceInfo | null;
  sop_a_title: string;
  sop_b_title: string;
  confidence: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: { role: string; content: string }[];
  created_at: string;
  updated_at: string;
}

export interface AnalyticsSummary {
  total_queries: number;
  confidence_breakdown: { high: number; medium: number; low: number };
  clarification_count: number;
  top_questions: { question: string; count: number }[];
  top_sops: { sop: string; count: number }[];
  feedback_summary: { total: number; thumbs_up: number; thumbs_down: number };
  failed_query_count: number;
}

export interface FailedQuery {
  timestamp: string;
  question: string;
  answer: string;
  confidence: string;
  active_sop: string | null;
}

export interface FeedbackEntry {
  timestamp: string;
  question: string;
  answer: string;
  rating: "up" | "down";
  active_sop: string | null;
  comment: string;
}
