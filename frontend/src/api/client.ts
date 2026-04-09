import type {
  AnalyticsSummary,
  ChatRequest,
  CompareRequest,
  CompareResult,
  Conversation,
  ConversationSummary,
  FailedQuery,
  FeedbackEntry,
  HealthInfo,
  SopEntry,
  StreamEvent,
} from "../types";

const BASE = "";

async function isBackendReachable(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}

async function readErrorText(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text.trim();
  } catch {
    return "";
  }
}

async function getServerErrorMessage(
  response: Response,
  fallback: string
): Promise<string> {
  const errorText = await readErrorText(response);

  if (response.status === 401) {
    return "Incorrect password";
  }

  if (response.status >= 500) {
    const backendReachable = await isBackendReachable();
    if (!backendReachable) {
      return "Backend server is not reachable. Start it with .\\start.ps1 and wait for it to finish loading.";
    }
    return `${fallback} (${response.status}). Check backend logs in .run\\backend\\`;
  }

  if (errorText) {
    return `${fallback}: ${errorText}`;
  }

  return `${fallback}: ${response.status}`;
}

export async function sendMessage(
  request: ChatRequest,
  onToken: (text: string) => void,
  onDone: (event: StreamEvent) => void,
  onError: (error: string) => void,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (err) {
    if (signal?.aborted) return;
    const backendReachable = await isBackendReachable();
    onError(
      backendReachable
        ? "Request failed before the server responded."
        : "Backend server is not reachable. Start it with .\\start.ps1 and wait for it to finish loading."
    );
    return;
  }

  if (!response.ok) {
    onError(await getServerErrorMessage(response, "Server error"));
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError("No response stream");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const json = line.slice(6).trim();
        if (!json) continue;

        try {
          const event: StreamEvent = JSON.parse(json);
          if (event.type === "token") {
            onToken(event.content || "");
          } else if (event.type === "done") {
            onDone(event);
          } else if (event.type === "fallback") {
            onToken(`\n\n*${event.content}*\n\n`);
          } else if (event.type === "error") {
            onError(event.content || "Unknown error");
          }
        } catch {
          // skip malformed events
        }
      }
    }
  } catch (err) {
    if (signal?.aborted) return;
    throw err;
  }
}

// ── Feedback ────────────────────────────────────────────────────────────

export async function submitFeedback(
  question: string,
  answer: string,
  rating: "up" | "down",
  activeSop?: string | null,
  comment?: string
): Promise<void> {
  await fetch(`${BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      answer,
      rating,
      active_sop: activeSop || null,
      comment: comment || "",
    }),
  });
}

// ── Compare ─────────────────────────────────────────────────────────────

export async function compareSops(request: CompareRequest): Promise<CompareResult> {
  const res = await fetch(`${BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error("Compare failed");
  return res.json();
}

// ── SOP list ────────────────────────────────────────────────────────────

export async function listSops(): Promise<SopEntry[]> {
  const res = await fetch(`${BASE}/api/sops`);
  if (!res.ok) return [];
  return res.json();
}

// ── Conversations ───────────────────────────────────────────────────────

export async function saveConversation(
  conversationId: string,
  messages: { role: string; content: string }[],
  title?: string
): Promise<void> {
  await fetch(`${BASE}/api/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      messages,
      title: title || "",
    }),
  });
}

export async function listConversations(limit = 50): Promise<ConversationSummary[]> {
  const res = await fetch(`${BASE}/api/conversations?limit=${limit}`);
  if (!res.ok) return [];
  return res.json();
}

export async function searchConversations(query: string): Promise<ConversationSummary[]> {
  const res = await fetch(`${BASE}/api/conversations/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) return [];
  return res.json();
}

export async function loadConversation(id: string): Promise<Conversation | null> {
  const res = await fetch(`${BASE}/api/conversations/${id}`);
  if (!res.ok) return null;
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${BASE}/api/conversations/${id}`, { method: "DELETE" });
}

// ── Provider status ─────────────────────────────────────────────────────

export async function providerStatus(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/api/status`);
  if (!res.ok) return {};
  return res.json();
}

// ── Admin ───────────────────────────────────────────────────────────────

export async function adminLogin(password: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/admin/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
  } catch {
    const backendReachable = await isBackendReachable();
    throw new Error(
      backendReachable
        ? "Login request failed before the server responded."
        : "Backend server is not reachable. Start it with .\\start.ps1 and wait for it to finish loading."
    );
  }
  if (!res.ok) {
    throw new Error(await getServerErrorMessage(res, "Admin login failed"));
  }
  const data = await res.json();
  return data.token;
}

export async function adminSync(token: string) {
  const res = await fetch(`${BASE}/api/admin/sync`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Sync failed");
  return res.json();
}

export async function adminRebuild(token: string) {
  const res = await fetch(`${BASE}/api/admin/rebuild`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Rebuild failed");
  return res.json();
}

export async function adminStatus(token: string) {
  const res = await fetch(`${BASE}/api/admin/status`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function adminAnalytics(token: string): Promise<AnalyticsSummary> {
  const res = await fetch(`${BASE}/api/admin/analytics`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to fetch analytics");
  return res.json();
}

export async function adminFeedback(token: string, limit = 200): Promise<FeedbackEntry[]> {
  const res = await fetch(`${BASE}/api/admin/feedback?limit=${limit}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) return [];
  return res.json();
}

export async function adminFailedQueries(token: string, limit = 200): Promise<FailedQuery[]> {
  const res = await fetch(`${BASE}/api/admin/failed-queries?limit=${limit}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) return [];
  return res.json();
}

export async function adminDismissFailedQuery(token: string, index: number): Promise<void> {
  await fetch(`${BASE}/api/admin/failed-queries/${index}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function healthCheck(): Promise<HealthInfo> {
  const res = await fetch(`${BASE}/api/health`);
  return res.json();
}
