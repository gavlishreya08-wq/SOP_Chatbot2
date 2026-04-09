import { useCallback, useRef, useState } from "react";
import { sendMessage } from "../api/client";
import type { AnswerMode, LlmProvider, Message, SourceInfo, StreamEvent } from "../types";

let idCounter = 0;
function nextId() {
  return `msg-${++idCounter}-${Date.now()}`;
}

function toHistory(messages: Message[]) {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .slice(-6)
    .map((message) => ({ role: message.role, content: message.content }));
}

export function useChat(llmProvider: LlmProvider) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeSop, setActiveSop] = useState<string | null>(null);
  const [sourceLocked, setSourceLocked] = useState(false);
  const [answerMode, setAnswerMode] = useState<AnswerMode>("detailed");
  const streamBufferRef = useRef("");
  const abortRef = useRef<AbortController | null>(null);

  const runRequest = useCallback(
    async (
      text: string,
      historySlice: { role: string; content: string }[],
      assistantId: string,
      activeSopForRequest: string | null
    ) => {
      setIsLoading(true);
      streamBufferRef.current = "";

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await sendMessage(
          {
            message: text,
            history: historySlice,
            active_sop: activeSopForRequest,
            stream: true,
            llm_provider: llmProvider,
            answer_mode: answerMode,
            source_locked: sourceLocked,
          },
          (token) => {
            streamBufferRef.current += token;
            const current = streamBufferRef.current;
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId ? { ...message, content: current } : message
              )
            );
          },
          (event: StreamEvent) => {
            if (event.active_sop !== undefined) {
              setActiveSop(event.active_sop ?? null);
            }
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: event.full_answer || streamBufferRef.current,
                      sources: event.sources as SourceInfo | null | undefined,
                      followup: event.followup,
                      image: event.image,
                      confidence: event.confidence,
                      suggestions: event.suggestions,
                    }
                  : message
              )
            );
            setIsLoading(false);
            abortRef.current = null;
          },
          (error) => {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: `Error: ${error}`,
                      sources: null,
                      followup: null,
                      image: null,
                      confidence: "low",
                      suggestions: [],
                    }
                  : message
              )
            );
            setIsLoading(false);
            abortRef.current = null;
          },
          controller.signal
        );
      } catch {
        if (!controller.signal.aborted) {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: "Failed to connect to the server.",
                    sources: null,
                    followup: null,
                    image: null,
                    confidence: "low",
                    suggestions: [],
                  }
                : message
            )
          );
        }
        setIsLoading(false);
        abortRef.current = null;
      }
    },
    [llmProvider, answerMode, sourceLocked]
  );

  const send = useCallback(
    async (text: string) => {
      const userMsg: Message = { id: nextId(), role: "user", content: text };
      const assistantId = nextId();

      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "" },
      ]);

      const historySlice = toHistory([...messages, userMsg]);
      await runRequest(text, historySlice, assistantId, activeSop);
    },
    [messages, activeSop, runRequest]
  );

  const retryLast = useCallback(async () => {
    if (isLoading) return;

    const lastAssistantIndex = [...messages]
      .map((message, index) => ({ message, index }))
      .reverse()
      .find((entry) => entry.message.role === "assistant")?.index;

    if (lastAssistantIndex === undefined) {
      return;
    }

    let userIndex = -1;
    for (let index = lastAssistantIndex - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "user") {
        userIndex = index;
        break;
      }
    }

    if (userIndex === -1) {
      return;
    }

    const question = messages[userIndex].content;
    const assistantId = messages[lastAssistantIndex].id;
    const historySlice = toHistory(messages.slice(0, lastAssistantIndex));

    setMessages((prev) =>
      prev.map((message, index) =>
        index === lastAssistantIndex
          ? {
              ...message,
              content: "",
              sources: null,
              followup: null,
              image: null,
              confidence: undefined,
              suggestions: [],
            }
          : message
      )
    );

    await runRequest(question, historySlice, assistantId, activeSop);
  }, [messages, activeSop, isLoading, runRequest]);

  const editAndResend = useCallback(
    async (messageId: string, newText: string) => {
      if (isLoading) return;

      const msgIndex = messages.findIndex((m) => m.id === messageId);
      if (msgIndex === -1 || messages[msgIndex].role !== "user") return;

      // Remove this message and everything after it
      const kept = messages.slice(0, msgIndex);
      setMessages(kept);

      // Send the edited message as new
      const userMsg: Message = { id: nextId(), role: "user", content: newText };
      const assistantId = nextId();

      setMessages([
        ...kept,
        userMsg,
        { id: assistantId, role: "assistant", content: "" },
      ]);

      const historySlice = toHistory([...kept, userMsg]);
      await runRequest(newText, historySlice, assistantId, activeSop);
    },
    [messages, activeSop, isLoading, runRequest]
  );

  const stopStreaming = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setMessages([]);
    setActiveSop(null);
    setIsLoading(false);
  }, []);

  const toggleSourceLock = useCallback(() => {
    setSourceLocked((prev) => !prev);
  }, []);

  const setFeedback = useCallback((messageId: string, rating: "up" | "down") => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId
          ? { ...msg, feedback: msg.feedback === rating ? null : rating }
          : msg
      )
    );
  }, []);

  return {
    messages,
    isLoading,
    activeSop,
    sourceLocked,
    answerMode,
    send,
    retryLast,
    editAndResend,
    stopStreaming,
    clearChat,
    toggleSourceLock,
    setAnswerMode,
    setFeedback,
    setMessages,
    setActiveSop,
  };
}
