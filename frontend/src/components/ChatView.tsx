import { Badge, Box, Button, Card, Flex, Heading, IconButton, Text } from "@radix-ui/themes";
import {
  Bot,
  FileText,
  PanelLeftOpen,
  Square,
  Sparkles,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useEffect, useRef } from "react";
import type { AnswerMode, HealthInfo, Message } from "../types";
import ChatInput from "./ChatInput";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: Message[];
  isLoading: boolean;
  onSend: (text: string) => void;
  onRetry: () => void;
  onStop: () => void;
  onEditMessage: (messageId: string, newText: string) => void;
  onFeedback: (messageId: string, rating: "up" | "down") => void;
  onShowMore: (messageId: string) => void;
  modelLabel: string;
  modelName: string;
  activeSop: string | null;
  health: HealthInfo | null;
  sourceLocked: boolean;
  onToggleSourceLock: () => void;
  answerMode: AnswerMode;
  onSetAnswerMode: (mode: AnswerMode) => void;
  onOpenSidebar: () => void;
}

function formatSopName(value: string) {
  return value
    .replace(/\.pdf$/i, "")
    .replace(/[_\-.()&]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function WelcomeScreen({ onSend }: { onSend: (text: string) => void }) {
  const starters = [
    "How do I create a Jira issue?",
    "Explain the change management workflow.",
    "What is the release deployment process?",
    "Summarize the source code management SOP.",
  ];

  return (
    <Box className="welcome-hero">
      <Box className="welcome-grid">
        <Card size="4" className="surface-panel welcome-card">
          <Badge color="cyan" variant="soft" radius="full">
            GEL SOP assistant
          </Badge>

          <Flex align="center" gap="3" mt="4">
            <Box className="brand-mark welcome-mark">
              <Bot size={28} color="white" />
            </Box>
            <Box>
              <Heading size="8" className="gradient-title">
                Ask operational questions with context.
              </Heading>
              <Text size="3" className="muted-copy" mt="3" as="p">
                Search policies, procedures, workflows, and ownership details across your
                SOP library with grounded answers and citations.
              </Text>
            </Box>
          </Flex>

          <Box className="starter-grid" mt="6">
            {starters.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="starter-tile"
                onClick={() => onSend(prompt)}
              >
                <Text size="2" weight="medium">
                  {prompt}
                </Text>
                <Text size="1" className="muted-copy" as="p" mt="2">
                  Launch this prompt directly into the conversation.
                </Text>
              </button>
            ))}
          </Box>
        </Card>

        <Flex direction="column" gap="3">
          <Card size="3" className="surface-panel welcome-card">
            <Text className="section-eyebrow">Capabilities</Text>
            <Box className="feature-list" mt="4">
              <Box className="feature-tile">
                <Flex align="center" gap="2">
                  <Sparkles size={16} className="accent-cyan" />
                  <Text size="2" weight="medium">
                    Grounded answers
                  </Text>
                </Flex>
                <Text size="1" className="muted-copy" mt="2" as="p">
                  Responses include cited SOP references, confidence, and suggested follow-ups.
                </Text>
              </Box>

              <Box className="feature-tile">
                <Flex align="center" gap="2">
                  <FileText size={16} className="accent-gold" />
                  <Text size="2" weight="medium">
                    SOP comparison
                  </Text>
                </Flex>
                <Text size="1" className="muted-copy" mt="2" as="p">
                  Compare two SOPs from the control panel and keep working in the same chat.
                </Text>
              </Box>

              <Box className="feature-tile">
                <Flex align="center" gap="2">
                  <Wifi size={16} className="accent-green" />
                  <Text size="2" weight="medium">
                    Multi-provider routing
                  </Text>
                </Flex>
                <Text size="1" className="muted-copy" mt="2" as="p">
                  Switch between configured model providers without changing backend logic.
                </Text>
              </Box>
            </Box>
          </Card>
        </Flex>
      </Box>
    </Box>
  );
}

function TypingIndicator() {
  return (
    <Card size="2" className="typing-indicator-card">
      <Flex align="center" gap="3">
        <Flex align="center" gap="1">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </Flex>
        <Text size="1" className="muted-copy">
          Building the answer...
        </Text>
      </Flex>
    </Card>
  );
}

function StatusBadges({ health }: { health: HealthInfo | null }) {
  if (!health) {
    return (
      <Badge color="red" variant="soft" radius="full">
        <WifiOff size={12} />
        Offline
      </Badge>
    );
  }

  const providerStatus = health.provider_status ? Object.entries(health.provider_status) : [];
  const visibleProviders = providerStatus.filter(([, info]) => info.configured);

  if (visibleProviders.length === 0) {
    return (
      <Badge color="gray" variant="soft" radius="full">
        Waiting for providers
      </Badge>
    );
  }

  return (
    <Flex gap="2" wrap="wrap">
      {visibleProviders.map(([name, info]) => (
        <Badge
          key={name}
          color={info.healthy ? "green" : "red"}
          variant="soft"
          radius="full"
        >
          {info.healthy ? <Wifi size={12} /> : <WifiOff size={12} />}
          {name}
        </Badge>
      ))}
    </Flex>
  );
}

export default function ChatView({
  messages,
  isLoading,
  onSend,
  onRetry,
  onStop,
  onEditMessage,
  onFeedback,
  onShowMore,
  modelLabel,
  modelName,
  activeSop,
  health,
  sourceLocked,
  onToggleSourceLock,
  answerMode,
  onSetAnswerMode,
  onOpenSidebar,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastAssistantId = [...messages]
    .reverse()
    .find((message) => message.role === "assistant")?.id;
  const activeSopLabel = activeSop ? formatSopName(activeSop) : null;

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  return (
    <Flex direction="column" className="chat-shell">
      <Card size="3" className="surface-panel chat-header-card">
        <Flex justify="between" align="start" gap="4" wrap="wrap" className="chat-header-row">
          <Flex align="start" gap="3" className="chat-header-meta">
            <IconButton
              variant="surface"
              color="gray"
              className="mobile-only"
              onClick={onOpenSidebar}
              aria-label="Open workspace controls"
            >
              <PanelLeftOpen size={18} />
            </IconButton>

            <Box>
              <Text className="section-eyebrow">Active model</Text>
              <Flex align="center" gap="2" wrap="wrap" mt="2">
                <Heading size="5">{modelLabel}</Heading>
                <Badge color="cyan" variant="surface" radius="full">
                  {modelName}
                </Badge>
              </Flex>
              <Box mt="3">
                <StatusBadges health={health} />
              </Box>
            </Box>
          </Flex>

          <Box className="chat-title-lockup">
            <Heading size="7" className="gradient-title">
              Prakriya AI
            </Heading>
            <Text size="2" className="muted-copy" mt="2" as="p">
              SOP search, explanation, and grounded comparison workspace
            </Text>
          </Box>

          <Card variant="ghost" className="active-sop-card">
            <Text className="section-eyebrow">Active SOP</Text>
            {activeSopLabel ? (
              <Flex align="center" gap="2" mt="2">
                <FileText size={16} className="accent-cyan" />
                <Text size="2" weight="medium" className="active-sop-text">
                  {activeSopLabel}
                </Text>
              </Flex>
            ) : (
              <Text size="2" className="muted-copy" mt="2">
                No SOP selected yet.
              </Text>
            )}
          </Card>
        </Flex>
      </Card>

      <Box ref={scrollRef} className="chat-scroll-region">
        {messages.length === 0 ? (
          <WelcomeScreen onSend={onSend} />
        ) : (
          <Flex direction="column" gap="4" className="chat-feed">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onFollowup={onSend}
                canRetry={message.role === "assistant" && message.id === lastAssistantId && !isLoading}
                onRetry={message.role === "assistant" && message.id === lastAssistantId ? onRetry : undefined}
                onEdit={message.role === "user" && !isLoading ? onEditMessage : undefined}
                onFeedback={message.role === "assistant" ? onFeedback : undefined}
                onShowMore={message.role === "assistant" && !isLoading ? onShowMore : undefined}
              />
            ))}

            {isLoading &&
              messages[messages.length - 1]?.role === "assistant" &&
              messages[messages.length - 1]?.content === "" ? (
                <TypingIndicator />
              ) : null}
          </Flex>
        )}
      </Box>

      {isLoading ? (
        <Flex justify="center">
          <Button variant="surface" color="red" size="2" onClick={onStop}>
            <Square size={12} fill="currentColor" />
            Stop generating
          </Button>
        </Flex>
      ) : null}

      <ChatInput
        onSend={onSend}
        isLoading={isLoading}
        sourceLocked={sourceLocked}
        onToggleSourceLock={onToggleSourceLock}
        answerMode={answerMode}
        onSetAnswerMode={onSetAnswerMode}
      />
    </Flex>
  );
}
