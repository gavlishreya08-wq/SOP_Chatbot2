import {
  Badge,
  Box,
  Button,
  Card,
  Flex,
  Heading,
  IconButton,
  Select,
  Text,
  TextArea,
  TextField,
} from "@radix-ui/themes";
import {
  ArrowLeftRight,
  Download,
  History,
  LogOut,
  MessageSquare,
  Search,
  Shield,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  compareSops,
  deleteConversation,
  listConversations,
  loadConversation,
  searchConversations,
} from "../api/client";
import type {
  CompareResult,
  ConversationSummary,
  LlmProvider,
  Message,
  ModelOption,
  SopEntry,
} from "../types";
import AdminDialog from "./AdminDialog";

interface Props {
  messages: Message[];
  onClearChat: () => void;
  onAskAgain: (text: string) => void;
  adminToken: string | null;
  onAdminLogin: (token: string) => void;
  onAdminLogout: () => void;
  modelOptions: ModelOption[];
  selectedProvider: LlmProvider;
  onSelectProvider: (provider: LlmProvider) => void;
  activeSop: string | null;
  sopList: SopEntry[];
  onRequestClose?: () => void;
}

function getRecentQuestions(messages: Message[]): string[] {
  const questions: string[] = [];
  const skip = new Set([
    "yes",
    "no",
    "ok",
    "okay",
    "sure",
    "y",
    "n",
    "hi",
    "hello",
    "bye",
    "thanks",
  ]);

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user" && !skip.has(message.content.toLowerCase().trim())) {
      if (!questions.includes(message.content)) {
        questions.push(message.content);
      }
      if (questions.length >= 6) break;
    }
  }

  return questions;
}

function formatSopName(value: string) {
  return value
    .replace(/\.pdf$/i, "")
    .replace(/[_\-.()&]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatTimestamp(value: string) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function downloadChat(messages: Message[]) {
  const lines = [
    "=".repeat(50),
    "  SOP CHATBOT - CONVERSATION HISTORY",
    `  Downloaded: ${new Date().toLocaleString()}`,
    "=".repeat(50),
    "",
  ];

  for (const message of messages) {
    if (message.role === "user") {
      lines.push(`YOU:  ${message.content}\n`);
      continue;
    }

    lines.push(`BOT:  ${message.content}\n`);
    if (message.sources) {
      lines.push(`  Source: ${message.sources.title} (v${message.sources.version})`);
      if (message.sources.citations?.length) {
        const citations = message.sources.citations
          .map((citation) =>
            `Page ${citation.page || "N/A"}${
              citation.section ? ` - ${citation.section}` : ""
            }`
          )
          .join("; ");
        lines.push(`  Citations: ${citations}`);
      }
      lines.push("");
    }
  }

  lines.push("=".repeat(50));
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sop_chat_${Date.now()}.txt`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function Sidebar({
  messages,
  onClearChat,
  onAskAgain,
  adminToken,
  onAdminLogin,
  onAdminLogout,
  modelOptions,
  selectedProvider,
  onSelectProvider,
  activeSop,
  sopList,
  onRequestClose,
}: Props) {
  const [showAdmin, setShowAdmin] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [historySearch, setHistorySearch] = useState("");
  const [compareQuestion, setCompareQuestion] = useState("");
  const [compareSopA, setCompareSopA] = useState("");
  const [compareSopB, setCompareSopB] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const recentQuestions = getRecentQuestions(messages);

  useEffect(() => {
    if (!showHistory) return;

    listConversations()
      .then(setConversations)
      .catch(() => setConversations([]));
  }, [showHistory]);

  const handleHistorySearch = async () => {
    try {
      if (historySearch.trim()) {
        const results = await searchConversations(historySearch);
        setConversations(results);
      } else {
        const all = await listConversations();
        setConversations(all);
      }
    } catch {
      setConversations([]);
    }
  };

  const handleLoadConversation = async (id: string) => {
    const conversation = await loadConversation(id);
    if (conversation && conversation.messages.length > 0) {
      onAskAgain(conversation.messages[0].content);
      setShowHistory(false);
      onRequestClose?.();
    }
  };

  const handleDeleteConversation = async (id: string) => {
    await deleteConversation(id);
    setConversations((current) => current.filter((conversation) => conversation.id !== id));
  };

  const handleCompare = async () => {
    if (!compareQuestion.trim() || !compareSopA || !compareSopB) return;

    setCompareLoading(true);
    try {
      const result = await compareSops({
        question: compareQuestion,
        sop_a: compareSopA,
        sop_b: compareSopB,
      });
      setCompareResult(result);
    } catch {
      setCompareResult(null);
    } finally {
      setCompareLoading(false);
    }
  };

  const handleClearChat = () => {
    onClearChat();
    onRequestClose?.();
  };

  const handleAskAgain = (text: string) => {
    onAskAgain(text);
    onRequestClose?.();
  };

  return (
    <>
      <Flex direction="column" className="sidebar-shell">
        <Card size="3" className="surface-panel brand-card">
          <Flex justify="between" align="start" gap="4">
            <Flex gap="3" align="start">
              <Box className="brand-mark">
                <MessageSquare size={24} color="white" />
              </Box>
              <Box>
                <Flex align="center" gap="2" wrap="wrap">
                  <Heading size="6">Prakriya AI</Heading>
                  <Badge color="cyan" variant="surface" radius="full">
                    SOP cockpit
                  </Badge>
                </Flex>
                <Text size="2" className="muted-copy" mt="2" as="p">
                  Search procedures, compare SOPs, and manage the knowledge base from a
                  single workspace.
                </Text>
              </Box>
            </Flex>

            {onRequestClose ? (
              <IconButton
                className="mobile-close-btn"
                variant="ghost"
                color="gray"
                onClick={onRequestClose}
                aria-label="Close controls"
              >
                <X size={18} />
              </IconButton>
            ) : null}
          </Flex>

          <Flex gap="2" wrap="wrap" mt="4">
            <Badge color={activeSop ? "amber" : "gray"} variant="surface" radius="full">
              {activeSop ? formatSopName(activeSop) : "No active SOP"}
            </Badge>
          </Flex>
        </Card>

        <Box className="sidebar-scroller">
          <Flex direction="column" gap="3">
            <Card size="3" className="surface-panel sidebar-card">
              <Text className="section-eyebrow">Actions</Text>
              <Heading size="4" mt="2">
                Conversation tools
              </Heading>

              <Flex direction="column" gap="2" mt="4">
                <Button
                  variant="surface"
                  color="gray"
                  size="3"
                  className="sidebar-action-button"
                  onClick={handleClearChat}
                >
                  <Trash2 size={16} />
                  <Flex direction="column" align="start" gap="1" className="sidebar-action-copy">
                    <Text size="2" weight="medium">
                      Clear chat
                    </Text>
                    <Text size="1" className="muted-copy">
                      Reset the current conversation thread.
                    </Text>
                  </Flex>
                </Button>

                <Button
                  variant="surface"
                  color="gray"
                  size="3"
                  className="sidebar-action-button"
                  disabled={messages.length === 0}
                  onClick={() => messages.length > 0 && downloadChat(messages)}
                >
                  <Download size={16} />
                  <Flex direction="column" align="start" gap="1" className="sidebar-action-copy">
                    <Text size="2" weight="medium">
                      Export chat
                    </Text>
                    <Text size="1" className="muted-copy">
                      Download the current transcript with citations.
                    </Text>
                  </Flex>
                </Button>

                <Button
                  variant={showHistory ? "soft" : "surface"}
                  color={showHistory ? "cyan" : "gray"}
                  size="3"
                  className="sidebar-action-button"
                  onClick={() => setShowHistory((current) => !current)}
                >
                  <History size={16} />
                  <Flex direction="column" align="start" gap="1" className="sidebar-action-copy">
                    <Text size="2" weight="medium">
                      Chat history
                    </Text>
                    <Text size="1" className="muted-copy">
                      Search, reload, or delete previous conversations.
                    </Text>
                  </Flex>
                </Button>

                <Button
                  variant={showCompare ? "soft" : "surface"}
                  color={showCompare ? "amber" : "gray"}
                  size="3"
                  className="sidebar-action-button"
                  onClick={() => setShowCompare((current) => !current)}
                >
                  <ArrowLeftRight size={16} />
                  <Flex direction="column" align="start" gap="1" className="sidebar-action-copy">
                    <Text size="2" weight="medium">
                      Compare SOPs
                    </Text>
                    <Text size="1" className="muted-copy">
                      Run side-by-side SOP analysis from the same panel.
                    </Text>
                  </Flex>
                </Button>
              </Flex>
            </Card>

            {showHistory ? (
              <Card size="3" className="surface-panel sidebar-card">
                <Flex justify="between" align="center" gap="3">
                  <Box>
                    <Text className="section-eyebrow">History</Text>
                    <Heading size="4" mt="2">
                      Previous chats
                    </Heading>
                  </Box>
                  <Badge color="cyan" variant="soft" radius="full">
                    {conversations.length}
                  </Badge>
                </Flex>

                <Flex gap="2" mt="4">
                  <TextField.Root
                    value={historySearch}
                    onChange={(event) => setHistorySearch(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        void handleHistorySearch();
                      }
                    }}
                    placeholder="Search saved chats"
                    className="sidebar-text-field"
                  />
                  <IconButton
                    variant="surface"
                    color="cyan"
                    onClick={() => void handleHistorySearch()}
                    aria-label="Search conversations"
                  >
                    <Search size={16} />
                  </IconButton>
                </Flex>

                <Box className="history-list" mt="4">
                  {conversations.length === 0 ? (
                    <Text size="2" className="muted-copy">
                      No saved chats yet.
                    </Text>
                  ) : (
                    <Flex direction="column" gap="2">
                      {conversations.map((conversation) => (
                        <Box key={conversation.id} className="history-row">
                          <button
                            type="button"
                            className="history-item"
                            onClick={() => void handleLoadConversation(conversation.id)}
                            title={conversation.title}
                          >
                            <Text size="2" weight="medium" className="history-item-title">
                              {conversation.title}
                            </Text>
                            <Text size="1" className="muted-copy">
                              {conversation.message_count} messages
                            </Text>
                            <Text size="1" className="muted-copy">
                              Updated {formatTimestamp(conversation.updated_at)}
                            </Text>
                          </button>

                          <IconButton
                            variant="ghost"
                            color="red"
                            aria-label={`Delete ${conversation.title}`}
                            onClick={() => void handleDeleteConversation(conversation.id)}
                          >
                            <Trash2 size={14} />
                          </IconButton>
                        </Box>
                      ))}
                    </Flex>
                  )}
                </Box>
              </Card>
            ) : null}

            <Card size="3" className="surface-panel sidebar-card">
              <Flex justify="between" align="center" gap="3">
                <Box>
                  <Text className="section-eyebrow">Routing</Text>
                  <Heading size="4" mt="2">
                    Model provider
                  </Heading>
                </Box>
                <Badge color="cyan" variant="soft" radius="full">
                  {selectedProvider}
                </Badge>
              </Flex>

              <Flex direction="column" gap="2" mt="4">
                {modelOptions.map((option) => {
                  const active = option.provider === selectedProvider;
                  const disabled = !option.enabled;

                  return (
                    <button
                      key={option.provider}
                      type="button"
                      className={`provider-tile${active ? " is-active" : ""}`}
                      onClick={() => !disabled && onSelectProvider(option.provider)}
                      disabled={disabled}
                      title={
                        disabled
                          ? `${option.label} is not configured on the backend`
                          : option.model
                      }
                    >
                      <Flex justify="between" align="start" gap="3">
                        <Flex align="center" gap="2">
                          <Sparkles size={15} />
                          <Text size="2" weight="medium">
                            {option.label}
                          </Text>
                        </Flex>

                        <Badge
                          color={disabled ? "gray" : active ? "cyan" : "green"}
                          variant={active ? "solid" : "soft"}
                          radius="full"
                        >
                          {disabled ? "Unavailable" : active ? "Active" : "Ready"}
                        </Badge>
                      </Flex>

                      <Text size="1" className="muted-copy" mt="2" as="p">
                        {option.model}
                      </Text>
                    </button>
                  );
                })}
              </Flex>
            </Card>

            {showCompare ? (
              <Card size="3" className="surface-panel sidebar-card">
                <Flex justify="between" align="center" gap="3">
                  <Box>
                    <Text className="section-eyebrow">Compare</Text>
                    <Heading size="4" mt="2">
                      SOP analysis
                    </Heading>
                  </Box>
                  <Badge color="amber" variant="soft" radius="full">
                    Insight mode
                  </Badge>
                </Flex>

                <Flex direction="column" gap="3" mt="4">
                  <Select.Root
                    value={compareSopA || undefined}
                    onValueChange={setCompareSopA}
                  >
                    <Select.Trigger
                      className="sidebar-select-trigger"
                      placeholder="Select SOP A"
                    />
                    <Select.Content>
                      {sopList.map((sop) => (
                        <Select.Item key={sop.source} value={sop.source}>
                          {sop.title}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>

                  <Select.Root
                    value={compareSopB || undefined}
                    onValueChange={setCompareSopB}
                  >
                    <Select.Trigger
                      className="sidebar-select-trigger"
                      placeholder="Select SOP B"
                    />
                    <Select.Content>
                      {sopList.map((sop) => (
                        <Select.Item key={sop.source} value={sop.source}>
                          {sop.title}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>

                  <TextArea
                    rows={4}
                    value={compareQuestion}
                    onChange={(event) => setCompareQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                        event.preventDefault();
                        void handleCompare();
                      }
                    }}
                    placeholder="What should the comparison focus on?"
                    className="sidebar-text-area"
                  />

                  <Button
                    variant="solid"
                    color="amber"
                    size="3"
                    onClick={() => void handleCompare()}
                    disabled={!compareSopA || !compareSopB || !compareQuestion.trim()}
                    loading={compareLoading}
                  >
                    Compare SOPs
                  </Button>
                </Flex>

                {compareResult ? (
                  <Box className="compare-result-panel">
                    <Flex gap="2" wrap="wrap">
                      <Badge color="cyan" variant="surface" radius="full">
                        {compareResult.sop_a_title}
                      </Badge>
                      <Badge color="amber" variant="surface" radius="full">
                        {compareResult.sop_b_title}
                      </Badge>
                      <Badge color="gray" variant="soft" radius="full">
                        {compareResult.confidence}
                      </Badge>
                    </Flex>

                    <Text size="2" className="compare-result-text" mt="3" as="p">
                      {compareResult.answer}
                    </Text>
                  </Box>
                ) : null}
              </Card>
            ) : null}

            <Card size="3" className="surface-panel sidebar-card">
              <Text className="section-eyebrow">Recall</Text>
              <Heading size="4" mt="2">
                Recent questions
              </Heading>

              <Box mt="4">
                {recentQuestions.length > 0 ? (
                  <Flex direction="column" gap="2">
                    {recentQuestions.map((question) => (
                      <Button
                        key={question}
                        variant="surface"
                        color="gray"
                        size="2"
                        className="question-chip"
                        onClick={() => handleAskAgain(question)}
                      >
                        {question}
                      </Button>
                    ))}
                  </Flex>
                ) : (
                  <Text size="2" className="muted-copy">
                    Your recent prompts will appear here for quick reruns.
                  </Text>
                )}
              </Box>
            </Card>

            <Card size="3" className="surface-panel sidebar-card">
              <Flex justify="between" align="center" gap="3">
                <Box>
                  <Text className="section-eyebrow">Admin</Text>
                  <Heading size="4" mt="2">
                    Knowledge operations
                  </Heading>
                </Box>
                <Badge
                  color={adminToken ? "green" : "gray"}
                  variant="soft"
                  radius="full"
                >
                  {adminToken ? "Signed in" : "Restricted"}
                </Badge>
              </Flex>

              <Text size="2" className="muted-copy" mt="3" as="p">
                Sync documents, rebuild the vector index, and inspect analytics without
                leaving the chat workspace.
              </Text>

              <Flex direction="column" gap="2" mt="4">
                <Button
                  variant="solid"
                  color={adminToken ? "green" : "cyan"}
                  size="3"
                  onClick={() => setShowAdmin(true)}
                >
                  <Shield size={16} />
                  {adminToken ? "Open admin console" : "Admin login"}
                </Button>

                {adminToken ? (
                  <Button
                    variant="ghost"
                    color="gray"
                    size="3"
                    onClick={onAdminLogout}
                  >
                    <LogOut size={16} />
                    Logout
                  </Button>
                ) : null}
              </Flex>
            </Card>
          </Flex>
        </Box>
      </Flex>

      <AdminDialog
        open={showAdmin}
        onClose={() => setShowAdmin(false)}
        token={adminToken}
        onLogin={onAdminLogin}
      />
    </>
  );
}
