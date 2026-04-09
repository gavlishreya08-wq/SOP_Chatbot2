import {
  Badge,
  Box,
  Button,
  Card,
  Dialog,
  Flex,
  Heading,
  IconButton,
  Tabs,
  Text,
  TextField,
} from "@radix-ui/themes";
import {
  AlertTriangle,
  BarChart3,
  MessageSquare,
  RefreshCw,
  Shield,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  adminAnalytics,
  adminDismissFailedQuery,
  adminFailedQueries,
  adminFeedback,
  adminLogin,
  adminRebuild,
  adminStatus,
  adminSync,
} from "../api/client";
import type { AnalyticsSummary, FailedQuery, FeedbackEntry } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  token: string | null;
  onLogin: (token: string) => void;
}

type AdminTab = "operations" | "analytics" | "feedback" | "failed";

function formatTimestamp(value: string | null) {
  if (!value) return "Never";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function AnalyticsPanel({ analytics }: { analytics: AnalyticsSummary | null }) {
  if (!analytics) {
    return (
      <Card size="3" className="surface-panel admin-section-card">
        <Text size="2" className="muted-copy">
          Loading analytics...
        </Text>
      </Card>
    );
  }

  return (
    <Flex direction="column" gap="3">
      <Box className="admin-stats-grid">
        <Card size="3" className="surface-panel admin-stat-card">
          <Text className="section-eyebrow">Queries</Text>
          <Heading size="8" mt="3">
            {analytics.total_queries}
          </Heading>
          <Text size="2" className="muted-copy" mt="2">
            Total requests processed
          </Text>
        </Card>

        <Card size="3" className="surface-panel admin-stat-card">
          <Text className="section-eyebrow">Feedback</Text>
          <Flex align="center" gap="3" mt="3">
            <Flex align="center" gap="1">
              <ThumbsUp size={16} className="accent-green" />
              <Text size="4" weight="medium">
                {analytics.feedback_summary.thumbs_up}
              </Text>
            </Flex>
            <Flex align="center" gap="1">
              <ThumbsDown size={16} className="accent-red" />
              <Text size="4" weight="medium">
                {analytics.feedback_summary.thumbs_down}
              </Text>
            </Flex>
          </Flex>
          <Text size="2" className="muted-copy" mt="2">
            Direct user feedback signals
          </Text>
        </Card>

        <Card size="3" className="surface-panel admin-stat-card">
          <Text className="section-eyebrow">Failures</Text>
          <Heading size="8" mt="3">
            {analytics.failed_query_count}
          </Heading>
          <Text size="2" className="muted-copy" mt="2">
            Logged low-confidence or failed results
          </Text>
        </Card>
      </Box>

      <Card size="3" className="surface-panel admin-section-card">
        <Flex justify="between" align="center" gap="3" wrap="wrap">
          <Box>
            <Text className="section-eyebrow">Confidence</Text>
            <Heading size="4" mt="2">
              Output quality breakdown
            </Heading>
          </Box>
          <Flex gap="2" wrap="wrap">
            <Badge color="green" variant="soft" radius="full">
              High {analytics.confidence_breakdown.high}
            </Badge>
            <Badge color="cyan" variant="soft" radius="full">
              Medium {analytics.confidence_breakdown.medium}
            </Badge>
            <Badge color="amber" variant="soft" radius="full">
              Low {analytics.confidence_breakdown.low}
            </Badge>
          </Flex>
        </Flex>
        <Text size="2" className="muted-copy" mt="3">
          Clarification prompts issued: {analytics.clarification_count}
        </Text>
      </Card>

      {analytics.top_questions.length > 0 ? (
        <Card size="3" className="surface-panel admin-section-card">
          <Text className="section-eyebrow">Top prompts</Text>
          <Heading size="4" mt="2">
            Most asked questions
          </Heading>
          <Flex direction="column" gap="2" mt="4" className="admin-list-scroll">
            {analytics.top_questions.slice(0, 10).map((question, index) => (
              <Box key={`${question.question}-${index}`} className="admin-entry">
                <Flex justify="between" align="start" gap="3">
                  <Text size="2" className="admin-entry-text">
                    {question.question}
                  </Text>
                  <Badge color="cyan" variant="soft" radius="full">
                    {question.count}
                  </Badge>
                </Flex>
              </Box>
            ))}
          </Flex>
        </Card>
      ) : null}

      {analytics.top_sops.length > 0 ? (
        <Card size="3" className="surface-panel admin-section-card">
          <Text className="section-eyebrow">Top sources</Text>
          <Heading size="4" mt="2">
            Most accessed SOPs
          </Heading>
          <Flex direction="column" gap="2" mt="4" className="admin-list-scroll">
            {analytics.top_sops.slice(0, 8).map((sop, index) => (
              <Box key={`${sop.sop}-${index}`} className="admin-entry">
                <Flex justify="between" align="start" gap="3">
                  <Text size="2" className="admin-entry-text">
                    {sop.sop.replace(/\.pdf$/i, "").replace(/[_\-.]+/g, " ")}
                  </Text>
                  <Badge color="gray" variant="surface" radius="full">
                    {sop.count}
                  </Badge>
                </Flex>
              </Box>
            ))}
          </Flex>
        </Card>
      ) : null}
    </Flex>
  );
}

function FeedbackPanel({ feedback }: { feedback: FeedbackEntry[] }) {
  if (feedback.length === 0) {
    return (
      <Card size="3" className="surface-panel admin-section-card">
        <Text size="2" className="muted-copy">
          No feedback yet.
        </Text>
      </Card>
    );
  }

  return (
    <Flex direction="column" gap="2" className="admin-list-scroll">
      {feedback.slice(0, 50).map((entry, index) => (
        <Card
          key={`${entry.timestamp}-${index}`}
          size="3"
          className={`surface-panel admin-entry ${entry.rating === "up" ? "good" : "bad"}`}
        >
          <Flex align="center" gap="2" wrap="wrap">
            <Badge
              color={entry.rating === "up" ? "green" : "red"}
              variant="soft"
              radius="full"
            >
              {entry.rating === "up" ? <ThumbsUp size={12} /> : <ThumbsDown size={12} />}
              {entry.rating}
            </Badge>
            <Text size="1" className="muted-copy">
              {formatTimestamp(entry.timestamp)}
            </Text>
          </Flex>

          <Text size="2" weight="medium" mt="3" as="p">
            Q: {entry.question.slice(0, 140)}
          </Text>
          <Text size="2" className="muted-copy" mt="2" as="p">
            A: {entry.answer.slice(0, 180)}
          </Text>
          {entry.comment ? (
            <Text size="2" mt="2" as="p">
              Note: {entry.comment}
            </Text>
          ) : null}
        </Card>
      ))}
    </Flex>
  );
}

function FailedQueriesPanel({
  queries,
  onDismiss,
}: {
  queries: FailedQuery[];
  onDismiss: (index: number) => void;
}) {
  if (queries.length === 0) {
    return (
      <Card size="3" className="surface-panel admin-section-card">
        <Text size="2" className="muted-copy">
          No failed queries.
        </Text>
      </Card>
    );
  }

  return (
    <Flex direction="column" gap="2" className="admin-list-scroll">
      {queries.slice(0, 50).map((query, index) => (
        <Card
          key={`${query.timestamp}-${index}`}
          size="3"
          className="surface-panel admin-entry warning"
        >
          <Flex justify="between" align="start" gap="3">
            <Box className="admin-entry-body">
              <Flex align="center" gap="2" wrap="wrap">
                <Badge color="amber" variant="soft" radius="full">
                  <AlertTriangle size={12} />
                  {query.confidence}
                </Badge>
                <Text size="1" className="muted-copy">
                  {formatTimestamp(query.timestamp)}
                </Text>
              </Flex>

              <Text size="2" weight="medium" mt="3" as="p">
                {query.question}
              </Text>

              {query.answer ? (
                <Text size="2" className="muted-copy" mt="2" as="p">
                  {query.answer.slice(0, 180)}
                </Text>
              ) : null}
            </Box>

            <IconButton
              variant="ghost"
              color="red"
              aria-label="Dismiss failed query"
              onClick={() => onDismiss(index)}
            >
              <Trash2 size={14} />
            </IconButton>
          </Flex>
        </Card>
      ))}
    </Flex>
  );
}

export default function AdminDialog({ open, onClose, token, onLogin }: Props) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ last_sync: string | null; total_documents: number } | null>(
    null
  );
  const [result, setResult] = useState("");
  const [activeTab, setActiveTab] = useState<AdminTab>("operations");
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([]);
  const [failedQueries, setFailedQueries] = useState<FailedQuery[]>([]);

  useEffect(() => {
    if (!open || !token) return;

    adminStatus(token).then(setStatus).catch(() => {});

    if (activeTab === "analytics") {
      adminAnalytics(token).then(setAnalytics).catch(() => {});
      return;
    }

    if (activeTab === "feedback") {
      adminFeedback(token).then(setFeedback).catch(() => {});
      return;
    }

    if (activeTab === "failed") {
      adminFailedQueries(token).then(setFailedQueries).catch(() => {});
    }
  }, [open, token, activeTab]);

  const handleLogin = async () => {
    setError("");
    setLoading(true);

    try {
      const nextToken = await adminLogin(password);
      onLogin(nextToken);
      setPassword("");
    } catch (value) {
      setError(value instanceof Error ? value.message : "Admin login failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSync = async () => {
    if (!token) return;

    setLoading(true);
    setResult("");
    try {
      const response = await adminSync(token);
      setResult(
        `Sync complete: ${response.new} new, ${response.updated} updated, ${response.unchanged} unchanged`
      );
      adminStatus(token).then(setStatus).catch(() => {});
    } catch {
      setResult("Sync failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const handleRebuild = async () => {
    if (!token) return;

    setLoading(true);
    setResult("");
    try {
      const response = await adminRebuild(token);
      setResult(`Index rebuilt with ${response.chunks} chunks.`);
    } catch {
      setResult("Rebuild failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const handleDismissFailedQuery = async (index: number) => {
    if (!token) return;

    await adminDismissFailedQuery(token, index);
    setFailedQueries((current) => current.filter((_, itemIndex) => itemIndex !== index));
  };

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) onClose();
      }}
    >
      <Dialog.Content className="admin-dialog">
        <Box className="admin-shell">
          <Flex justify="between" align="start" gap="3" className="admin-header">
            <Box>
              <Dialog.Title size="6">Admin console</Dialog.Title>
              <Dialog.Description size="2" className="muted-copy" mt="2">
                Sync SOP content, rebuild retrieval assets, and inspect production signals.
              </Dialog.Description>
            </Box>

            <IconButton
              variant="ghost"
              color="gray"
              aria-label="Close admin console"
              onClick={onClose}
            >
              <X size={18} />
            </IconButton>
          </Flex>

          <Box className="admin-body">
            {!token ? (
              <Card size="4" className="surface-panel admin-login-card">
                <Flex align="center" gap="2">
                  <Shield size={18} className="accent-cyan" />
                  <Heading size="5">Admin authentication</Heading>
                </Flex>

                <Text size="2" className="muted-copy" mt="3" as="p">
                  Enter the admin password to unlock sync operations, analytics, feedback,
                  and failure review.
                </Text>

                <TextField.Root
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      void handleLogin();
                    }
                  }}
                  placeholder="Enter admin password"
                  className="admin-password-field"
                  mt="4"
                />

                {error ? (
                  <Text size="2" mt="3" className="accent-red">
                    {error}
                  </Text>
                ) : null}

                <Button
                  size="3"
                  color="cyan"
                  mt="4"
                  loading={loading}
                  disabled={!password}
                  onClick={() => void handleLogin()}
                >
                  Login
                </Button>
              </Card>
            ) : (
              <Tabs.Root value={activeTab} onValueChange={(value) => setActiveTab(value as AdminTab)}>
                <Tabs.List size="2" className="admin-tabs-list">
                  <Tabs.Trigger value="operations">
                    <RefreshCw size={14} />
                    Operations
                  </Tabs.Trigger>
                  <Tabs.Trigger value="analytics">
                    <BarChart3 size={14} />
                    Analytics
                  </Tabs.Trigger>
                  <Tabs.Trigger value="feedback">
                    <MessageSquare size={14} />
                    Feedback
                  </Tabs.Trigger>
                  <Tabs.Trigger value="failed">
                    <AlertTriangle size={14} />
                    Failed queries
                  </Tabs.Trigger>
                </Tabs.List>

                <Tabs.Content value="operations">
                  <Flex direction="column" gap="3">
                    <Card size="3" className="surface-panel admin-section-card">
                      <Text className="section-eyebrow">Status</Text>
                      <Heading size="4" mt="2">
                        Knowledge base state
                      </Heading>

                      <Flex gap="2" wrap="wrap" mt="4">
                        <Badge color="cyan" variant="soft" radius="full">
                          Last sync {formatTimestamp(status?.last_sync ?? null)}
                        </Badge>
                        <Badge color="gray" variant="surface" radius="full">
                          Documents {status?.total_documents ?? 0}
                        </Badge>
                      </Flex>
                    </Card>

                    <Flex className="admin-actions-row">
                      <Button
                        size="3"
                        color="cyan"
                        loading={loading}
                        onClick={() => void handleSync()}
                      >
                        <RefreshCw size={16} />
                        Sync SOPs
                      </Button>

                      <Button
                        size="3"
                        variant="surface"
                        color="gray"
                        loading={loading}
                        onClick={() => void handleRebuild()}
                      >
                        <RefreshCw size={16} />
                        Rebuild index
                      </Button>
                    </Flex>

                    {result ? (
                      <Box className={`admin-result${result.includes("failed") ? " error" : ""}`}>
                        {result}
                      </Box>
                    ) : null}
                  </Flex>
                </Tabs.Content>

                <Tabs.Content value="analytics">
                  <AnalyticsPanel analytics={analytics} />
                </Tabs.Content>

                <Tabs.Content value="feedback">
                  <FeedbackPanel feedback={feedback} />
                </Tabs.Content>

                <Tabs.Content value="failed">
                  <FailedQueriesPanel
                    queries={failedQueries}
                    onDismiss={(index) => void handleDismissFailedQuery(index)}
                  />
                </Tabs.Content>
              </Tabs.Root>
            )}
          </Box>
        </Box>
      </Dialog.Content>
    </Dialog.Root>
  );
}
