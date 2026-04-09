import {
  Badge,
  Box,
  Button,
  Card,
  Flex,
  IconButton,
  Text,
  TextArea,
} from "@radix-ui/themes";
import {
  Bot,
  Check,
  Copy,
  Edit3,
  Lightbulb,
  RefreshCcw,
  ThumbsDown,
  ThumbsUp,
  User,
} from "lucide-react";
import { useState } from "react";
import { submitFeedback } from "../api/client";
import type { Message } from "../types";
import SourceCard from "./SourceCard";

interface Props {
  message: Message;
  onFollowup: (text: string) => void;
  onRetry?: () => void;
  canRetry?: boolean;
  onEdit?: (messageId: string, newText: string) => void;
  onFeedback?: (messageId: string, rating: "up" | "down") => void;
}

function formatContent(text: string): string {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  const lines = html.split("\n");
  const output: string[] = [];
  let inUnorderedList = false;
  let inOrderedList = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      if (inUnorderedList) {
        output.push("</ul>");
        inUnorderedList = false;
      }
      if (inOrderedList) {
        output.push("</ol>");
        inOrderedList = false;
      }
      continue;
    }

    const checkboxMatch = trimmed.match(/^-\s*\[([ xX])\]\s+(.+)/);
    if (checkboxMatch) {
      if (inOrderedList) {
        output.push("</ol>");
        inOrderedList = false;
      }
      if (!inUnorderedList) {
        output.push('<ul class="checklist">');
        inUnorderedList = true;
      }
      const checked = checkboxMatch[1] !== " ";
      output.push(
        `<li class="check-item">${checked ? "&#9745;" : "&#9744;"} ${checkboxMatch[2]}</li>`
      );
      continue;
    }

    const bulletMatch = trimmed.match(/^[\u2022\-*]\s+(.+)/);
    const numberMatch = trimmed.match(/^\d+[.)]\s+(.+)/);

    if (bulletMatch) {
      if (inOrderedList) {
        output.push("</ol>");
        inOrderedList = false;
      }
      if (!inUnorderedList) {
        output.push("<ul>");
        inUnorderedList = true;
      }
      output.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }

    if (numberMatch) {
      if (inUnorderedList) {
        output.push("</ul>");
        inUnorderedList = false;
      }
      if (!inOrderedList) {
        output.push("<ol>");
        inOrderedList = true;
      }
      output.push(`<li>${numberMatch[1]}</li>`);
      continue;
    }

    if (inUnorderedList) {
      output.push("</ul>");
      inUnorderedList = false;
    }
    if (inOrderedList) {
      output.push("</ol>");
      inOrderedList = false;
    }
    output.push(`<p>${trimmed}</p>`);
  }

  if (inUnorderedList) output.push("</ul>");
  if (inOrderedList) output.push("</ol>");

  return output.join("");
}

function confidenceColor(value: Message["confidence"]) {
  switch (value) {
    case "high":
      return "green";
    case "medium":
      return "cyan";
    case "low":
      return "amber";
    default:
      return "gray";
  }
}

function confidenceLabel(value: Message["confidence"]) {
  switch (value) {
    case "high":
      return "High confidence";
    case "medium":
      return "Medium confidence";
    case "low":
      return "Low confidence";
    default:
      return "";
  }
}

export default function MessageBubble({
  message,
  onFollowup,
  onRetry,
  canRetry = false,
  onEdit,
  onFeedback,
}: Props) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyCitation = async () => {
    if (!message.sources) return;

    const citations = (message.sources.citations || [])
      .map(
        (citation) =>
          `Page ${citation.page || "N/A"}${
            citation.section ? ` - ${citation.section}` : ""
          }`
      )
      .join("; ");

    await navigator.clipboard.writeText(
      `Source: ${message.sources.title} (v${message.sources.version}) | ${citations}`
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (rating: "up" | "down") => {
    if (!onFeedback) return;

    onFeedback(message.id, rating);
    submitFeedback("", message.content, rating).catch(() => {});
  };

  const handleEditSubmit = () => {
    if (!onEdit || !editText.trim()) return;
    onEdit(message.id, editText.trim());
    setEditing(false);
  };

  return (
    <Flex direction="column" align={isUser ? "end" : "start"} gap="2" className="message-stack">
      <Flex
        gap="3"
        align="start"
        direction={isUser ? "row-reverse" : "row"}
        className="message-row"
      >
        <Box className={`message-avatar ${isUser ? "user" : "assistant"}`}>
          {isUser ? <User size={18} color="white" /> : <Bot size={18} color="#67e8f9" />}
        </Box>

        <Card
          size="3"
          className={`surface-panel message-bubble ${
            isUser ? "message-user" : "message-assistant"
          }`}
        >
          <Flex justify="between" align="center" gap="3" wrap="wrap">
            <Text size="1" className="message-label">
              {isUser ? "You" : "Prakriya AI"}
            </Text>

            {!isUser ? (
              <Flex gap="2" wrap="wrap">
                {message.confidence ? (
                  <Badge
                    color={confidenceColor(message.confidence)}
                    variant="soft"
                    radius="full"
                  >
                    {confidenceLabel(message.confidence)}
                  </Badge>
                ) : null}

                {message.sources ? (
                  <Badge color="cyan" variant="surface" radius="full">
                    Cited answer
                  </Badge>
                ) : null}
              </Flex>
            ) : null}
          </Flex>

          <Box mt="3">
            {isUser ? (
              editing ? (
                <Flex direction="column" gap="3">
                  <TextArea
                    rows={3}
                    value={editText}
                    onChange={(event) => setEditText(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        handleEditSubmit();
                      }
                    }}
                    autoFocus
                    className="message-edit-area"
                  />

                  <Flex gap="2" wrap="wrap">
                    <Button color="cyan" onClick={handleEditSubmit}>
                      Send update
                    </Button>
                    <Button
                      variant="ghost"
                      color="gray"
                      onClick={() => {
                        setEditing(false);
                        setEditText(message.content);
                      }}
                    >
                      Cancel
                    </Button>
                  </Flex>
                </Flex>
              ) : (
                <Text size="3" className="bubble-user-text">
                  {message.content}
                </Text>
              )
            ) : (
              <div
                className="msg-content rich-answer"
                dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
              />
            )}
          </Box>
        </Card>
      </Flex>

      {isUser && onEdit && !editing ? (
        <Flex className="message-tools-inline">
          <Button variant="ghost" color="gray" size="1" onClick={() => setEditing(true)}>
            <Edit3 size={13} />
            Edit prompt
          </Button>
        </Flex>
      ) : null}

      {!isUser && message.content ? (
        <Flex gap="2" wrap="wrap" className="message-toolbar">
          <Button variant="ghost" color="gray" size="1" onClick={() => void handleCopy()}>
            {copied ? <Check size={12} /> : <Copy size={12} />}
            {copied ? "Copied" : "Copy"}
          </Button>

          {message.sources ? (
            <Button variant="ghost" color="gray" size="1" onClick={() => void handleCopyCitation()}>
              <Copy size={12} />
              Citation
            </Button>
          ) : null}

          {onFeedback ? (
            <>
              <IconButton
                variant={message.feedback === "up" ? "soft" : "ghost"}
                color={message.feedback === "up" ? "green" : "gray"}
                size="1"
                aria-label="Helpful"
                onClick={() => handleFeedback("up")}
              >
                <ThumbsUp size={13} />
              </IconButton>
              <IconButton
                variant={message.feedback === "down" ? "soft" : "ghost"}
                color={message.feedback === "down" ? "red" : "gray"}
                size="1"
                aria-label="Not helpful"
                onClick={() => handleFeedback("down")}
              >
                <ThumbsDown size={13} />
              </IconButton>
            </>
          ) : null}

          {canRetry && onRetry ? (
            <Button variant="ghost" color="gray" size="1" onClick={onRetry}>
              <RefreshCcw size={12} />
              Retry
            </Button>
          ) : null}
        </Flex>
      ) : null}

      {message.image ? (
        <Box className="message-image-wrap">
          <img
            src={`/static/flowcharts/${message.image.split(/[\\/]/).pop()}`}
            alt="Flowchart"
            className="message-image"
          />
        </Box>
      ) : null}

      {message.sources ? (
        <Box className="message-support">
          <SourceCard source={message.sources} />
        </Box>
      ) : null}

      {message.suggestions && message.suggestions.length > 0 ? (
        <Flex gap="2" wrap="wrap" className="suggestions-row">
          {message.suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="suggestion-pill"
              onClick={() => onFollowup(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </Flex>
      ) : null}

      {message.followup ? (
        <button
          type="button"
          className="followup-card"
          onClick={() => onFollowup(message.followup!)}
        >
          <Flex align="center" gap="2">
            <Lightbulb size={16} className="accent-gold" />
            <Text size="2">{message.followup}</Text>
          </Flex>
        </button>
      ) : null}
    </Flex>
  );
}
