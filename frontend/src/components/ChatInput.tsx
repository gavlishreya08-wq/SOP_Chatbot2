import {
  Badge,
  Box,
  Button,
  Card,
  Flex,
  Select,
  Switch,
  Text,
  TextArea,
} from "@radix-ui/themes";
import { Lock, SendHorizontal, Unlock } from "lucide-react";
import { useRef, useState } from "react";
import type { AnswerMode } from "../types";

interface Props {
  onSend: (text: string) => void;
  isLoading: boolean;
  sourceLocked: boolean;
  onToggleSourceLock: () => void;
  answerMode: AnswerMode;
  onSetAnswerMode: (mode: AnswerMode) => void;
}

const ANSWER_MODES: { value: AnswerMode; label: string }[] = [
  { value: "brief", label: "Brief" },
  { value: "detailed", label: "Detailed" },
  { value: "checklist", label: "Checklist" },
  { value: "step-by-step", label: "Step-by-step" },
  { value: "only-responsibilities", label: "Responsibilities only" },
  { value: "only-objective", label: "Objective only" },
];

export default function ChatInput({
  onSend,
  isLoading,
  sourceLocked,
  onToggleSourceLock,
  answerMode,
  onSetAnswerMode,
}: Props) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const text = value.trim();
    if (!text || isLoading) return;
    onSend(text);
    setValue("");
    inputRef.current?.focus();
  };

  return (
    <Box className="composer-wrap">
      <Card size="3" className="surface-panel composer-card">
        <Flex justify="between" align="center" gap="3" wrap="wrap" className="composer-toolbar">
          <Flex align="center" gap="2" wrap="wrap" className="composer-settings">
            <Badge color="cyan" variant="soft" radius="full">
              Response settings
            </Badge>

            <Box className="composer-setting-control">
              <Text size="1" className="composer-setting-label">
                Answer mode
              </Text>
              <Select.Root value={answerMode} onValueChange={(value) => onSetAnswerMode(value as AnswerMode)}>
                <Select.Trigger className="composer-select" />
                <Select.Content>
                  {ANSWER_MODES.map((mode) => (
                    <Select.Item key={mode.value} value={mode.value}>
                      {mode.label}
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Root>
            </Box>

            <Flex align="center" gap="2" className="composer-lock-control">
              {sourceLocked ? (
                <Lock size={15} className="accent-cyan" />
              ) : (
                <Unlock size={15} className="muted-icon" />
              )}
              <Box>
                <Text size="1" className="composer-setting-label">
                  Source lock
                </Text>
                <Text size="1" className="muted-copy">
                  {sourceLocked ? "Pinned to active SOP" : "Auto switching enabled"}
                </Text>
              </Box>
              <Switch checked={sourceLocked} onCheckedChange={() => onToggleSourceLock()} color="cyan" />
            </Flex>
          </Flex>

          <Text size="1" className="muted-copy">
            {isLoading ? "Generating response..." : "Grounded by your SOP knowledge base"}
          </Text>
        </Flex>

        <Flex align="end" gap="3" className="composer-row">
          <Box className="composer-field">
            <TextArea
              ref={inputRef}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSubmit();
                }
              }}
              disabled={isLoading}
              autoFocus
              rows={3}
              placeholder="Ask about workflows, approvals, SOP responsibilities, or a specific document..."
              className="composer-textarea"
            />

            <Flex justify="between" align="center" className="composer-hints">
              <Text size="1" className="muted-copy">
                Enter to send. Shift+Enter for a new line.
              </Text>
              <Text size="1" className="muted-copy">
                {sourceLocked ? "Answers stay on the selected SOP." : "The assistant may route across SOPs."}
              </Text>
            </Flex>
          </Box>

          <Button
            size="3"
            color="cyan"
            className="composer-send-button"
            onClick={handleSubmit}
            disabled={isLoading || !value.trim()}
          >
            <SendHorizontal size={18} />
            Send
          </Button>
        </Flex>
      </Card>
    </Box>
  );
}
