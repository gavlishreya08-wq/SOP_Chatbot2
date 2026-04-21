import { Box, Dialog, Flex, VisuallyHidden } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { healthCheck, listSops } from "./api/client";
import ChatView from "./components/ChatView";
import Sidebar from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import type { HealthInfo, LlmProvider, ModelOption, SopEntry } from "./types";

const fallbackModelOptions: ModelOption[] = [
  {
    provider: "gemini",
    label: "Gemini",
    model: "Configured on backend",
    enabled: true,
  },
  {
    provider: "groq",
    label: "Groq",
    model: "Configured on backend",
    enabled: true,
  },
];

export default function App() {
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>("gemini");
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(fallbackModelOptions);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [sopList, setSopList] = useState<SopEntry[]>([]);
  const [adminToken, setAdminToken] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const chat = useChat(selectedProvider);

  useEffect(() => {
    let cancelled = false;

    healthCheck()
      .then((data) => {
        if (cancelled) return;
        setHealth(data);
        setModelOptions(
          data.available_models?.length ? data.available_models : fallbackModelOptions
        );
        setSelectedProvider((current) => {
          const enabledProviders = (data.available_models || []).filter(
            (option) => option.enabled
          );
          if (enabledProviders.some((option) => option.provider === current)) {
            return current;
          }
          return enabledProviders[0]?.provider || data.llm_provider || current;
        });
      })
      .catch(() => {
        if (!cancelled) {
          setModelOptions(fallbackModelOptions);
        }
      });

    listSops()
      .then((sops) => {
        if (!cancelled) setSopList(sops);
      })
      .catch(() => {
        if (!cancelled) setSopList([]);
      });

    const interval = setInterval(() => {
      healthCheck()
        .then((data) => {
          if (!cancelled) setHealth(data);
        })
        .catch(() => {});
    }, 30000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const selectedModel =
    modelOptions.find((option) => option.provider === selectedProvider) || null;

  return (
    <Box className="app-shell">
      <Box className="app-orb app-orb-one" />
      <Box className="app-orb app-orb-two" />

      <Flex className="app-layout">
        <Box className="app-sidebar desktop-only">
          <Sidebar
            messages={chat.messages}
            onClearChat={chat.clearChat}
            onAskAgain={chat.send}
            adminToken={adminToken}
            onAdminLogin={setAdminToken}
            onAdminLogout={() => setAdminToken(null)}
            modelOptions={modelOptions}
            selectedProvider={selectedProvider}
            onSelectProvider={setSelectedProvider}
            activeSop={chat.activeSop}
            sopList={sopList}
          />
        </Box>

        <Box className="app-main">
          <ChatView
            messages={chat.messages}
            isLoading={chat.isLoading}
            onSend={chat.send}
            onRetry={chat.retryLast}
            onStop={chat.stopStreaming}
            onEditMessage={chat.editAndResend}
            onFeedback={chat.setFeedback}
            onShowMore={chat.showMore}
            modelLabel={selectedModel?.label || health?.llm_provider || "Model"}
            modelName={selectedModel?.model || health?.model || "Configured on backend"}
            activeSop={chat.activeSop}
            health={health}
            sourceLocked={chat.sourceLocked}
            onToggleSourceLock={chat.toggleSourceLock}
            answerMode={chat.answerMode}
            onSetAnswerMode={chat.setAnswerMode}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        </Box>
      </Flex>

      <Dialog.Root open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <Dialog.Content className="mobile-sidebar-dialog">
          <VisuallyHidden>
            <Dialog.Title>Workspace controls</Dialog.Title>
            <Dialog.Description>
              Configure models, answer settings, history, comparison, and admin tools.
            </Dialog.Description>
          </VisuallyHidden>
          <Sidebar
            messages={chat.messages}
            onClearChat={chat.clearChat}
            onAskAgain={chat.send}
            adminToken={adminToken}
            onAdminLogin={setAdminToken}
            onAdminLogout={() => setAdminToken(null)}
            modelOptions={modelOptions}
            selectedProvider={selectedProvider}
            onSelectProvider={setSelectedProvider}
            activeSop={chat.activeSop}
            sopList={sopList}
            onRequestClose={() => setSidebarOpen(false)}
          />
        </Dialog.Content>
      </Dialog.Root>
    </Box>
  );
}
