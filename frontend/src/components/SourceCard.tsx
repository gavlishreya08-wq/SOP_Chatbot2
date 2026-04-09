import { Badge, Box, Card, Flex, Text } from "@radix-ui/themes";
import { ExternalLink, FileText } from "lucide-react";
import type { SourceInfo } from "../types";

interface Props {
  source: SourceInfo;
}

export default function SourceCard({ source }: Props) {
  return (
    <Card size="3" className="surface-panel source-card">
      <Flex justify="between" align="start" gap="4" wrap="wrap" className="source-card-header">
        <Flex align="start" gap="3">
          <Box className="source-card-icon">
            <FileText size={18} className="accent-cyan" />
          </Box>

          <Box>
            <Text size="2" weight="medium" as="p">
              {source.title}
            </Text>
            <Text size="1" className="muted-copy" mt="1" as="p">
              Referenced SOP metadata and citation points for this response.
            </Text>
          </Box>
        </Flex>

        {source.link ? (
          <a
            href={source.link}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link-button"
          >
            <ExternalLink size={13} />
            Open SOP
          </a>
        ) : null}
      </Flex>

      <Flex gap="2" wrap="wrap" mt="4">
        <Badge color="cyan" variant="soft" radius="full">
          Version {source.version}
        </Badge>
        <Badge color="gray" variant="surface" radius="full">
          Created {source.created_date}
        </Badge>
        {source.pages && source.pages.length > 0 ? (
          <Badge color="gray" variant="surface" radius="full">
            Pages {source.pages.join(", ")}
          </Badge>
        ) : null}
      </Flex>

      {source.citations && source.citations.length > 0 ? (
        <Flex gap="2" wrap="wrap" mt="4">
          {source.citations.map((citation, index) => (
            <Box
              key={`${citation.page || "na"}-${citation.section || "na"}-${index}`}
              className="citation-chip"
            >
              {citation.page ? `Page ${citation.page}` : "Page N/A"}
              {citation.section ? ` | ${citation.section}` : ""}
            </Box>
          ))}
        </Flex>
      ) : null}
    </Card>
  );
}
