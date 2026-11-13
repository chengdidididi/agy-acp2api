import { AcpPromptContentBlock } from "../types/acp.js";
import { OpenAIMessage, OpenAIToolDefinition } from "../types/openai.js";
import {
  extractContentSegments,
  resolveImageSourceToBlock,
} from "./image-content.js";
import { extractTextContent } from "./session-hash.js";

export interface CompiledContext {
  systemPromptAppend: string;
  promptText: string;
  promptBlocks?: AcpPromptContentBlock[];
}

/**
 * Builds the minimal system prompt append instructions injected into `session/new` via `_meta.antigravity.systemPrompt.append`.
 * Paired with the worker's `excludeDefaultComponents: true` custom agent, only a single-line guard is needed
 * to suppress the residual `manage_task` control-plane tool, plus compact `<tool_call>` formatting when external tools are present.
 */
export function buildSystemPromptAppend(
  messages: OpenAIMessage[],
  tools?: OpenAIToolDefinition[],
  toolChoice?: unknown
): string {
  const systemMessages = messages
    .filter((m) => m.role === "system" || m.role === "developer")
    .map((m) => extractTextContent(m.content).trim())
    .filter(Boolean);

  const blocks: string[] = [
    "Never call built-in tools (such as manage_task).",
  ];

  if (tools && tools.length > 0 && toolChoice !== "none") {
    const toolDefsJson = JSON.stringify(
      tools.map((t) => ({
        name: t.function.name,
        ...(t.function.description ? { description: t.function.description } : {}),
        parameters: t.function.parameters || { type: "object", properties: {} },
      }))
    );

    let toolProtocol =
      'To call external tools from <available_tools>, emit raw XML (valid JSON inside, no code fences):\n<tool_call>\n{"name":"tool_name","arguments":{}}\n</tool_call>\nFor parallel calls, emit multiple <tool_call> blocks back-to-back. After <tool_call>, stop immediately and wait for <tool_response> without commentary or repetition.';

    if (toolChoice === "required") {
      toolProtocol +=
        "\ntool_choice is 'required': you MUST emit at least one <tool_call> block in this turn.";
    } else if (
      typeof toolChoice === "object" &&
      toolChoice !== null &&
      "function" in toolChoice &&
      typeof (toolChoice as { function?: { name?: string } }).function?.name === "string"
    ) {
      const forcedName = (toolChoice as { function: { name: string } }).function.name;
      toolProtocol += `\nYou MUST call "${forcedName}" using a <tool_call> block in this turn.`;
    }

    blocks.push(toolProtocol);
    blocks.push(`<available_tools>\n${toolDefsJson}\n</available_tools>`);
  }

  if (systemMessages.length > 0) {
    blocks.push(systemMessages.join("\n\n"));
  }

  return blocks.join("\n\n");
}

/**
 * Serializes a single non-system OpenAI message into a clear transcript block.
 */
function serializeSingleMessage(msg: OpenAIMessage): string {
  const text = extractTextContent(msg.content).trim();

  if (msg.role === "user") {
    return text;
  }

  if (msg.role === "assistant") {
    const parts: string[] = [];
    if (text) {
      // Strip any historical post-tool_call self-correction hallucinations from polluted transcripts
      const cleanedText =
        msg.tool_calls && msg.tool_calls.length > 0
          ? text
              .replace(/^\s*(?:_\]|\]<\/tool_call>|<\/tool_call>|\]{2,}>\s*\u03f5?).*$/gms, "")
              .trim()
          : text;
      if (cleanedText) parts.push(cleanedText);
    }
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      for (const tc of msg.tool_calls) {
        let parsedArgs: unknown = {};
        try {
          parsedArgs = JSON.parse(tc.function.arguments || "{}");
        } catch {
          parsedArgs = tc.function.arguments;
        }
        parts.push(
          `<tool_call>\n${JSON.stringify({
            name: tc.function.name,
            arguments: parsedArgs,
          })}\n</tool_call>`
        );
      }
    }
    return parts.join("\n");
  }

  if (msg.role === "tool") {
    const idAttr = msg.tool_call_id ? ` id="${msg.tool_call_id}"` : "";
    const nameAttr = msg.name ? ` name="${msg.name}"` : "";
    return `<tool_response${idAttr}${nameAttr}>\n${text}\n</tool_response>`;
  }

  return text;
}

/**
 * Compiles full conversation history when creating a brand new ACP session (Cache Miss).
 * Prior turns are fenced inside `<conversation_history>` and the trailing user message
 * always retains an explicit `[USER]` tag so historical turns and active user input never blend.
 */
export function compileFullPrompt(
  messages: OpenAIMessage[],
  tools?: OpenAIToolDefinition[],
  toolChoice?: unknown
): CompiledContext {
  const systemPromptAppend = buildSystemPromptAppend(messages, tools, toolChoice);
  const nonSystem = messages.filter((m) => m.role !== "system" && m.role !== "developer");

  // Single user message fast-path (zero wrapper overhead on Turn 1)
  if (nonSystem.length === 1 && nonSystem[0].role === "user") {
    return {
      systemPromptAppend,
      promptText: extractTextContent(nonSystem[0].content),
    };
  }

  let lastUserIdx = -1;
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    if (nonSystem[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  const formatTaggedMessage = (m: OpenAIMessage): string => {
    const body = serializeSingleMessage(m);
    if (m.role === "tool") {
      return body;
    }
    return `[${m.role.toUpperCase()}]\n${body}`;
  };

  if (lastUserIdx > 0) {
    const historyBlock = nonSystem
      .slice(0, lastUserIdx)
      .map(formatTaggedMessage)
      .join("\n\n");
    const activeTurnBlock = nonSystem
      .slice(lastUserIdx)
      .map(formatTaggedMessage)
      .join("\n\n");

    return {
      systemPromptAppend,
      promptText: `<conversation_history>\n${historyBlock}\n</conversation_history>\n\n${activeTurnBlock}`,
    };
  }

  // Fallback when all messages belong to Turn 1 (e.g. [user, assistant(tool_call), tool])
  return {
    systemPromptAppend,
    promptText: nonSystem.map(formatTaggedMessage).join("\n\n"),
  };
}

/**
 * Compiles ONLY the delta messages (e.g. `<tool_response>` or latest user message)
 * when a Session Anchor + Bookmark match hits an existing warm ACP session.
 */
export function compileDeltaPrompt(
  deltaMessages: OpenAIMessage[],
  isRetryAfterInterrupt: boolean = false
): string {
  if (deltaMessages.length === 0) {
    return "";
  }

  // Pure tool responses (single or parallel tool results)
  if (deltaMessages.every((m) => m.role === "tool")) {
    return deltaMessages.map((m) => serializeSingleMessage(m)).join("\n\n");
  }

  // Standard single user follow-up on a clean session
  if (
    deltaMessages.length === 1 &&
    deltaMessages[0].role === "user" &&
    !isRetryAfterInterrupt
  ) {
    return serializeSingleMessage(deltaMessages[0]);
  }

  // Mixed delta (e.g. cancelled <tool_response> + new user message, multi-message catch-up,
  // or retry after an interrupted/cancelled turn): explicitly tag [USER] / [ASSISTANT] roles.
  return deltaMessages
    .map((m) => {
      const body = serializeSingleMessage(m);
      if (m.role === "tool") {
        return body;
      }
      return `[${m.role.toUpperCase()}]\n${body}`;
    })
    .join("\n\n");
}

class PromptBlockBuilder {
  private blocks: AcpPromptContentBlock[] = [];

  public pushText(text: string): void {
    if (!text) return;
    const last = this.blocks[this.blocks.length - 1];
    if (last && last.type === "text") {
      last.text += text;
    } else {
      this.blocks.push({ type: "text", text });
    }
  }

  public pushImage(imageBlock: AcpPromptContentBlock): void {
    this.blocks.push(imageBlock);
  }

  public finalize(): AcpPromptContentBlock[] {
    const hasImage = this.blocks.some((b) => b.type === "image");
    if (!hasImage) {
      return this.blocks.length > 0 ? this.blocks : [{ type: "text", text: "" }];
    }

    const cleaned: AcpPromptContentBlock[] = [];
    for (const b of this.blocks) {
      if (b.type === "text" && !/\S/.test(b.text)) {
        continue;
      }
      const prev = cleaned[cleaned.length - 1];
      if (b.type === "text" && prev && prev.type === "text") {
        prev.text += b.text;
      } else {
        cleaned.push(b);
      }
    }

    return cleaned.length > 0 ? cleaned : [{ type: "text", text: "" }];
  }
}

async function appendSingleMessageBlocks(
  builder: PromptBlockBuilder,
  msg: OpenAIMessage,
  abortSignal?: AbortSignal,
  trimUserText: boolean = true
): Promise<void> {
  const segments = extractContentSegments(msg.content);
  const hasImages = segments.some((s) => s.kind !== "text");

  if (!hasImages) {
    if (msg.role === "user" && !trimUserText) {
      builder.pushText(extractTextContent(msg.content));
    } else {
      builder.pushText(serializeSingleMessage(msg));
    }
    return;
  }

  if (msg.role === "tool") {
    const idAttr = msg.tool_call_id ? ` id="${msg.tool_call_id}"` : "";
    const nameAttr = msg.name ? ` name="${msg.name}"` : "";
    builder.pushText(`<tool_response${idAttr}${nameAttr}>\n`);
    for (const seg of segments) {
      if (seg.kind === "text") {
        const trimmed = seg.text.trim();
        if (trimmed) builder.pushText(trimmed);
      } else {
        builder.pushImage(await resolveImageSourceToBlock(seg, abortSignal));
      }
    }
    builder.pushText(`\n</tool_response>`);
    return;
  }

  if (msg.role === "assistant") {
    for (const seg of segments) {
      if (seg.kind !== "text") {
        builder.pushImage(await resolveImageSourceToBlock(seg, abortSignal));
      }
    }
    const serialized = serializeSingleMessage(msg);
    if (serialized) {
      builder.pushText(serialized);
    }
    return;
  }

  // `user` (or fallback role) with one or more attached images
  for (const seg of segments) {
    if (seg.kind === "text") {
      const txt = trimUserText ? seg.text.trim() : seg.text;
      if (txt) builder.pushText(txt);
    } else {
      builder.pushImage(await resolveImageSourceToBlock(seg, abortSignal));
    }
  }
}

async function appendMessageSequenceBlocks(
  builder: PromptBlockBuilder,
  msgs: OpenAIMessage[],
  tagged: boolean,
  abortSignal?: AbortSignal
): Promise<void> {
  for (let i = 0; i < msgs.length; i++) {
    if (i > 0) {
      builder.pushText("\n\n");
    }
    const m = msgs[i];
    if (tagged && m.role !== "tool") {
      builder.pushText(`[${m.role.toUpperCase()}]\n`);
    }
    await appendSingleMessageBlocks(builder, m, abortSignal, true);
  }
}

/**
 * Multimodal-aware full prompt compiler for `session/new` (Cache Miss).
 * Preserves exact chronological ordering of text and `{ type: "image", mimeType, data }`
 * ACP ContentBlocks while remaining 100% identical to `compileFullPrompt` on text-only inputs.
 */
export async function compileFullPromptBlocks(
  messages: OpenAIMessage[],
  tools?: OpenAIToolDefinition[],
  toolChoice?: unknown,
  abortSignal?: AbortSignal
): Promise<CompiledContext & { promptBlocks: AcpPromptContentBlock[] }> {
  const base = compileFullPrompt(messages, tools, toolChoice);
  const nonSystem = messages.filter(
    (m) => m.role !== "system" && m.role !== "developer"
  );

  const builder = new PromptBlockBuilder();

  if (nonSystem.length === 1 && nonSystem[0].role === "user") {
    await appendSingleMessageBlocks(builder, nonSystem[0], abortSignal, false);
    return {
      ...base,
      promptBlocks: builder.finalize(),
    };
  }

  let lastUserIdx = -1;
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    if (nonSystem[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  if (lastUserIdx > 0) {
    builder.pushText("<conversation_history>\n");
    await appendMessageSequenceBlocks(
      builder,
      nonSystem.slice(0, lastUserIdx),
      true,
      abortSignal
    );
    builder.pushText("\n</conversation_history>\n\n");
    await appendMessageSequenceBlocks(
      builder,
      nonSystem.slice(lastUserIdx),
      true,
      abortSignal
    );
    return {
      ...base,
      promptBlocks: builder.finalize(),
    };
  }

  await appendMessageSequenceBlocks(builder, nonSystem, true, abortSignal);
  return {
    ...base,
    promptBlocks: builder.finalize(),
  };
}

/**
 * Multimodal-aware delta prompt compiler for warm ACP sessions (Session Cache HIT).
 * Sends only the newly added text and/or image `ContentBlock`s for the current turn.
 */
export async function compileDeltaPromptBlocks(
  deltaMessages: OpenAIMessage[],
  isRetryAfterInterrupt: boolean = false,
  abortSignal?: AbortSignal
): Promise<AcpPromptContentBlock[]> {
  if (deltaMessages.length === 0) {
    return [{ type: "text", text: "" }];
  }

  const builder = new PromptBlockBuilder();

  if (deltaMessages.every((m) => m.role === "tool")) {
    await appendMessageSequenceBlocks(builder, deltaMessages, false, abortSignal);
    return builder.finalize();
  }

  if (
    deltaMessages.length === 1 &&
    deltaMessages[0].role === "user" &&
    !isRetryAfterInterrupt
  ) {
    await appendSingleMessageBlocks(builder, deltaMessages[0], abortSignal, true);
    return builder.finalize();
  }

  await appendMessageSequenceBlocks(builder, deltaMessages, true, abortSignal);
  return builder.finalize();
}
