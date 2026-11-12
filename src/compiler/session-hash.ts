import crypto from "node:crypto";
import { OpenAIMessage, OpenAIToolDefinition } from "../types/openai.js";

export const MAX_CONSECUTIVE_SESSION_ABORTS = 2;
export const MAX_SESSION_CONTEXT_TOKENS = 900_000;

export type SessionInvalidationReason =
  | "expired"
  | "consecutive_aborts"
  | "history_rewound"
  | "tools_changed"
  | "context_limit"
  | "empty_delta";

export interface SessionAffinityEntry {
  sessionId: string;
  workerId: string;
  modelId: string;
  toolsSignature: string;
  firstUserFp: string;
  firstAnyAssistantFp: string | null;
  firstTextAssistantFp: string | null;
  lastSyncedAssistantFp: string;
  lastSyncedAssistantTextFp: string | null;
  lastAbortedAssistantFp: string | null;
  lastAbortedAssistantTextFp: string | null;
  syncedTurnCount: number;
  consecutiveAborts: number;
  hasPendingInterruptedTurn: boolean;
  lastTotalTokens: number;
  messageCount: number;
  updatedAt: number;
}

export interface PrefixMatchResult {
  entry: SessionAffinityEntry;
  deltaMessages: OpenAIMessage[];
  matchedPrefixLength: number;
  isRetryAfterInterrupt: boolean;
}

export interface SessionMatchCallbacks {
  isSessionAlive: (workerId: string, sessionId: string) => boolean;
  isWorkerIdle: (workerId: string) => boolean;
  onInvalidateSession?: (
    workerId: string,
    sessionId: string,
    reason: SessionInvalidationReason
  ) => void;
}

export function extractTextContent(content: OpenAIMessage["content"]): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part.type === "text" ? part.text || "" : ""))
      .join("\n");
  }
  return String(content);
}

function sortJsonDeep(val: unknown): unknown {
  if (Array.isArray(val)) {
    return val.map(sortJsonDeep);
  }
  if (val !== null && typeof val === "object") {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(val as Record<string, unknown>).sort()) {
      sorted[key] = sortJsonDeep((val as Record<string, unknown>)[key]);
    }
    return sorted;
  }
  return val;
}

export function canonicalizeJsonString(raw: string | undefined): string {
  if (!raw) return "{}";
  try {
    const parsed = JSON.parse(raw);
    return JSON.stringify(sortJsonDeep(parsed));
  } catch {
    return raw.trim();
  }
}

function sha256Short(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

function normalizeTextForFingerprint(raw: string): string {
  return raw.replace(/\r\n/g, "\n").replace(/\s+/g, " ").trim();
}

export function computeUserFingerprint(msg: OpenAIMessage): string {
  const norm = normalizeTextForFingerprint(extractTextContent(msg.content));
  return `usr:${sha256Short(norm)}`;
}

export function computeAssistantTextFingerprint(
  msg: OpenAIMessage
): string | null {
  const norm = normalizeTextForFingerprint(extractTextContent(msg.content));
  if (!norm) return null;
  return `atxt:${sha256Short(norm)}`;
}

export function computeAssistantFullFingerprint(msg: OpenAIMessage): string {
  const norm = normalizeTextForFingerprint(extractTextContent(msg.content));
  const toolCalls = (msg.tool_calls || []).map((tc) => ({
    name: tc.function?.name || "",
    arguments: canonicalizeJsonString(tc.function?.arguments),
  }));
  return `afull:${sha256Short(JSON.stringify({ text: norm, toolCalls }))}`;
}

export function computeToolsSignature(
  tools: OpenAIToolDefinition[] | undefined
): string {
  if (!tools || tools.length === 0) return "tools:none";
  const canonical = tools
    .map((t) => ({
      name: t.function.name,
      parameters: sortJsonDeep(t.function.parameters || {}),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
  return `tools:${sha256Short(JSON.stringify(canonical))}`;
}

export function canonicalizeMessage(
  msg: OpenAIMessage
): Record<string, unknown> {
  const role = msg.role === "developer" ? "system" : msg.role;
  const text = extractTextContent(msg.content).trim();

  if (role === "assistant") {
    const toolCalls = (msg.tool_calls || []).map((tc) => ({
      name: tc.function?.name || "",
      arguments: canonicalizeJsonString(tc.function?.arguments),
    }));
    return {
      role: "assistant",
      content: text,
      toolCalls,
    };
  }

  if (role === "tool") {
    return {
      role: "tool",
      content: text,
    };
  }

  return {
    role,
    content: text,
  };
}

export function computeConversationHash(
  baseModelId: string,
  tools: OpenAIToolDefinition[] | undefined,
  messages: OpenAIMessage[]
): string {
  const canonicalTools = (tools || []).map((t) => ({
    name: t.function.name,
    parameters: sortJsonDeep(t.function.parameters || {}),
  }));

  const canonicalMessages = messages.map(canonicalizeMessage);

  const payload = JSON.stringify({
    model: baseModelId,
    tools: canonicalTools,
    messages: canonicalMessages,
  });

  return crypto.createHash("sha256").update(payload).digest("hex");
}

function getNonSystemMessages(messages: OpenAIMessage[]): OpenAIMessage[] {
  return messages.filter(
    (m) => m.role !== "system" && m.role !== "developer"
  );
}

function extractConversationAnchors(nonSystem: OpenAIMessage[]): {
  firstUserFp: string;
  firstAnyAssistantFp: string | null;
  firstTextAssistantFp: string | null;
} | null {
  const firstUserIdx = nonSystem.findIndex((m) => m.role === "user");
  if (firstUserIdx === -1) return null;

  const firstUserFp = computeUserFingerprint(nonSystem[firstUserIdx]);
  let firstAnyAssistantFp: string | null = null;
  let firstTextAssistantFp: string | null = null;

  for (let i = firstUserIdx + 1; i < nonSystem.length; i++) {
    const m = nonSystem[i];
    if (m.role !== "assistant") continue;
    if (!firstAnyAssistantFp) {
      firstAnyAssistantFp = computeAssistantFullFingerprint(m);
    }
    if (!firstTextAssistantFp) {
      const txtFp = computeAssistantTextFingerprint(m);
      if (txtFp) {
        firstTextAssistantFp = txtFp;
      }
    }
    if (firstAnyAssistantFp && firstTextAssistantFp) {
      break;
    }
  }

  if (!firstAnyAssistantFp && !firstTextAssistantFp) {
    return null;
  }

  return {
    firstUserFp,
    firstAnyAssistantFp,
    firstTextAssistantFp,
  };
}

function findBookmarkIndex(
  nonSystem: OpenAIMessage[],
  fullFp: string,
  textFp: string | null
): number {
  // Pass 1: Exact assistant message match (text + tool_calls)
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const m = nonSystem[i];
    if (m.role !== "assistant") continue;
    if (computeAssistantFullFingerprint(m) === fullFp) {
      return i;
    }
  }

  // Pass 2: Text-only match when client stripped historical tool_calls from completed turns
  if (textFp) {
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      const m = nonSystem[i];
      if (m.role !== "assistant") continue;
      if (m.tool_calls && m.tool_calls.length > 0) continue;
      if (computeAssistantTextFingerprint(m) === textFp) {
        return i;
      }
    }
  }

  return -1;
}

export class SessionAffinityRegistry {
  /** Keyed by `sessionId` (one canonical stateful record per live ACP session) */
  private sessions = new Map<string, SessionAffinityEntry>();
  private ttlMs: number;

  constructor(ttlMs: number = 1_800_000) {
    this.ttlMs = ttlMs;
  }

  public setTtlMs(
    ttlMs: number,
    onExpire?: (
      workerId: string,
      sessionId: string,
      reason: SessionInvalidationReason
    ) => void
  ): void {
    this.ttlMs = ttlMs;
    this.pruneExpired(onExpire);
  }

  public clear(): void {
    this.sessions.clear();
  }

  public hasActiveSessionsForWorker(workerId: string): boolean {
    this.pruneExpired();
    for (const entry of this.sessions.values()) {
      if (entry.workerId === workerId) {
        return true;
      }
    }
    return false;
  }

  public countSessionsForWorker(workerId: string): number {
    let count = 0;
    for (const entry of this.sessions.values()) {
      if (entry.workerId === workerId) count++;
    }
    return count;
  }

  /**
   * Peeks at candidate sessions matching `(baseModelId + firstUser + firstAssistant)`
   * without mutating state. Used by `WorkerPool` to wait briefly if the owning worker
   * is currently finishing a `session/cancel` settle from an aborted/retried turn.
   */
  public findCandidateEntries(
    baseModelId: string,
    messages: OpenAIMessage[]
  ): SessionAffinityEntry[] {
    const nonSystem = getNonSystemMessages(messages);
    const anchors = extractConversationAnchors(nonSystem);
    if (!anchors) return [];

    const now = Date.now();
    const matches: Array<{
      entry: SessionAffinityEntry;
      hasBookmark: boolean;
    }> = [];

    for (const entry of this.sessions.values()) {
      if (now - entry.updatedAt > this.ttlMs) continue;
      if (entry.modelId !== baseModelId) continue;
      if (entry.firstUserFp !== anchors.firstUserFp) continue;

      const textAnchorMatched = Boolean(
        anchors.firstTextAssistantFp &&
          entry.firstTextAssistantFp &&
          anchors.firstTextAssistantFp === entry.firstTextAssistantFp
      );
      const anyAnchorMatched = Boolean(
        anchors.firstAnyAssistantFp &&
          entry.firstAnyAssistantFp &&
          anchors.firstAnyAssistantFp === entry.firstAnyAssistantFp
      );

      if (!textAnchorMatched && !anyAnchorMatched) continue;

      const hasBookmark =
        findBookmarkIndex(
          nonSystem,
          entry.lastSyncedAssistantFp,
          entry.lastSyncedAssistantTextFp
        ) !== -1;
      matches.push({ entry, hasBookmark });
    }

    matches.sort((a, b) => {
      if (a.hasBookmark !== b.hasBookmark) {
        return a.hasBookmark ? -1 : 1;
      }
      return b.entry.updatedAt - a.entry.updatedAt;
    });

    return matches.map((m) => m.entry);
  }

  /**
   * Matches an incoming OpenAI `messages` array to an active ACP session using
   * `(firstUser + firstAssistant)` anchoring + `lastSyncedAssistant` progress bookmark,
   * while enforcing the 4 zero-cost session health checks:
   *   1. Consecutive abort circuit breaker (`consecutiveAborts >= 2`)
   *   2. Client history rewind / deletion detection (missing progress bookmark)
   *   3. External tool definitions changed (`toolsSignature`)
   *   4. Context window fullness guard (`lastTotalTokens >= MAX_SESSION_CONTEXT_TOKENS`)
   */
  public findBestMatchingPrefix(
    baseModelId: string,
    tools: OpenAIToolDefinition[] | undefined,
    messages: OpenAIMessage[],
    callbacks:
      | SessionMatchCallbacks
      | ((workerId: string, sessionId: string) => boolean)
  ): PrefixMatchResult | null {
    const cb: SessionMatchCallbacks =
      typeof callbacks === "function"
        ? {
            isSessionAlive: callbacks,
            isWorkerIdle: () => true,
          }
        : callbacks;

    this.pruneExpired(cb.onInvalidateSession);

    const nonSystem = getNonSystemMessages(messages);
    const candidates = this.findCandidateEntries(baseModelId, messages);
    if (candidates.length === 0) return null;

    const incomingToolsSig = computeToolsSignature(tools);

    for (const entry of candidates) {
      if (!cb.isSessionAlive(entry.workerId, entry.sessionId)) {
        this.sessions.delete(entry.sessionId);
        continue;
      }

      if (!cb.isWorkerIdle(entry.workerId)) {
        continue;
      }

      // Health Check 1: Consecutive abort circuit breaker
      if (entry.consecutiveAborts >= MAX_CONSECUTIVE_SESSION_ABORTS) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "consecutive_aborts"
        );
        continue;
      }

      // Health Check 2: External tool definitions changed mid-conversation
      if (entry.toolsSignature !== incomingToolsSig) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "tools_changed"
        );
        continue;
      }

      // Health Check 3: Context window fullness guard
      if (entry.lastTotalTokens >= MAX_SESSION_CONTEXT_TOKENS) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "context_limit"
        );
        continue;
      }

      // Health Check 4: Progress bookmark & client history rewind/deletion check
      const bookmarkIdx = findBookmarkIndex(
        nonSystem,
        entry.lastSyncedAssistantFp,
        entry.lastSyncedAssistantTextFp
      );

      if (bookmarkIdx === -1) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "history_rewound"
        );
        continue;
      }

      let sliceStartIdx = bookmarkIdx + 1;

      // If the previous attempt on this session was interrupted mid-stream and the client
      // kept that truncated partial assistant message in history, advance past it so we
      // don't resend partial assistant output that `agy acp` already generated.
      if (
        entry.hasPendingInterruptedTurn &&
        (entry.lastAbortedAssistantFp || entry.lastAbortedAssistantTextFp)
      ) {
        const trailingSlice = nonSystem.slice(sliceStartIdx);
        const abortedRelIdx = findBookmarkIndex(
          trailingSlice,
          entry.lastAbortedAssistantFp || "",
          entry.lastAbortedAssistantTextFp
        );
        if (abortedRelIdx !== -1 && abortedRelIdx < trailingSlice.length - 1) {
          sliceStartIdx += abortedRelIdx + 1;
        }
      }

      const deltaMessages = nonSystem.slice(sliceStartIdx);
      if (deltaMessages.length === 0) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "empty_delta"
        );
        continue;
      }

      // Refresh TTL on hit!
      entry.updatedAt = Date.now();
      return {
        entry,
        deltaMessages,
        matchedPrefixLength: sliceStartIdx,
        isRetryAfterInterrupt: entry.hasPendingInterruptedTurn,
      };
    }

    return null;
  }

  /**
   * Registers or updates a live ACP session after a successful `session/prompt` turn,
   * refreshing its TTL and resetting consecutive abort counters.
   */
  public registerTurnState(
    baseModelId: string,
    tools: OpenAIToolDefinition[] | undefined,
    fullMessagesWithReply: OpenAIMessage[],
    workerId: string,
    sessionId: string,
    totalTokens: number = 0
  ): string {
    const nonSystem = getNonSystemMessages(fullMessagesWithReply);
    const anchors = extractConversationAnchors(nonSystem);
    const lastAsst = [...nonSystem]
      .reverse()
      .find((m) => m.role === "assistant");

    if (!anchors || !lastAsst) {
      return sessionId;
    }

    const existing = this.sessions.get(sessionId);
    const lastSyncedAssistantFp = computeAssistantFullFingerprint(lastAsst);
    const lastSyncedAssistantTextFp = computeAssistantTextFingerprint(lastAsst);

    this.sessions.set(sessionId, {
      sessionId,
      workerId,
      modelId: baseModelId,
      toolsSignature: existing?.toolsSignature || computeToolsSignature(tools),
      firstUserFp: existing?.firstUserFp || anchors.firstUserFp,
      firstAnyAssistantFp:
        existing?.firstAnyAssistantFp || anchors.firstAnyAssistantFp,
      firstTextAssistantFp:
        existing?.firstTextAssistantFp || anchors.firstTextAssistantFp,
      lastSyncedAssistantFp,
      lastSyncedAssistantTextFp,
      lastAbortedAssistantFp: null,
      lastAbortedAssistantTextFp: null,
      syncedTurnCount: (existing?.syncedTurnCount ?? 0) + 1,
      consecutiveAborts: 0,
      hasPendingInterruptedTurn: false,
      lastTotalTokens:
        totalTokens > 0 ? totalTokens : existing?.lastTotalTokens ?? 0,
      messageCount: nonSystem.length,
      updatedAt: Date.now(),
    });

    return sessionId;
  }

  /**
   * Records a timeout, client cancellation, or RPC error on a session:
   * - Refreshes TTL so a quick retry won't expire mid-recovery;
   * - Allows the 1st interruption (`consecutiveAborts === 1`) to reuse the warm session;
   * - Trips the circuit breaker (`shouldDestroy: true`) ifTurn 1 never completed or if
   *   `consecutiveAborts >= MAX_CONSECUTIVE_SESSION_ABORTS` (2).
   */
  public recordTurnInterruption(
    sessionId: string,
    partialAssistantMsg?: OpenAIMessage
  ): { shouldDestroy: boolean; consecutiveAborts: number } {
    const existing = this.sessions.get(sessionId);
    if (!existing || existing.syncedTurnCount === 0) {
      this.sessions.delete(sessionId);
      return { shouldDestroy: true, consecutiveAborts: 1 };
    }

    existing.consecutiveAborts += 1;
    existing.hasPendingInterruptedTurn = true;
    existing.updatedAt = Date.now();

    if (partialAssistantMsg) {
      const hasText =
        extractTextContent(partialAssistantMsg.content).trim().length > 0;
      const hasTools = Boolean(
        partialAssistantMsg.tool_calls &&
          partialAssistantMsg.tool_calls.length > 0
      );
      if (hasText || hasTools) {
        existing.lastAbortedAssistantFp =
          computeAssistantFullFingerprint(partialAssistantMsg);
        existing.lastAbortedAssistantTextFp =
          computeAssistantTextFingerprint(partialAssistantMsg);
      }
    }

    if (existing.consecutiveAborts >= MAX_CONSECUTIVE_SESSION_ABORTS) {
      this.sessions.delete(sessionId);
      return {
        shouldDestroy: true,
        consecutiveAborts: existing.consecutiveAborts,
      };
    }

    return {
      shouldDestroy: false,
      consecutiveAborts: existing.consecutiveAborts,
    };
  }

  public removeSession(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  public removeWorkerSessions(workerId: string): void {
    for (const [sid, entry] of this.sessions.entries()) {
      if (entry.workerId === workerId) {
        this.sessions.delete(sid);
      }
    }
  }

  public pruneExpired(
    onExpire?: (
      workerId: string,
      sessionId: string,
      reason: SessionInvalidationReason
    ) => void
  ): void {
    const now = Date.now();
    for (const [sid, entry] of this.sessions.entries()) {
      if (now - entry.updatedAt > this.ttlMs) {
        this.sessions.delete(sid);
        onExpire?.(entry.workerId, entry.sessionId, "expired");
      }
    }
  }

  public size(): number {
    this.pruneExpired();
    return this.sessions.size;
  }
}
