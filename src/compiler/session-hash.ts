import crypto from "node:crypto";
import { OpenAIMessage, OpenAIToolDefinition } from "../types/openai.js";
import { extractImageFingerprints } from "./image-content.js";

export const MAX_CONSECUTIVE_SESSION_ABORTS = 2;
export const MAX_SESSION_CONTEXT_TOKENS = 900_000;
export const MAX_SYNCED_CHECKPOINTS = 6;

export type SessionInvalidationReason =
  | "expired"
  | "consecutive_aborts"
  | "history_rewound"
  | "tools_changed"
  | "context_limit"
  | "empty_delta";

export interface SyncedTurnCheckpoint {
  assistantFullFp: string;
  assistantTextFp: string | null;
  /** Fingerprint of the latest `user` message at or before this checkpoint */
  triggerUserFp: string | null;
}

export interface SessionAffinityEntry {
  sessionId: string;
  workerId: string;
  modelId: string;
  toolsSignature: string;
  firstUserFp: string;
  firstAnyAssistantFp: string | null;
  firstTextAssistantFp: string | null;
  lastSyncedAssistantFp: string | null;
  lastSyncedAssistantTextFp: string | null;
  checkpoints: SyncedTurnCheckpoint[];
  lastAbortedAssistantFp: string | null;
  lastAbortedAssistantTextFp: string | null;
  lastInterruptedUserFp: string | null;
  syncedTurnCount: number;
  consecutiveAborts: number;
  hasPendingInterruptedTurn: boolean;
  lastTotalTokens: number;
  messageCount: number;
  updatedAt: number;
}

export interface RecordInterruptionOptions {
  isClientAbort?: boolean;
  lastUserMsg?: OpenAIMessage;
  partialAssistantMsg?: OpenAIMessage;
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
    const textParts: string[] = [];
    for (const part of content) {
      if (part && (part.type === "text" || part.type === "input_text")) {
        textParts.push(part.text || "");
      }
    }
    return textParts.join("\n");
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
  const imageFps = extractImageFingerprints(msg.content);
  if (imageFps.length === 0) {
    return `usr:${sha256Short(norm)}`;
  }
  return `usr:${sha256Short(JSON.stringify({ text: norm, images: imageFps }))}`;
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
  const imageFps = extractImageFingerprints(msg.content);

  if (role === "assistant") {
    const toolCalls = (msg.tool_calls || []).map((tc) => ({
      name: tc.function?.name || "",
      arguments: canonicalizeJsonString(tc.function?.arguments),
    }));
    return {
      role: "assistant",
      content: text,
      toolCalls,
      ...(imageFps.length > 0 ? { images: imageFps } : {}),
    };
  }

  if (role === "tool") {
    return {
      role: "tool",
      content: text,
      ...(imageFps.length > 0 ? { images: imageFps } : {}),
    };
  }

  return {
    role,
    content: text,
    ...(imageFps.length > 0 ? { images: imageFps } : {}),
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
  hasAssistantMessages: boolean;
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

  return {
    firstUserFp,
    firstAnyAssistantFp,
    firstTextAssistantFp,
    hasAssistantMessages: Boolean(firstAnyAssistantFp || firstTextAssistantFp),
  };
}

function findBookmarkIndex(
  nonSystem: OpenAIMessage[],
  fullFp: string | null,
  textFp: string | null
): number {
  if (fullFp) {
    // Pass 1: Exact assistant message match (text + tool_calls)
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      const m = nonSystem[i];
      if (m.role !== "assistant") continue;
      if (computeAssistantFullFingerprint(m) === fullFp) {
        return i;
      }
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

interface SliceResolution {
  sliceStartIdx: number;
  deltaMessages: OpenAIMessage[];
  isFallbackOrInterrupt: boolean;
}

function resolveEntrySlice(
  entry: SessionAffinityEntry,
  nonSystem: OpenAIMessage[]
): SliceResolution | null {
  const hasAssistantInIncoming = nonSystem.some((m) => m.role === "assistant");

  // Case 0: Session was pre-registered on Turn 1 (`session/new`) and interrupted before
  // any assistant reply completed (`checkpoints.length === 0`).
  if (entry.checkpoints.length === 0) {
    if (hasAssistantInIncoming) return null;
    let sliceStartIdx = 0;
    if (
      entry.lastInterruptedUserFp &&
      nonSystem.length > 1 &&
      nonSystem[0].role === "user" &&
      computeUserFingerprint(nonSystem[0]) === entry.lastInterruptedUserFp
    ) {
      sliceStartIdx = 1;
    }
    const deltaMessages = nonSystem.slice(sliceStartIdx);
    if (deltaMessages.length === 0) return null;
    return {
      sliceStartIdx,
      deltaMessages,
      isFallbackOrInterrupt: entry.hasPendingInterruptedTurn,
    };
  }

  // Case 1: Match the latest synced checkpoint (standard fast path)
  const latest = entry.checkpoints[entry.checkpoints.length - 1];
  const latestIdx = findBookmarkIndex(
    nonSystem,
    latest.assistantFullFp,
    latest.assistantTextFp
  );

  if (latestIdx !== -1) {
    let sliceStartIdx = latestIdx + 1;

    if (entry.hasPendingInterruptedTurn) {
      // Advance past truncated partial assistant reply if client preserved it
      if (entry.lastAbortedAssistantFp || entry.lastAbortedAssistantTextFp) {
        const trailingSlice = nonSystem.slice(sliceStartIdx);
        const abortedRelIdx = findBookmarkIndex(
          trailingSlice,
          entry.lastAbortedAssistantFp,
          entry.lastAbortedAssistantTextFp
        );
        if (abortedRelIdx !== -1 && abortedRelIdx < trailingSlice.length - 1) {
          sliceStartIdx += abortedRelIdx + 1;
        }
      }

      // If client kept the cancelled user message (`U4`) and appended a corrected user message (`U5`),
      // skip the already-sent `U4` so `agy acp` only receives `U5`.
      if (entry.lastInterruptedUserFp) {
        const trailingAfter = nonSystem.slice(sliceStartIdx);
        if (
          trailingAfter.length > 1 &&
          trailingAfter[0].role === "user" &&
          computeUserFingerprint(trailingAfter[0]) ===
            entry.lastInterruptedUserFp
        ) {
          sliceStartIdx += 1;
        }
      }
    }

    const deltaMessages = nonSystem.slice(sliceStartIdx);
    if (deltaMessages.length === 0) return null;
    return {
      sliceStartIdx,
      deltaMessages,
      isFallbackOrInterrupt: entry.hasPendingInterruptedTurn,
    };
  }

  // Case 2: Sliding bookmark fallback — handles frontend-cancel vs backend-finish race
  // (e.g. backend finished short reply `A4` in ~2s and advanced bookmark, while frontend
  // cancelled and discarded `A4`, keeping `A3`), as well as mid-turn tool-loop rollbacks.
  for (let k = entry.checkpoints.length - 2; k >= 0; k--) {
    const skipped = entry.checkpoints.slice(k + 1);
    const skippedTextTurns = skipped.filter(
      (c) => c.assistantTextFp !== null
    ).length;
    if (skippedTextTurns > 1) {
      break;
    }

    const prev = entry.checkpoints[k];
    const prevIdx = findBookmarkIndex(
      nonSystem,
      prev.assistantFullFp,
      prev.assistantTextFp
    );
    if (prevIdx === -1) continue;

    const trailingAfterPrev = nonSystem.slice(prevIdx + 1);
    // Reject if there is a diverged assistant message after `prev`
    if (trailingAfterPrev.some((m) => m.role === "assistant")) {
      return null;
    }

    let sliceStartIdx = prevIdx + 1;
    const nextCheckpoint = entry.checkpoints[k + 1];
    if (
      nextCheckpoint?.triggerUserFp &&
      trailingAfterPrev.length > 1 &&
      trailingAfterPrev[0].role === "user" &&
      computeUserFingerprint(trailingAfterPrev[0]) ===
        nextCheckpoint.triggerUserFp
    ) {
      // Backend already processed `trailingAfterPrev[0]` (e.g. `"就是"`), only send the new follow-up
      sliceStartIdx += 1;
    }

    const deltaMessages = nonSystem.slice(sliceStartIdx);
    if (deltaMessages.length === 0) return null;
    return {
      sliceStartIdx,
      deltaMessages,
      isFallbackOrInterrupt: true,
    };
  }

  // Case 3: Turn-1 uncommitted tool-loop rollback where client sent `[U1, U2_continue]` (0 assistant messages)
  if (!hasAssistantInIncoming && entry.firstTextAssistantFp === null) {
    let sliceStartIdx = 0;
    if (
      nonSystem.length > 1 &&
      nonSystem[0].role === "user" &&
      computeUserFingerprint(nonSystem[0]) === entry.firstUserFp
    ) {
      sliceStartIdx = 1;
    }
    const deltaMessages = nonSystem.slice(sliceStartIdx);
    if (deltaMessages.length === 0) return null;
    return {
      sliceStartIdx,
      deltaMessages,
      isFallbackOrInterrupt: true,
    };
  }

  return null;
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
   * Checks whether an incoming request has matching `toolsSignature` and a valid non-empty
   * delta slice for `entry` (used before preempting an in-flight session).
   */
  public hasValidBookmarkForEntry(
    entry: SessionAffinityEntry,
    tools: OpenAIToolDefinition[] | undefined,
    messages: OpenAIMessage[]
  ): boolean {
    if (entry.consecutiveAborts >= MAX_CONSECUTIVE_SESSION_ABORTS) return false;
    if (entry.lastTotalTokens >= MAX_SESSION_CONTEXT_TOKENS) return false;
    if (entry.toolsSignature !== computeToolsSignature(tools)) return false;
    const nonSystem = getNonSystemMessages(messages);
    return resolveEntrySlice(entry, nonSystem) !== null;
  }

  /**
   * Peeks at candidate sessions matching `(baseModelId + firstUser + firstAssistant)`
   * (or an interrupted/in-flight Turn-1 session when incoming has 0 assistant messages)
   * without mutating state.
   */
  public findCandidateEntries(
    baseModelId: string,
    messages: OpenAIMessage[],
    isSessionRunning?: (workerId: string, sessionId: string) => boolean
  ): SessionAffinityEntry[] {
    const nonSystem = getNonSystemMessages(messages);
    const anchors = extractConversationAnchors(nonSystem);
    if (!anchors) return [];

    const now = Date.now();
    const matches: Array<{
      entry: SessionAffinityEntry;
      hasValidSlice: boolean;
    }> = [];

    for (const entry of this.sessions.values()) {
      if (now - entry.updatedAt > this.ttlMs) continue;
      if (entry.modelId !== baseModelId) continue;
      if (entry.firstUserFp !== anchors.firstUserFp) continue;

      if (anchors.hasAssistantMessages) {
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
      } else {
        // Incoming request has 0 assistant messages: ONLY match an uncommitted Turn-1 session
        // that is either awaiting recovery after cancel/interrupt or still in-flight on that session.
        const running =
          isSessionRunning?.(entry.workerId, entry.sessionId) ?? false;
        const canMatchTurn1 =
          entry.firstTextAssistantFp === null &&
          (entry.syncedTurnCount === 0 ||
            entry.hasPendingInterruptedTurn ||
            running);
        if (!canMatchTurn1) continue;
      }

      const hasValidSlice = resolveEntrySlice(entry, nonSystem) !== null;
      matches.push({ entry, hasValidSlice });
    }

    matches.sort((a, b) => {
      if (a.hasValidSlice !== b.hasValidSlice) {
        return a.hasValidSlice ? -1 : 1;
      }
      return b.entry.updatedAt - a.entry.updatedAt;
    });

    return matches.map((m) => m.entry);
  }

  /**
   * Matches an incoming OpenAI `messages` array to an active ACP session using
   * `(firstUser + firstAssistant)` anchoring + sliding `checkpoints` progress bookmarks,
   * while enforcing zero-cost session health checks:
   *   1. Consecutive error/timeout circuit breaker (`consecutiveAborts >= 2`)
   *   2. External tool definitions match (`toolsSignature` — non-matching helper requests skip without destroying)
   *   3. Context window fullness guard (`lastTotalTokens >= MAX_SESSION_CONTEXT_TOKENS`)
   *   4. Sliding progress bookmark & deep history rewind detection
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

      // Health Check 1: Consecutive error/timeout circuit breaker
      if (entry.consecutiveAborts >= MAX_CONSECUTIVE_SESSION_ABORTS) {
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          "consecutive_aborts"
        );
        continue;
      }

      // Health Check 2: External tool definitions mismatch — skip without destroying so
      // tool-less background helper requests (e.g. title generation) never nuke the main session.
      if (entry.toolsSignature !== incomingToolsSig) {
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

      // Health Check 4: Sliding progress bookmark & history rewind check
      const resolved = resolveEntrySlice(entry, nonSystem);
      if (!resolved) {
        const hasAnyDeltaCandidate =
          entry.checkpoints.length > 0 &&
          findBookmarkIndex(
            nonSystem,
            entry.lastSyncedAssistantFp,
            entry.lastSyncedAssistantTextFp
          ) ===
            nonSystem.length - 1;
        this.sessions.delete(entry.sessionId);
        cb.onInvalidateSession?.(
          entry.workerId,
          entry.sessionId,
          hasAnyDeltaCandidate ? "empty_delta" : "history_rewound"
        );
        continue;
      }

      // Refresh TTL on hit!
      entry.updatedAt = Date.now();
      return {
        entry,
        deltaMessages: resolved.deltaMessages,
        matchedPrefixLength: resolved.sliceStartIdx,
        isRetryAfterInterrupt: resolved.isFallbackOrInterrupt,
      };
    }

    return null;
  }

  /**
   * Pre-registers a brand-new ACP session immediately after `session/new` so that if
   * the user cancels Turn 1 in-flight (or a second follow-up preempts a hung Turn 1 stream),
   * the gateway can still locate, preempt, and reuse the warm Turn-1 session.
   */
  public registerInitialSession(
    baseModelId: string,
    tools: OpenAIToolDefinition[] | undefined,
    initialMessages: OpenAIMessage[],
    workerId: string,
    sessionId: string
  ): void {
    const nonSystem = getNonSystemMessages(initialMessages);
    const anchors = extractConversationAnchors(nonSystem);
    if (!anchors) return;

    const lastUser = [...nonSystem].reverse().find((m) => m.role === "user");
    const lastUserFp = lastUser ? computeUserFingerprint(lastUser) : null;

    const initialCheckpoints: SyncedTurnCheckpoint[] = [];
    for (let i = 0; i < nonSystem.length; i++) {
      const m = nonSystem[i];
      if (m.role !== "assistant") continue;
      let triggerUserFp: string | null = null;
      for (let j = i - 1; j >= 0; j--) {
        if (nonSystem[j].role === "user") {
          triggerUserFp = computeUserFingerprint(nonSystem[j]);
          break;
        }
      }
      initialCheckpoints.push({
        assistantFullFp: computeAssistantFullFingerprint(m),
        assistantTextFp: computeAssistantTextFingerprint(m),
        triggerUserFp,
      });
    }
    const trimmedCheckpoints = initialCheckpoints.slice(-MAX_SYNCED_CHECKPOINTS);
    const latestCp =
      trimmedCheckpoints.length > 0
        ? trimmedCheckpoints[trimmedCheckpoints.length - 1]
        : null;

    this.sessions.set(sessionId, {
      sessionId,
      workerId,
      modelId: baseModelId,
      toolsSignature: computeToolsSignature(tools),
      firstUserFp: anchors.firstUserFp,
      firstAnyAssistantFp: anchors.firstAnyAssistantFp,
      firstTextAssistantFp: anchors.firstTextAssistantFp,
      lastSyncedAssistantFp: latestCp?.assistantFullFp ?? null,
      lastSyncedAssistantTextFp: latestCp?.assistantTextFp ?? null,
      checkpoints: trimmedCheckpoints,
      lastAbortedAssistantFp: null,
      lastAbortedAssistantTextFp: null,
      lastInterruptedUserFp: lastUserFp,
      syncedTurnCount: trimmedCheckpoints.length,
      consecutiveAborts: 0,
      hasPendingInterruptedTurn: false,
      lastTotalTokens: 0,
      messageCount: nonSystem.length,
      updatedAt: Date.now(),
    });
  }

  /**
   * Registers or updates a live ACP session after a successful `session/prompt` turn,
   * appending to its sliding `checkpoints` window, refreshing its TTL, and resetting
   * consecutive error abort counters.
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
    let lastAsstIdx = -1;
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      if (nonSystem[i].role === "assistant") {
        lastAsstIdx = i;
        break;
      }
    }

    if (!anchors || lastAsstIdx === -1) {
      return sessionId;
    }

    const lastAsst = nonSystem[lastAsstIdx];
    let triggerUserFp: string | null = null;
    for (let j = lastAsstIdx - 1; j >= 0; j--) {
      if (nonSystem[j].role === "user") {
        triggerUserFp = computeUserFingerprint(nonSystem[j]);
        break;
      }
    }

    const existing = this.sessions.get(sessionId);
    const lastSyncedAssistantFp = computeAssistantFullFingerprint(lastAsst);
    const lastSyncedAssistantTextFp = computeAssistantTextFingerprint(lastAsst);

    const nextCheckpoint: SyncedTurnCheckpoint = {
      assistantFullFp: lastSyncedAssistantFp,
      assistantTextFp: lastSyncedAssistantTextFp,
      triggerUserFp,
    };

    const prevCheckpoints = existing?.checkpoints || [];
    const updatedCheckpoints =
      prevCheckpoints.length > 0 &&
      prevCheckpoints[prevCheckpoints.length - 1].assistantFullFp ===
        lastSyncedAssistantFp
        ? prevCheckpoints
        : [...prevCheckpoints, nextCheckpoint].slice(-MAX_SYNCED_CHECKPOINTS);

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
      checkpoints: updatedCheckpoints,
      lastAbortedAssistantFp: null,
      lastAbortedAssistantTextFp: null,
      lastInterruptedUserFp: null,
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
   * - Manual client cancellations (`isClientAbort: true`) keep the session alive and
   *   NEVER increment `consecutiveAborts` (healthy `session/cancel`);
   * - Timeouts / RPC errors (`isClientAbort: false`) increment `consecutiveAborts` and
   *   trip the 2-strike circuit breaker (`shouldDestroy: true`) when `>= 2` or if Turn 1 failed.
   */
  public recordTurnInterruption(
    sessionId: string,
    optionsOrPartial?: RecordInterruptionOptions | OpenAIMessage
  ): { shouldDestroy: boolean; consecutiveAborts: number } {
    const existing = this.sessions.get(sessionId);
    if (!existing) {
      return { shouldDestroy: true, consecutiveAborts: 1 };
    }

    const opts: RecordInterruptionOptions =
      optionsOrPartial && "role" in optionsOrPartial
        ? { partialAssistantMsg: optionsOrPartial }
        : (optionsOrPartial as RecordInterruptionOptions) || {};

    const isClientAbort = Boolean(opts.isClientAbort);
    const partialAssistantMsg = opts.partialAssistantMsg;
    const lastUserMsg = opts.lastUserMsg;

    // Only genuine errors/timeouts on an unsynced Turn 1 destroy the session;
    // manual user cancellation on Turn 1 keeps the warm session ready for retry/continue.
    if (!isClientAbort && existing.syncedTurnCount === 0) {
      this.sessions.delete(sessionId);
      return { shouldDestroy: true, consecutiveAborts: 1 };
    }

    if (!isClientAbort) {
      existing.consecutiveAborts += 1;
    }
    existing.hasPendingInterruptedTurn = true;
    existing.updatedAt = Date.now();

    if (lastUserMsg && lastUserMsg.role === "user") {
      existing.lastInterruptedUserFp = computeUserFingerprint(lastUserMsg);
    }

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
