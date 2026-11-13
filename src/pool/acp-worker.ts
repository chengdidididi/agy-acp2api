import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import readline from "node:readline";
import { stripModelThinkingSuffix } from "../compiler/model-resolver.js";
import { config } from "../config.js";
import { debugStore } from "../telemetry/debug-store.js";
import { telemetryStore } from "../telemetry/metrics-store.js";
import {
  AcpConfigOption,
  AcpModelInfo,
  AcpPromptContentBlock,
  AcpPromptResult,
  AcpSessionNewResult,
  AcpSessionUpdateParams,
  JsonRpcResponse,
} from "../types/acp.js";
import { CcpaModelSniffer } from "./ccpa-sniffer.js";
import {
  cleanupWorkerSandbox,
  createWorkerSandbox,
  syncWorkerSandboxAuth,
  WorkerSandboxEnv,
} from "./sandbox.js";

export interface PromptStreamCallbacks {
  onThoughtChunk: (text: string) => void;
  onMessageChunk: (text: string) => void;
  onNativeToolCall?: (toolName: string, rawInput: unknown) => void;
  onActualModelDetected?: (actualModelVersion: string) => void;
}

export interface ExtendedPromptExecutionResult extends AcpPromptResult {
  ttftMs: number;
  durationMs: number;
  actualModelVersion: string;
}

export class PromptInterruptedError extends Error {
  public readonly isTimeout: boolean;
  public readonly isClientAbort: boolean;
  public readonly cancelSettled: boolean;
  public readonly promptSent: boolean;

  constructor(
    message: string,
    opts: {
      isTimeout: boolean;
      isClientAbort: boolean;
      cancelSettled: boolean;
      promptSent: boolean;
    }
  ) {
    super(message);
    this.name = "PromptInterruptedError";
    this.isTimeout = opts.isTimeout;
    this.isClientAbort = opts.isClientAbort;
    this.cancelSettled = opts.cancelSettled;
    this.promptSent = opts.promptSent;
  }
}

const CANCEL_SETTLE_TIMEOUT_MS = 3000;

interface PendingRpc {
  resolve: (val: any) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
}

export type WorkerState = "STARTING" | "IDLE" | "BUSY" | "DEAD";

export type WorkerLogStream = "stdout" | "stderr" | "stdin" | "sys";

export interface WorkerLiveLogLine {
  seq: number;
  timestamp: number;
  stream: WorkerLogStream;
  sessionId: string | null;
  text: string;
}

const LIVE_LOG_RING_CAPACITY = 1000;
const MAX_LIVE_LOG_LINE_CHARS = 8192;
const PRE_SESSION_STDERR_CAPACITY = 20;

interface PendingRpcMeta {
  method: string;
  sessionId: string | null;
  rawStdinLine: string;
  timestamp: number;
}

export function sanitizePayloadForLog(
  payload: Record<string, unknown>
): Record<string, unknown> {
  if (payload.method !== "session/prompt") return payload;
  const params = payload.params as Record<string, unknown> | undefined;
  if (!params || !Array.isArray(params.prompt)) return payload;

  let changed = false;
  const sanitizedPrompt = params.prompt.map((block: any) => {
    if (
      block &&
      typeof block === "object" &&
      block.type === "image" &&
      typeof block.data === "string"
    ) {
      changed = true;
      return {
        ...block,
        data: `[base64 omitted, ${block.data.length} chars]`,
      };
    }
    return block;
  });

  if (!changed) return payload;
  return {
    ...payload,
    params: {
      ...params,
      prompt: sanitizedPrompt,
    },
  };
}

export class AcpWorker {
  public readonly id: string;
  public readonly sandbox: WorkerSandboxEnv;
  public readonly createdAt: number = Date.now();
  public state: WorkerState = "STARTING";
  public lastUsedAt: number = Date.now();
  public availableModels: AcpModelInfo[] = [];
  public totalRequestsHandled: number = 0;

  private proc: ChildProcessWithoutNullStreams | null = null;
  private sniffer: CcpaModelSniffer = new CcpaModelSniffer();
  private rpcIdCounter = 1;
  private pendingRpcs = new Map<number | string, PendingRpc>();
  private activeStreamListeners = new Map<string, PromptStreamCallbacks>();
  private activeSessions = new Set<string>();
  private cancellingSessions = new Set<string>();
  private currentPromptSessionId: string | null = null;
  private currentPromptAbortFn: (() => void) | null = null;
  private idleWaiters: Array<() => void> = [];
  private cancelListeners = new Map<string, Array<() => void>>();
  private sessionAssignedModel = new Map<string, string>();
  private sessionConfirmedConfig = new Map<
    string,
    { model: string; thoughtLevel: string }
  >();
  private logRing: WorkerLiveLogLine[] = [];
  private logSeqCounter = 0;
  private pendingRpcMeta = new Map<number | string, PendingRpcMeta>();
  private lastActiveSessionId: string | null = null;
  private preSessionStderr: Array<{ timestamp: number; line: string }> = [];
  private onDeathCallback?: (workerId: string) => void;

  constructor(id: string, onDeath?: (workerId: string) => void) {
    this.id = id;
    this.sandbox = createWorkerSandbox(id);
    this.onDeathCallback = onDeath;
  }

  public async start(): Promise<void> {
    const snifferPort = await this.sniffer.start((detectedVer) => {
      const cleanVer = stripModelThinkingSuffix(detectedVer);
      for (const listener of this.activeStreamListeners.values()) {
        listener.onActualModelDetected?.(cleanVer);
      }
    });

    const args = [
      "acp",
      "--sandbox",
      "--include-default-customizations=false",
      `--gemini-dir=${this.sandbox.absGeminiDir}`,
      `--agent=${this.sandbox.agentName}`,
      `--cloud_code_endpoint=http://127.0.0.1:${snifferPort}`,
    ];

    const childEnv: NodeJS.ProcessEnv = {
      ...process.env,
      ...(process.platform === "linux"
        ? { SSH_CONNECTION: process.env.SSH_CONNECTION || "127.0.0.1 0 127.0.0.1 22" }
        : {}),
    };

    this.pushLiveLog(
      "sys",
      null,
      `[spawn] ${config.agyBin} ${args.join(" ")}`
    );

    this.proc = spawn(config.agyBin, args, {
      cwd: this.sandbox.jailCwd,
      env: childEnv,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    this.proc.on("exit", (code, signal) => {
      const exitMsg = `[exit] Worker ${this.id} exited (code=${code ?? "null"}, signal=${signal ?? "null"})`;
      this.pushLiveLog("sys", this.currentPromptSessionId, exitMsg);
      const isErrExit = (code !== null && code !== 0) || Boolean(signal);
      if (this.state !== "DEAD") {
        for (const sid of this.activeSessions) {
          debugStore.recordEvent({
            sessionId: sid,
            workerId: this.id,
            direction: "sys",
            category: isErrExit ? "error" : "sys",
            method: "process/exit",
            summary: exitMsg,
            content: exitMsg,
            rawFrame: JSON.stringify({ event: "exit", code, signal }),
            sessionStatusOnCreate: "CLOSED",
          });
        }
      }
      this.handleProcessDeath(
        new Error(`Worker ${this.id} process exited unexpectedly`)
      );
    });

    this.proc.on("error", (err) => {
      this.pushLiveLog("sys", this.currentPromptSessionId, `[error] ${err.message}`);
      if (this.state !== "DEAD") {
        for (const sid of this.activeSessions) {
          debugStore.recordEvent({
            sessionId: sid,
            workerId: this.id,
            direction: "sys",
            category: "error",
            method: "process/error",
            summary: err.message,
            content: err.stack || err.message,
            rawFrame: JSON.stringify({ event: "error", message: err.message }),
            sessionStatusOnCreate: "CLOSED",
          });
        }
      }
      this.handleProcessDeath(err);
    });

    const rl = readline.createInterface({
      input: this.proc.stdout,
      crlfDelay: Infinity,
    });

    rl.on("line", (line) => {
      this.handleStdoutLine(line);
    });

    const errRl = readline.createInterface({
      input: this.proc.stderr,
      crlfDelay: Infinity,
    });

    errRl.on("line", (line) => {
      this.handleStderrLine(line);
    });

    // Perform ACP JSON-RPC handshake
    await this.sendRpc("initialize", {
      protocolVersion: 1,
      clientCapabilities: {},
    });

    this.state = "IDLE";
    this.lastUsedAt = Date.now();
  }

  public getPid(): number | null {
    return this.proc?.pid ?? null;
  }

  public getSnifferPort(): number {
    return this.sniffer.getPort();
  }

  public getLiveLogs(afterSeq: number = 0, limit: number = 500): {
    lines: WorkerLiveLogLine[];
    lastSeq: number;
    totalLines: number;
  } {
    const safeLimit = Math.max(1, Math.min(LIVE_LOG_RING_CAPACITY, limit));
    const filtered =
      afterSeq > 0
        ? this.logRing.filter((l) => l.seq > afterSeq)
        : this.logRing.slice(-safeLimit);
    const lines = filtered.length > safeLimit ? filtered.slice(-safeLimit) : filtered;
    return {
      lines,
      lastSeq: this.logSeqCounter,
      totalLines: this.logRing.length,
    };
  }

  public clearLiveLogs(): void {
    this.logRing = [];
  }

  public getLiveLogCount(): number {
    return this.logRing.length;
  }

  public getLastLogSeq(): number {
    return this.logSeqCounter;
  }

  public snapshotLiveLogs(): WorkerLiveLogLine[] {
    return [...this.logRing];
  }

  public getActiveSessionIds(): string[] {
    return Array.from(this.activeSessions);
  }

  public hasSession(sessionId: string): boolean {
    return this.activeSessions.has(sessionId);
  }

  public isCancellingSession(sessionId: string): boolean {
    return this.cancellingSessions.has(sessionId);
  }

  public getCurrentPromptSessionId(): string | null {
    return this.currentPromptSessionId;
  }

  /**
   * Actively cancels an in-flight `runPrompt` on `sessionId` when a newer request for
   * the exact same conversation arrives before a downstream proxy/client closed the old HTTP socket.
   */
  public preemptSession(sessionId: string): boolean {
    if (
      this.state !== "BUSY" ||
      this.currentPromptSessionId !== sessionId ||
      !this.currentPromptAbortFn
    ) {
      return false;
    }
    this.currentPromptAbortFn();
    return true;
  }

  public hasIdleWaiters(): boolean {
    return this.idleWaiters.length > 0;
  }

  public notifyIdle(): void {
    const waiters = this.idleWaiters.splice(0, this.idleWaiters.length);
    for (const fn of waiters) {
      fn();
    }
  }

  private notifySessionCancelling(sessionId: string): void {
    const list = this.cancelListeners.get(sessionId);
    if (!list) return;
    this.cancelListeners.delete(sessionId);
    for (const fn of list) {
      fn();
    }
  }

  /**
   * Waits for this worker to return to `IDLE` when an incoming retry request matches
   * a session on this worker that is currently settling `session/cancel` (or about to
   * receive the client's TCP abort within `raceGraceMs`).
   */
  public waitForSessionSettle(
    sessionId: string,
    cancelSettleMs: number = 3500,
    raceGraceMs: number = 1000
  ): Promise<boolean> {
    if (this.state === "IDLE") {
      return Promise.resolve(this.hasSession(sessionId));
    }
    if (this.state === "DEAD" || !this.hasSession(sessionId)) {
      return Promise.resolve(false);
    }

    return new Promise<boolean>((resolve) => {
      let done = false;
      let timer: NodeJS.Timeout | null = null;

      const finish = () => {
        if (done) return;
        done = true;
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        const idleIdx = this.idleWaiters.indexOf(onIdle);
        if (idleIdx !== -1) {
          this.idleWaiters.splice(idleIdx, 1);
        }
        const cList = this.cancelListeners.get(sessionId);
        if (cList) {
          const cIdx = cList.indexOf(onCancelStart);
          if (cIdx !== -1) cList.splice(cIdx, 1);
          if (cList.length === 0) this.cancelListeners.delete(sessionId);
        }
        resolve(this.state === "IDLE" && this.hasSession(sessionId));
      };

      const onIdle = () => {
        finish();
      };

      const onCancelStart = () => {
        if (done) return;
        if (timer) clearTimeout(timer);
        timer = setTimeout(finish, cancelSettleMs);
        timer.unref?.();
      };

      this.idleWaiters.push(onIdle);

      if (this.cancellingSessions.has(sessionId)) {
        timer = setTimeout(finish, cancelSettleMs);
        timer.unref?.();
      } else {
        const list = this.cancelListeners.get(sessionId) || [];
        list.push(onCancelStart);
        this.cancelListeners.set(sessionId, list);
        timer = setTimeout(finish, raceGraceMs);
        timer.unref?.();
      }
    });
  }

  /**
   * Creates a new ACP session on this worker in native `default` mode (no plan/build switching),
   * sets the target `acpModelId`, and injects `<system_override>` + external tool schemas.
   */
  public async createSession(
    systemPromptAppend: string,
    acpModelId: string
  ): Promise<string> {
    syncWorkerSandboxAuth(this.sandbox.absAppDataDir);
    const res = await this.sendRpc<AcpSessionNewResult>("session/new", {
      cwd: this.sandbox.jailCwd,
      mcpServers: [],
      _meta: {
        antigravity: {
          systemPrompt: {
            append: systemPromptAppend,
          },
        },
      },
    });

    const sessionId = res.sessionId;
    this.activeSessions.add(sessionId);
    this.lastActiveSessionId = sessionId;
    debugStore.registerSession(sessionId, this.id, "ACTIVE");

    if (res.models?.availableModels?.length) {
      this.availableModels = res.models.availableModels;
    }

    if (res.configOptions) {
      this.updateConfirmedConfigFromOptions(sessionId, res.configOptions);
    }

    await this.ensureSessionModel(sessionId, acpModelId);

    return sessionId;
  }

  /**
   * Ensures the session is configured to use `acpModelId`.
   * If a reused session requests a different thinking effort variant (e.g. `low` -> `high`),
   * calls `session/set_model` dynamically on the existing session.
   */
  public async ensureSessionModel(
    sessionId: string,
    acpModelId: string
  ): Promise<void> {
    if (!acpModelId) return;
    if (this.sessionAssignedModel.get(sessionId) === acpModelId) {
      return;
    }

    try {
      await this.sendRpc("session/set_model", {
        sessionId,
        modelId: acpModelId,
      });
      this.sessionAssignedModel.set(sessionId, acpModelId);
    } catch {
      // Non-fatal if modelId falls back to default
    }
  }

  /**
   * Executes a `session/prompt` turn on an existing `sessionId`, measuring TTFT,
   * capturing actual responding modelVersion via the local CCPA sniffer + ACP config updates,
   * and cleanly settling `session/cancel` on both client aborts and gateway timeouts.
   */
  public async runPrompt(
    sessionId: string,
    promptInput: string | AcpPromptContentBlock[],
    acpModelId: string,
    callbacks: PromptStreamCallbacks,
    abortSignal?: AbortSignal
  ): Promise<ExtendedPromptExecutionResult> {
    if (abortSignal?.aborted) {
      throw new PromptInterruptedError("Request aborted before prompt started", {
        isTimeout: false,
        isClientAbort: true,
        cancelSettled: true,
        promptSent: false,
      });
    }

    this.currentPromptSessionId = sessionId;
    let preemptedBeforePrompt = false;

    const onPrePromptAbort = () => {
      preemptedBeforePrompt = true;
      this.cancellingSessions.add(sessionId);
      this.notifySessionCancelling(sessionId);
    };
    this.currentPromptAbortFn = onPrePromptAbort;
    if (abortSignal) {
      abortSignal.addEventListener("abort", onPrePromptAbort, { once: true });
    }
    try {
      await this.ensureSessionModel(sessionId, acpModelId);
    } finally {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", onPrePromptAbort);
      }
    }

    if (abortSignal?.aborted || preemptedBeforePrompt) {
      this.currentPromptSessionId = null;
      this.currentPromptAbortFn = null;
      this.cancellingSessions.delete(sessionId);
      throw new PromptInterruptedError("Request aborted before prompt started", {
        isTimeout: false,
        isClientAbort: true,
        cancelSettled: true,
        promptSent: false,
      });
    }

    if (this.state === "DEAD" || !this.proc?.stdin.writable) {
      this.currentPromptSessionId = null;
      this.currentPromptAbortFn = null;
      throw new PromptInterruptedError(`Worker ${this.id} is not writable`, {
        isTimeout: false,
        isClientAbort: false,
        cancelSettled: false,
        promptSent: false,
      });
    }

    this.sniffer.resetTurnModel();
    const startTime = Date.now();
    let firstTokenTime: number | null = null;

    const wrappedCallbacks: PromptStreamCallbacks = {
      onThoughtChunk: (text) => {
        if (firstTokenTime === null) {
          firstTokenTime = Date.now();
        }
        callbacks.onThoughtChunk(text);
      },
      onMessageChunk: (text) => {
        if (firstTokenTime === null) {
          firstTokenTime = Date.now();
        }
        callbacks.onMessageChunk(text);
      },
      onNativeToolCall: callbacks.onNativeToolCall,
      onActualModelDetected: callbacks.onActualModelDetected,
    };

    this.activeStreamListeners.set(sessionId, wrappedCallbacks);

    const rpcId = this.rpcIdCounter++;
    const timeoutMs = config.requestTimeoutMs;

    try {
      const result = await new Promise<AcpPromptResult>((resolve, reject) => {
        let settled = false;
        let interruptReason: "abort" | "timeout" | null = null;
        let cancelSettleTimer: NodeJS.Timeout | null = null;

        const cleanup = () => {
          clearTimeout(requestTimer);
          if (cancelSettleTimer) {
            clearTimeout(cancelSettleTimer);
            cancelSettleTimer = null;
          }
          if (abortSignal) {
            abortSignal.removeEventListener("abort", onAbort);
          }
        };

        const markCancelStart = (reason: "abort" | "timeout") => {
          if (settled || interruptReason !== null) return;
          interruptReason = reason;
          clearTimeout(requestTimer);

          // Detach stream callbacks immediately so post-cancel chunks do not leak
          this.activeStreamListeners.delete(sessionId);

          this.cancellingSessions.add(sessionId);
          this.notifySessionCancelling(sessionId);
          this.sendNotification("session/cancel", { sessionId });

          cancelSettleTimer = setTimeout(() => {
            if (settled) return;
            settled = true;
            cleanup();
            this.pendingRpcs.delete(rpcId);
            this.cancellingSessions.delete(sessionId);

            const msg =
              reason === "timeout"
                ? `ACP RPC 'session/prompt' timed out after ${timeoutMs}ms and session/cancel did not settle within ${CANCEL_SETTLE_TIMEOUT_MS}ms`
                : `Request aborted by client and ACP session/cancel did not settle within ${CANCEL_SETTLE_TIMEOUT_MS}ms`;

            reject(
              new PromptInterruptedError(msg, {
                isTimeout: reason === "timeout",
                isClientAbort: reason === "abort",
                cancelSettled: false,
                promptSent: true,
              })
            );
          }, CANCEL_SETTLE_TIMEOUT_MS);
        };

        const requestTimer = setTimeout(() => {
          markCancelStart("timeout");
        }, timeoutMs);

        const onAbort = () => {
          markCancelStart("abort");
        };
        this.currentPromptAbortFn = onAbort;

        if (abortSignal && !abortSignal.aborted) {
          abortSignal.addEventListener("abort", onAbort, { once: true });
        }

        this.pendingRpcs.set(rpcId, {
          timer: requestTimer,
          resolve: (rpcResult: AcpPromptResult) => {
            if (settled) return;
            settled = true;
            cleanup();
            this.cancellingSessions.delete(sessionId);

            if (interruptReason !== null) {
              const msg =
                interruptReason === "timeout"
                  ? `ACP RPC 'session/prompt' timed out after ${timeoutMs}ms`
                  : "Request aborted by client";
              reject(
                new PromptInterruptedError(msg, {
                  isTimeout: interruptReason === "timeout",
                  isClientAbort: interruptReason === "abort",
                  cancelSettled: true,
                  promptSent: true,
                })
              );
              return;
            }

            resolve(rpcResult);
          },
          reject: (rpcErr: Error) => {
            if (settled) return;
            settled = true;
            cleanup();
            this.cancellingSessions.delete(sessionId);

            const workerAlive = this.state !== "DEAD";
            if (interruptReason !== null) {
              const msg =
                interruptReason === "timeout"
                  ? `ACP RPC 'session/prompt' timed out after ${timeoutMs}ms`
                  : "Request aborted by client";
              reject(
                new PromptInterruptedError(msg, {
                  isTimeout: interruptReason === "timeout",
                  isClientAbort: interruptReason === "abort",
                  cancelSettled: workerAlive,
                  promptSent: true,
                })
              );
              return;
            }

            reject(rpcErr);
          },
        });

        const promptBlocks: AcpPromptContentBlock[] =
          typeof promptInput === "string"
            ? [{ type: "text", text: promptInput }]
            : promptInput.length > 0
              ? promptInput
              : [{ type: "text", text: "" }];

        this.writeJson({
          jsonrpc: "2.0",
          id: rpcId,
          method: "session/prompt",
          params: {
            sessionId,
            prompt: promptBlocks,
          },
        });

        if (abortSignal?.aborted) {
          onAbort();
        }
      });

      const endTime = Date.now();
      const durationMs = endTime - startTime;
      const ttftMs =
        firstTokenTime !== null ? firstTokenTime - startTime : durationMs;

      this.totalRequestsHandled++;
      const actualModelVersion = this.resolveActualModelString(
        sessionId,
        acpModelId
      );

      return {
        ...result,
        ttftMs,
        durationMs,
        actualModelVersion,
      };
    } finally {
      this.currentPromptSessionId = null;
      this.currentPromptAbortFn = null;
      this.activeStreamListeners.delete(sessionId);
      this.lastUsedAt = Date.now();
    }
  }

  public resolveActualModelString(
    sessionId: string,
    fallbackAcpModelId: string
  ): string {
    const upstreamModel = this.sniffer.getDetectedModelVersion();
    const confirmed = this.sessionConfirmedConfig.get(sessionId);

    if (upstreamModel && upstreamModel !== "gemini-3.5-flash-lite") {
      return stripModelThinkingSuffix(upstreamModel);
    }

    if (confirmed?.model) {
      return stripModelThinkingSuffix(confirmed.model);
    }

    return stripModelThinkingSuffix(upstreamModel || fallbackAcpModelId);
  }

  public async closeSession(sessionId: string): Promise<void> {
    if (!this.activeSessions.has(sessionId)) return;
    this.activeSessions.delete(sessionId);
    this.cancellingSessions.delete(sessionId);
    this.cancelListeners.delete(sessionId);
    this.sessionAssignedModel.delete(sessionId);
    this.sessionConfirmedConfig.delete(sessionId);
    if (this.lastActiveSessionId === sessionId) {
      this.lastActiveSessionId = this.getActiveSessionIds().pop() ?? null;
    }
    telemetryStore.markSessionClosed(sessionId);
    try {
      await this.sendRpc("session/close", { sessionId }, 5000);
    } catch {}
    debugStore.markSessionClosed(sessionId, config.logRetentionMs);
  }

  public destroy(): void {
    this.state = "DEAD";
    this.pushLiveLog("sys", this.currentPromptSessionId, `[destroy] Worker ${this.id} destroyed`);
    telemetryStore.markWorkerSessionsClosed(this.id);
    for (const [, pending] of this.pendingRpcs) {
      clearTimeout(pending.timer);
      pending.reject(new Error(`Worker ${this.id} destroyed`));
    }
    this.pendingRpcs.clear();
    this.pendingRpcMeta.clear();
    this.currentPromptSessionId = null;
    this.currentPromptAbortFn = null;
    this.activeStreamListeners.clear();
    this.activeSessions.clear();
    this.cancellingSessions.clear();
    this.cancelListeners.clear();
    this.sessionAssignedModel.clear();
    this.sessionConfirmedConfig.clear();
    this.lastActiveSessionId = null;
    this.notifyIdle();

    this.sniffer.stop();

    if (this.proc) {
      try {
        this.proc.kill("SIGKILL");
      } catch {}
      this.proc = null;
    }

    cleanupWorkerSandbox(this.sandbox);
  }

  private updateConfirmedConfigFromOptions(
    sessionId: string,
    options: AcpConfigOption[]
  ): void {
    const current = this.sessionConfirmedConfig.get(sessionId) || {
      model: "",
      thoughtLevel: "",
    };
    for (const opt of options) {
      if (opt.id === "model" && opt.currentValue) {
        current.model = opt.currentValue;
      } else if (opt.id === "thought_level" && opt.currentValue) {
        current.thoughtLevel = opt.currentValue;
      }
    }
    this.sessionConfirmedConfig.set(sessionId, current);
  }

  private pushLiveLog(
    stream: WorkerLogStream,
    sessionId: string | null,
    text: string,
    timestamp: number = Date.now()
  ): void {
    const capped =
      text.length > MAX_LIVE_LOG_LINE_CHARS
        ? text.slice(0, MAX_LIVE_LOG_LINE_CHARS) + "…[truncated]"
        : text;
    const entry: WorkerLiveLogLine = {
      seq: ++this.logSeqCounter,
      timestamp,
      stream,
      sessionId,
      text: capped,
    };
    this.logRing.push(entry);
    if (this.logRing.length > LIVE_LOG_RING_CAPACITY) {
      this.logRing.splice(0, this.logRing.length - LIVE_LOG_RING_CAPACITY);
    }
  }

  private getTargetSessionForWorkerOutput(): string | null {
    if (this.currentPromptSessionId) return this.currentPromptSessionId;
    if (this.lastActiveSessionId && this.activeSessions.has(this.lastActiveSessionId)) {
      return this.lastActiveSessionId;
    }
    return null;
  }

  private handleStderrLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;
    const now = Date.now();
    const sid = this.getTargetSessionForWorkerOutput();
    this.pushLiveLog("stderr", sid, line, now);

    if (sid) {
      debugStore.recordEvent({
        sessionId: sid,
        workerId: this.id,
        direction: "stderr",
        category: "stderr",
        method: "stderr",
        summary: trimmed,
        content: line,
        rawFrame: line,
        timestamp: now,
      });
    } else {
      this.preSessionStderr.push({ timestamp: now, line });
      if (this.preSessionStderr.length > PRE_SESSION_STDERR_CAPACITY) {
        this.preSessionStderr.shift();
      }
    }
  }

  private handleStdoutLine(line: string): void {
    const now = Date.now();
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) {
      const sid = this.getTargetSessionForWorkerOutput();
      if (trimmed.length > 0) {
        this.pushLiveLog("stdout", sid, line, now);
        if (sid) {
          debugStore.recordEvent({
            sessionId: sid,
            workerId: this.id,
            direction: "stdout",
            category: "stdout_raw",
            method: "stdout",
            summary: trimmed,
            content: line,
            rawFrame: line,
            timestamp: now,
          });
        }
      }
      return;
    }

    let msg: Record<string, any>;
    try {
      msg = JSON.parse(trimmed);
    } catch {
      const sid = this.getTargetSessionForWorkerOutput();
      this.pushLiveLog("stdout", sid, line, now);
      if (sid) {
        debugStore.recordEvent({
          sessionId: sid,
          workerId: this.id,
          direction: "stdout",
          category: "stdout_raw",
          method: "stdout/invalid_json",
          summary: trimmed,
          content: line,
          rawFrame: line,
          timestamp: now,
        });
      }
      return;
    }

    const msgSessionId: string | null =
      (typeof msg.params?.sessionId === "string" ? msg.params.sessionId : null) ??
      (msg.id !== undefined ? this.pendingRpcMeta.get(msg.id)?.sessionId ?? null : null) ??
      (typeof msg.result?.sessionId === "string" ? msg.result.sessionId : null) ??
      this.currentPromptSessionId;

    this.pushLiveLog("stdout", msgSessionId, trimmed, now);

    // 1. Handle incoming JSON-RPC Requests from ACP Server (`session/request_permission`)
    if (typeof msg.method === "string" && msg.id !== undefined) {
      this.handleServerRequest(msg.id, msg.method, msg.params || {}, trimmed, now);
      return;
    }

    // 2. Handle incoming JSON-RPC Notifications (`session/update`)
    if (msg.method === "session/update" && msg.params) {
      this.handleSessionUpdate(msg.params as AcpSessionUpdateParams, trimmed, now);
      return;
    }

    // 3. Handle JSON-RPC Responses to our requests
    if (msg.id !== undefined) {
      const meta = this.pendingRpcMeta.get(msg.id);
      this.pendingRpcMeta.delete(msg.id);

      const pending = this.pendingRpcs.get(msg.id);
      if (!pending) return;
      this.pendingRpcs.delete(msg.id);
      clearTimeout(pending.timer);

      const rpcRes = msg as JsonRpcResponse;
      if (meta) {
        this.recordRpcResponseDebug(msg.id, meta, rpcRes, trimmed, now);
      }

      if (rpcRes.error) {
        pending.reject(
          new Error(
            `ACP RPC Error [${rpcRes.error.code}]: ${rpcRes.error.message}`
          )
        );
      } else {
        pending.resolve(rpcRes.result);
      }
    }
  }

  private recordRpcResponseDebug(
    rpcId: number | string,
    meta: PendingRpcMeta,
    rpcRes: JsonRpcResponse,
    rawFrame: string,
    timestamp: number
  ): void {
    if (meta.method === "initialize") {
      return;
    }

    if (meta.method === "session/new") {
      const resultObj = rpcRes.result as Record<string, any> | undefined;
      const newSid =
        typeof resultObj?.sessionId === "string" ? resultObj.sessionId : null;
      const targetSid = newSid || `failed-new-${this.id}-${rpcId}`;
      const statusOnCreate: "ACTIVE" | "CLOSED" = newSid ? "ACTIVE" : "CLOSED";

      debugStore.registerSession(targetSid, this.id, statusOnCreate);

      // Replay any pre-session stderr captured right before/during this session creation
      const stderrLines = this.preSessionStderr.splice(0, this.preSessionStderr.length);
      for (const s of stderrLines) {
        debugStore.recordEvent({
          sessionId: targetSid,
          workerId: this.id,
          direction: "stderr",
          category: "stderr",
          method: "stderr",
          summary: s.line.trim() || "(stderr)",
          content: s.line,
          rawFrame: s.line,
          timestamp: s.timestamp,
          sessionStatusOnCreate: statusOnCreate,
        });
      }

      debugStore.recordEvent({
        sessionId: targetSid,
        workerId: this.id,
        direction: "stdin",
        category: "rpc_req",
        method: "session/new",
        summary: "session/new -> create ACP session",
        content: meta.rawStdinLine,
        rawFrame: meta.rawStdinLine,
        timestamp: meta.timestamp,
        sessionStatusOnCreate: statusOnCreate,
      });

      if (rpcRes.error) {
        debugStore.recordEvent({
          sessionId: targetSid,
          workerId: this.id,
          direction: "stdout",
          category: "error",
          method: "session/new",
          summary: `RPC Error [${rpcRes.error.code}]: ${rpcRes.error.message}`,
          content: JSON.stringify(rpcRes.error, null, 2),
          rawFrame,
          timestamp,
          sessionStatusOnCreate: "CLOSED",
        });
      } else {
        debugStore.recordEvent({
          sessionId: targetSid,
          workerId: this.id,
          direction: "stdout",
          category: "rpc_res",
          method: "session/new",
          summary: `sessionId=${targetSid}`,
          content: JSON.stringify(rpcRes.result ?? {}, null, 2),
          rawFrame,
          timestamp,
          sessionStatusOnCreate: "ACTIVE",
        });
      }
      return;
    }

    const sid = meta.sessionId;
    if (!sid) return;
    const statusOnCreate: "ACTIVE" | "CLOSED" = this.activeSessions.has(sid)
      ? "ACTIVE"
      : "CLOSED";

    if (rpcRes.error) {
      debugStore.recordEvent({
        sessionId: sid,
        workerId: this.id,
        direction: "stdout",
        category: "error",
        method: meta.method,
        summary: `RPC Error [${rpcRes.error.code}]: ${rpcRes.error.message}`,
        content: JSON.stringify(rpcRes.error, null, 2),
        rawFrame,
        timestamp,
        sessionStatusOnCreate: statusOnCreate,
      });
    } else {
      const resultObj = (rpcRes.result ?? {}) as Record<string, any>;
      let summary = `${meta.method} ok`;
      if (meta.method === "session/prompt" && resultObj.stopReason) {
        const total = resultObj.usage?.totalTokens ?? 0;
        summary = `stop=${resultObj.stopReason}${total ? ` · tokens=${total}` : ""}`;
      }
      debugStore.recordEvent({
        sessionId: sid,
        workerId: this.id,
        direction: "stdout",
        category: "rpc_res",
        method: meta.method,
        summary,
        content: JSON.stringify(resultObj, null, 2),
        rawFrame,
        timestamp,
        sessionStatusOnCreate: statusOnCreate,
      });
    }
  }

  private handleServerRequest(
    id: number | string,
    method: string,
    params: Record<string, any>,
    rawFrame: string,
    timestamp: number
  ): void {
    const sid = typeof params?.sessionId === "string" ? params.sessionId : null;
    if (sid) {
      debugStore.recordEvent({
        sessionId: sid,
        workerId: this.id,
        direction: "stdout",
        category: "rpc_srv",
        method,
        summary:
          method === "session/request_permission"
            ? `permission request (rejected by sandbox)`
            : `server request: ${method}`,
        content: JSON.stringify(params, null, 2),
        rawFrame,
        timestamp,
      });
    }

    if (method === "session/request_permission") {
      const options: Array<{ optionId: string; kind?: string }> =
        params?.options || [];
      const rejectOpt = options.find(
        (o) =>
          o.kind === "reject_once" ||
          o.kind === "reject_always" ||
          o.optionId?.toLowerCase().includes("reject")
      );

      const resultPayload = rejectOpt
        ? { outcome: { outcome: "selected", optionId: rejectOpt.optionId } }
        : { outcome: { outcome: "cancelled" } };

      this.writeJson(
        {
          jsonrpc: "2.0",
          id,
          result: resultPayload,
        },
        sid
      );

      if (params?.sessionId) {
        this.sendNotification("session/cancel", {
          sessionId: params.sessionId,
        });
      }
      return;
    }

    this.writeJson(
      {
        jsonrpc: "2.0",
        id,
        error: {
          code: -32601,
          message: `Client method '${method}' is disabled by agy2api read-only sandbox.`,
        },
      },
      sid
    );
  }

  private handleSessionUpdate(
    params: AcpSessionUpdateParams,
    rawFrame: string,
    timestamp: number
  ): void {
    if (!params.update) return;
    const { sessionUpdate, content, configOptions } = params.update;

    if (sessionUpdate === "config_option_update" && Array.isArray(configOptions)) {
      this.updateConfirmedConfigFromOptions(params.sessionId, configOptions);
    }

    if (params.sessionId) {
      if (sessionUpdate === "agent_thought_chunk" && content?.text) {
        debugStore.appendStreamChunk({
          sessionId: params.sessionId,
          workerId: this.id,
          category: "thought",
          method: "session/update:agent_thought_chunk",
          text: content.text,
          rawFrame,
          timestamp,
        });
      } else if (sessionUpdate === "agent_message_chunk" && content?.text) {
        debugStore.appendStreamChunk({
          sessionId: params.sessionId,
          workerId: this.id,
          category: "message",
          method: "session/update:agent_message_chunk",
          text: content.text,
          rawFrame,
          timestamp,
        });
      } else if (
        sessionUpdate === "tool_call" ||
        sessionUpdate === "tool_call_update"
      ) {
        const title =
          params.update.title || params.update.kind || params.update.toolCallId || "tool";
        const status = params.update.status ? ` [${params.update.status}]` : "";
        debugStore.recordEvent({
          sessionId: params.sessionId,
          workerId: this.id,
          direction: "stdout",
          category: "tool_call",
          method: `session/update:${sessionUpdate}`,
          summary: `${title}${status}`,
          content: JSON.stringify(params.update, null, 2),
          rawFrame,
          timestamp,
        });
      } else if (sessionUpdate === "config_option_update") {
        const summary = Array.isArray(configOptions)
          ? configOptions.map((o) => `${o.id}=${o.currentValue}`).join(", ")
          : "config_option_update";
        debugStore.recordEvent({
          sessionId: params.sessionId,
          workerId: this.id,
          direction: "stdout",
          category: "config",
          method: "session/update:config_option_update",
          summary,
          content: JSON.stringify(params.update, null, 2),
          rawFrame,
          timestamp,
        });
      } else {
        debugStore.recordEvent({
          sessionId: params.sessionId,
          workerId: this.id,
          direction: "stdout",
          category: "stdout_raw",
          method: `session/update:${sessionUpdate}`,
          summary:
            content?.text ||
            params.update.title ||
            sessionUpdate,
          content: JSON.stringify(params.update, null, 2),
          rawFrame,
          timestamp,
        });
      }
    }

    const listener = this.activeStreamListeners.get(params.sessionId);
    if (!listener) return;

    if (sessionUpdate === "agent_thought_chunk" && content?.text) {
      listener.onThoughtChunk(content.text);
    } else if (sessionUpdate === "agent_message_chunk" && content?.text) {
      listener.onMessageChunk(content.text);
    } else if (sessionUpdate === "tool_call") {
      const title =
        params.update.title || params.update.kind || "unknown_native_tool";
      listener.onNativeToolCall?.(title, params.update.rawInput);
    }
  }

  private sendRpc<T = any>(
    method: string,
    params: Record<string, unknown>,
    timeoutMs: number = 60_000
  ): Promise<T> {
    if (this.state === "DEAD" || !this.proc?.stdin.writable) {
      return Promise.reject(new Error(`Worker ${this.id} is not writable`));
    }

    const id = this.rpcIdCounter++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRpcs.delete(id);
        this.pendingRpcMeta.delete(id);
        reject(new Error(`ACP RPC '${method}' timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingRpcs.set(id, { resolve, reject, timer });
      this.writeJson({
        jsonrpc: "2.0",
        id,
        method,
        params,
      });
    });
  }

  private sendNotification(
    method: string,
    params: Record<string, unknown>
  ): void {
    this.writeJson({
      jsonrpc: "2.0",
      method,
      params,
    });
  }

  private writeJson(
    payload: Record<string, unknown>,
    explicitSessionId?: string | null
  ): void {
    const now = Date.now();
    const sanitized = sanitizePayloadForLog(payload);
    const sanitizedLine = JSON.stringify(sanitized);
    const params = sanitized.params as Record<string, unknown> | undefined;
    const sid =
      explicitSessionId ??
      (typeof params?.sessionId === "string" ? params.sessionId : null);
    const method = typeof sanitized.method === "string" ? sanitized.method : "";
    const id = sanitized.id as number | string | undefined;

    this.pushLiveLog("stdin", sid, sanitizedLine, now);

    if (id !== undefined && method) {
      this.pendingRpcMeta.set(id, {
        method,
        sessionId: sid,
        rawStdinLine: sanitizedLine,
        timestamp: now,
      });
    }

    // Record outgoing RPC request/notification (defer session/new until sessionId is known)
    if (sid && method && method !== "session/new") {
      let summary = method;
      if (method === "session/set_model" && params?.modelId) {
        summary = `set_model -> ${String(params.modelId)}`;
      } else if (method === "session/prompt" && Array.isArray(params?.prompt)) {
        const blocks = params.prompt as Array<Record<string, any>>;
        const textPreview = blocks
          .filter((b) => b?.type === "text" && typeof b.text === "string")
          .map((b) => b.text)
          .join(" ")
          .replace(/\s+/g, " ")
          .trim();
        const imgCount = blocks.filter((b) => b?.type === "image").length;
        const parts: string[] = [];
        if (textPreview) parts.push(textPreview.slice(0, 100));
        if (imgCount > 0) parts.push(`[${imgCount} image${imgCount > 1 ? "s" : ""}]`);
        summary = parts.join(" ") || "session/prompt";
      }
      debugStore.recordEvent({
        sessionId: sid,
        workerId: this.id,
        direction: "stdin",
        category: "rpc_req",
        method,
        summary,
        content: JSON.stringify(sanitized, null, 2),
        rawFrame: sanitizedLine,
        timestamp: now,
        sessionStatusOnCreate: this.activeSessions.has(sid) ? "ACTIVE" : "CLOSED",
      });
    }

    if (this.proc?.stdin.writable) {
      this.proc.stdin.write(JSON.stringify(payload) + "\n");
    }
  }

  private handleProcessDeath(err: Error): void {
    if (this.state === "DEAD") return;
    this.state = "DEAD";
    telemetryStore.markWorkerSessionsClosed(this.id);
    for (const [, pending] of this.pendingRpcs) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pendingRpcs.clear();
    this.pendingRpcMeta.clear();
    this.currentPromptSessionId = null;
    this.currentPromptAbortFn = null;
    this.activeStreamListeners.clear();
    this.activeSessions.clear();
    this.cancellingSessions.clear();
    this.cancelListeners.clear();
    this.sessionAssignedModel.clear();
    this.sessionConfirmedConfig.clear();
    this.lastActiveSessionId = null;
    this.notifyIdle();
    this.sniffer.stop();
    cleanupWorkerSandbox(this.sandbox);
    this.onDeathCallback?.(this.id);
  }
}
