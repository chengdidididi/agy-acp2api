import crypto from "node:crypto";
import { config, onConfigChange } from "../config.js";
import { SessionAffinityRegistry } from "../compiler/session-hash.js";
import { AcpModelInfo } from "../types/acp.js";
import { OpenAIMessage, OpenAIToolDefinition } from "../types/openai.js";
import { AcpWorker, PromptInterruptedError } from "./acp-worker.js";
import { sweepStaleSandboxes } from "./sandbox.js";

export interface LeaseAcquisition {
  worker: AcpWorker;
  sessionId: string | null;
  isReusedSession: boolean;
  deltaMessages: OpenAIMessage[];
  matchedPrefixLength: number;
  isRetryAfterInterrupt: boolean;
}

interface QueuedWaiter {
  resolve: (worker: AcpWorker) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
}

const DEFAULT_FALLBACK_MODELS: AcpModelInfo[] = [
  { modelId: "gemini-4-carbon-high", name: "Gemini 4 Carbon (High)" },
  { modelId: "gemini-4-carbon-medium", name: "Gemini 4 Carbon (Medium)" },
  { modelId: "gemini-4-carbon-low", name: "Gemini 4 Carbon (Low)" },
  { modelId: "gemini-4-argon-high", name: "Gemini 4 Argon (High)" },
  { modelId: "gemini-4-argon-medium", name: "Gemini 4 Argon (Medium)" },
  { modelId: "gemini-4-argon-low", name: "Gemini 4 Argon (Low)" },
  { modelId: "gemini-3.8-flash-high", name: "Gemini 3.8 Flash (High)" },
  { modelId: "gemini-3.8-flash-medium", name: "Gemini 3.8 Flash (Medium)" },
  { modelId: "gemini-3.8-flash-low", name: "Gemini 3.8 Flash (Low)" },
  { modelId: "claude-opus-5-5-high", name: "Claude Opus 5.5 (High)" },
  { modelId: "claude-opus-5-5-medium", name: "Claude Opus 5.5 (Medium)" },
  { modelId: "claude-opus-5-5-low", name: "Claude Opus 5.5 (Low)" },
  { modelId: "claude-sonnet-5-5-high", name: "Claude Sonnet 5.5 (High)" },
  { modelId: "claude-sonnet-5-5-medium", name: "Claude Sonnet 5.5 (Medium)" },
  { modelId: "claude-sonnet-5-5-low", name: "Claude Sonnet 5.5 (Low)" },
  { modelId: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium)" },
];

export class WorkerPool {
  private workers = new Map<string, AcpWorker>();
  private waitQueue: QueuedWaiter[] = [];
  private sweepInterval: NodeJS.Timeout | null = null;
  private cachedModels: AcpModelInfo[] = [...DEFAULT_FALLBACK_MODELS];
  public readonly sessionRegistry: SessionAffinityRegistry;

  constructor() {
    this.sessionRegistry = new SessionAffinityRegistry(config.sessionTtlMs);
    onConfigChange((newCfg) => {
      this.sessionRegistry.setTtlMs(newCfg.sessionTtlMs, (wId, sId) => {
        this.workers.get(wId)?.closeSession(sId).catch(() => {});
      });
      this.ensureMinWorkers();
      this.reapIdleBurstWorkers();
    });
  }

  public async initialize(): Promise<void> {
    sweepStaleSandboxes();

    const startPromises: Promise<AcpWorker>[] = [];
    for (let i = 0; i < config.minWorkers; i++) {
      startPromises.push(this.spawnNewWorker());
    }
    await Promise.all(startPromises);

    this.sweepInterval = setInterval(() => {
      this.reapIdleBurstWorkers();
    }, 10_000);
    this.sweepInterval.unref?.();
  }

  /**
   * Acquires a worker for the given OpenAI request.
   * Uses `baseModelId` + `(firstUser + firstAssistant)` anchoring so changing `reasoning_effort`,
   * retrying after a timeout/abort (`session/cancel`), or stripping historical `tool_calls`
   * still hits the warm ACP session.
   */
  public async acquireForRequest(
    baseModelId: string,
    tools: OpenAIToolDefinition[] | undefined,
    messages: OpenAIMessage[]
  ): Promise<LeaseAcquisition> {
    // 0. If the matching warm session is currently settling `session/cancel` from an
    // aborted/timed-out request right before this retry, wait briefly for it to become IDLE.
    const candidates = this.sessionRegistry.findCandidateEntries(
      baseModelId,
      messages
    );
    if (candidates.length > 0) {
      const top = candidates[0];
      const candidateWorker = this.workers.get(top.workerId);
      if (
        candidateWorker &&
        candidateWorker.state === "BUSY" &&
        candidateWorker.hasSession(top.sessionId)
      ) {
        await candidateWorker.waitForSessionSettle(top.sessionId, 3500, 1000);
      }
    }

    // 1. Check Session Anchor + Bookmark affinity & health checks
    const match = this.sessionRegistry.findBestMatchingPrefix(
      baseModelId,
      tools,
      messages,
      {
        isSessionAlive: (workerId, sessionId) => {
          const w = this.workers.get(workerId);
          return Boolean(w && w.state !== "DEAD" && w.hasSession(sessionId));
        },
        isWorkerIdle: (workerId) => {
          const w = this.workers.get(workerId);
          return Boolean(w && w.state === "IDLE");
        },
        onInvalidateSession: (workerId, sessionId) => {
          this.workers.get(workerId)?.closeSession(sessionId).catch(() => {});
        },
      }
    );

    if (match) {
      const worker = this.workers.get(match.entry.workerId)!;
      worker.state = "BUSY";
      worker.lastUsedAt = Date.now();
      return {
        worker,
        sessionId: match.entry.sessionId,
        isReusedSession: true,
        deltaMessages: match.deltaMessages,
        matchedPrefixLength: match.matchedPrefixLength,
        isRetryAfterInterrupt: match.isRetryAfterInterrupt,
      };
    }

    // 2. Cache Miss: pick the IDLE worker holding the fewest active sessions
    let bestIdleWorker: AcpWorker | null = null;
    let bestIdleSessionCount = Infinity;
    for (const worker of this.workers.values()) {
      if (worker.state === "IDLE") {
        const count = this.sessionRegistry.countSessionsForWorker(worker.id);
        if (count < bestIdleSessionCount) {
          bestIdleWorker = worker;
          bestIdleSessionCount = count;
          if (count === 0) break;
        }
      }
    }

    if (bestIdleWorker) {
      bestIdleWorker.state = "BUSY";
      bestIdleWorker.lastUsedAt = Date.now();
      return {
        worker: bestIdleWorker,
        sessionId: null,
        isReusedSession: false,
        deltaMessages: messages,
        matchedPrefixLength: 0,
        isRetryAfterInterrupt: false,
      };
    }

    // 3. All current workers are busy: spawn a burst worker if below maxWorkers
    if (this.workers.size < config.maxWorkers) {
      const burstWorker = await this.spawnNewWorker();
      burstWorker.state = "BUSY";
      burstWorker.lastUsedAt = Date.now();
      return {
        worker: burstWorker,
        sessionId: null,
        isReusedSession: false,
        deltaMessages: messages,
        matchedPrefixLength: 0,
        isRetryAfterInterrupt: false,
      };
    }

    // 4. At maxWorkers capacity: wait in queue for the next available worker
    const queuedWorker = await this.enqueueWaiter(60_000);
    queuedWorker.state = "BUSY";
    queuedWorker.lastUsedAt = Date.now();
    return {
      worker: queuedWorker,
      sessionId: null,
      isReusedSession: false,
      deltaMessages: messages,
      matchedPrefixLength: 0,
      isRetryAfterInterrupt: false,
    };
  }

  /**
   * Evaluates session health after a timeout, client abort, or RPC error:
   * - Hung `session/cancel` (`!cancelSettled`) -> destroys hung worker;
   * - Pre-prompt abort on a brand-new session -> closes unused session;
   * - 1st settled interruption on a warm session -> keeps session alive for KV-cached retry;
   * - 2nd consecutive interruption (or Turn 1 failure) -> closes poisoned session.
   */
  public async handleFailedTurnSession(
    worker: AcpWorker,
    sessionId: string | null,
    isReusedSession: boolean,
    err: unknown,
    partialAssistantMsg?: OpenAIMessage
  ): Promise<void> {
    if (!sessionId) return;

    if (worker.state === "DEAD") {
      this.sessionRegistry.removeWorkerSessions(worker.id);
      return;
    }

    if (err instanceof PromptInterruptedError) {
      if (!err.cancelSettled) {
        this.sessionRegistry.removeSession(sessionId);
        this.terminateWorker(worker.id);
        return;
      }
      if (!err.promptSent) {
        if (!isReusedSession) {
          await this.closeWorkerSession(worker.id, sessionId);
        }
        return;
      }
    }

    const { shouldDestroy } = this.sessionRegistry.recordTurnInterruption(
      sessionId,
      partialAssistantMsg
    );
    if (shouldDestroy) {
      await this.closeWorkerSession(worker.id, sessionId);
    }
  }

  public releaseWorker(worker: AcpWorker): void {
    if (worker.state === "DEAD" || !this.workers.has(worker.id)) {
      this.ensureMinWorkers();
      return;
    }

    if (worker.availableModels.length > 0) {
      this.cachedModels = worker.availableModels;
    }

    // Give priority to an immediate affinity retry waiting on this specific session's cancel-settle
    if (worker.hasIdleWaiters()) {
      worker.state = "IDLE";
      worker.lastUsedAt = Date.now();
      worker.notifyIdle();
      queueMicrotask(() => {
        if (
          worker.state === "IDLE" &&
          this.workers.has(worker.id) &&
          this.waitQueue.length > 0
        ) {
          const waiter = this.waitQueue.shift()!;
          clearTimeout(waiter.timer);
          worker.state = "BUSY";
          worker.lastUsedAt = Date.now();
          waiter.resolve(worker);
        }
      });
      return;
    }

    while (this.waitQueue.length > 0) {
      const waiter = this.waitQueue.shift()!;
      clearTimeout(waiter.timer);
      worker.state = "BUSY";
      worker.lastUsedAt = Date.now();
      waiter.resolve(worker);
      return;
    }

    worker.state = "IDLE";
    worker.lastUsedAt = Date.now();
    worker.notifyIdle();
  }

  public getAvailableModels(): AcpModelInfo[] {
    return this.cachedModels;
  }

  public getStats() {
    let idle = 0;
    let busy = 0;
    let starting = 0;
    for (const w of this.workers.values()) {
      if (w.state === "IDLE") idle++;
      else if (w.state === "BUSY") busy++;
      else if (w.state === "STARTING") starting++;
    }
    return {
      totalWorkers: this.workers.size,
      idleWorkers: idle,
      busyWorkers: busy,
      startingWorkers: starting,
      queuedRequests: this.waitQueue.length,
      cachedSessionHashes: this.sessionRegistry.size(),
    };
  }

  public getDetailedWorkers() {
    const now = Date.now();
    return Array.from(this.workers.values()).map((w) => {
      const idleElapsed = w.state === "IDLE" ? now - w.lastUsedAt : 0;
      const idleTtlRemainingMs = Math.max(0, config.workerIdleTtlMs - idleElapsed);
      return {
        id: w.id,
        pid: w.getPid(),
        state: w.state,
        snifferPort: w.getSnifferPort(),
        relAppDataDir: w.sandbox.relAppDataDir,
        activeSessions: w.getActiveSessionIds(),
        activeSessionCount: w.getActiveSessionIds().length,
        totalRequestsHandled: w.totalRequestsHandled,
        createdAt: w.createdAt,
        lastUsedAt: w.lastUsedAt,
        idleTtlRemainingMs,
      };
    });
  }

  public async spawnManualWorker(): Promise<AcpWorker> {
    if (this.workers.size >= config.maxWorkers) {
      throw new Error(`Maximum worker limit (${config.maxWorkers}) reached`);
    }
    return this.spawnNewWorker();
  }

  public terminateWorker(workerId: string): boolean {
    const w = this.workers.get(workerId);
    if (!w) return false;
    this.workers.delete(workerId);
    this.sessionRegistry.removeWorkerSessions(workerId);
    w.destroy();
    this.ensureMinWorkers();
    return true;
  }

  public async closeWorkerSession(
    workerId: string,
    sessionId: string
  ): Promise<boolean> {
    const w = this.workers.get(workerId);
    if (!w) return false;
    this.sessionRegistry.removeSession(sessionId);
    await w.closeSession(sessionId);
    return true;
  }

  public shutdown(): void {
    if (this.sweepInterval) {
      clearInterval(this.sweepInterval);
      this.sweepInterval = null;
    }
    for (const waiter of this.waitQueue) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("WorkerPool shutting down"));
    }
    this.waitQueue = [];

    for (const worker of this.workers.values()) {
      worker.destroy();
    }
    this.workers.clear();
  }

  private async spawnNewWorker(): Promise<AcpWorker> {
    const id = crypto.randomUUID().slice(0, 8);
    const worker = new AcpWorker(id, (deadId) => {
      this.workers.delete(deadId);
      this.sessionRegistry.removeWorkerSessions(deadId);
      this.ensureMinWorkers();
    });

    this.workers.set(id, worker);
    try {
      await worker.start();
      return worker;
    } catch (err) {
      this.workers.delete(id);
      worker.destroy();
      throw err;
    }
  }

  private ensureMinWorkers(): void {
    while (this.workers.size < config.minWorkers) {
      this.spawnNewWorker().catch(() => {});
    }
  }

  private enqueueWaiter(timeoutMs: number): Promise<AcpWorker> {
    return new Promise<AcpWorker>((resolve, reject) => {
      const timer = setTimeout(() => {
        const idx = this.waitQueue.findIndex((w) => w.timer === timer);
        if (idx !== -1) this.waitQueue.splice(idx, 1);
        reject(
          new Error(`Timed out waiting for an available ACP worker (${timeoutMs}ms)`)
        );
      }, timeoutMs);

      this.waitQueue.push({ resolve, reject, timer });
    });
  }

  private reapIdleBurstWorkers(): void {
    this.sessionRegistry.pruneExpired((wId, sId) => {
      this.workers.get(wId)?.closeSession(sId).catch(() => {});
    });

    if (this.workers.size <= config.minWorkers) return;

    const now = Date.now();
    for (const [id, worker] of this.workers.entries()) {
      if (this.workers.size <= config.minWorkers) break;
      if (
        worker.state === "IDLE" &&
        now - worker.lastUsedAt > config.workerIdleTtlMs &&
        !this.sessionRegistry.hasActiveSessionsForWorker(id)
      ) {
        this.workers.delete(id);
        this.sessionRegistry.removeWorkerSessions(id);
        worker.destroy();
      }
    }
  }
}
