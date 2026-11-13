import { DatabaseSync } from "node:sqlite";
import { config } from "../config.js";
import { TelemetryStore, telemetryStore } from "./metrics-store.js";

export type DebugLogDirection = "stdin" | "stdout" | "stderr" | "sys";
export type DebugLogCategory =
  | "rpc_req"
  | "rpc_res"
  | "rpc_srv"
  | "thought"
  | "message"
  | "tool_call"
  | "config"
  | "stderr"
  | "stdout_raw"
  | "error"
  | "sys";

export interface DebugSessionSummary {
  sessionId: string;
  workerId: string;
  createdAt: number;
  lastLogAt: number;
  closedAt: number | null;
  status: "ACTIVE" | "CLOSED";
  logCount: number;
  rawLineCount: number;
  errorCount: number;
  baseModelId?: string;
  lastThinkingLevel?: string;
  lastActualModelVersion?: string;
}

export interface DebugLogItem {
  id: number;
  sessionId: string;
  workerId: string;
  timestamp: number;
  updatedAt: number;
  direction: DebugLogDirection;
  category: DebugLogCategory;
  method: string;
  summary: string;
  chunkCount: number;
  contentLength: number;
  rawTruncated: boolean;
}

export interface DebugLogDetail extends DebugLogItem {
  content: string;
  rawFrames: string[];
}

export interface DebugSummary {
  activeSessions: number;
  closedSessions: number;
  totalSessions: number;
  totalLogs: number;
  totalRawLines: number;
  totalErrors: number;
}

const MAX_CONTENT_BYTES = 256 * 1024;
const MAX_RAW_FRAMES = 200;
const MAX_RAW_FRAMES_BYTES = 64 * 1024;
const CHUNK_FLUSH_DELAY_MS = 400;

interface OpenChunkBuffer {
  rowId: number;
  sessionId: string;
  workerId: string;
  category: "thought" | "message";
  method: string;
  timestamp: number;
  updatedAt: number;
  content: string;
  contentTruncated: boolean;
  rawFrames: string[];
  rawFramesBytes: number;
  rawTruncated: boolean;
  chunkCount: number;
  pendingChunks: number;
  dirty: boolean;
  flushTimer: NodeJS.Timeout | null;
}

function makePreviewSummary(text: string, maxLen: number = 140): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= maxLen) return flat;
  return flat.slice(0, maxLen - 1) + "…";
}

function truncateUtf8(str: string, maxBytes: number): { text: string; truncated: boolean } {
  const buf = Buffer.from(str, "utf-8");
  if (buf.length <= maxBytes) {
    return { text: str, truncated: false };
  }
  const sliced = buf.subarray(0, maxBytes).toString("utf-8");
  return { text: sliced, truncated: true };
}

export class DebugLogStore {
  private db: DatabaseSync;
  private openChunks = new Map<string, OpenChunkBuffer>();
  private closed = false;

  constructor(store: TelemetryStore = telemetryStore) {
    this.db = store.getDb();
    this.initSchema();

    store.onSessionClosed((sessionId) => {
      this.markSessionClosed(sessionId, config.logRetentionMs);
    });
    store.onWorkerSessionsClosed((workerId) => {
      this.markWorkerSessionsClosed(workerId, config.logRetentionMs);
    });
    store.onBeforeClose(() => {
      this.flushAll();
      this.closed = true;
    });
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS debug_sessions (
        session_id TEXT PRIMARY KEY,
        worker_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_log_at INTEGER NOT NULL,
        closed_at INTEGER,
        status TEXT NOT NULL,
        log_count INTEGER NOT NULL DEFAULT 0,
        raw_line_count INTEGER NOT NULL DEFAULT 0,
        error_count INTEGER NOT NULL DEFAULT 0
      );

      CREATE TABLE IF NOT EXISTS session_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        direction TEXT NOT NULL,
        category TEXT NOT NULL,
        method TEXT NOT NULL,
        summary TEXT NOT NULL,
        content TEXT NOT NULL,
        raw_frames TEXT NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 1,
        raw_truncated INTEGER NOT NULL DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_debug_sessions_last_log ON debug_sessions(last_log_at DESC);
      CREATE INDEX IF NOT EXISTS idx_debug_sessions_status_closed ON debug_sessions(status, closed_at);
      CREATE INDEX IF NOT EXISTS idx_session_logs_session_id ON session_logs(session_id, id DESC);
      CREATE INDEX IF NOT EXISTS idx_session_logs_timestamp ON session_logs(timestamp DESC, id DESC);
    `);

    // Recover leftover ACTIVE sessions from previous gateway runs as CLOSED so retention applies
    this.db.exec(`
      UPDATE debug_sessions
      SET status = 'CLOSED',
          closed_at = COALESCE(closed_at, last_log_at)
      WHERE status = 'ACTIVE'
    `);
  }

  public registerSession(
    sessionId: string,
    workerId: string,
    status: "ACTIVE" | "CLOSED" = "ACTIVE"
  ): void {
    if (this.closed || !sessionId) return;
    const now = Date.now();
    const closedAt = status === "CLOSED" ? now : null;
    this.db
      .prepare(
        `
      INSERT INTO debug_sessions (
        session_id, worker_id, created_at, last_log_at, closed_at, status, log_count, raw_line_count, error_count
      ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0)
      ON CONFLICT(session_id) DO UPDATE SET
        worker_id = excluded.worker_id,
        status = CASE WHEN excluded.status = 'ACTIVE' THEN 'ACTIVE' ELSE debug_sessions.status END,
        closed_at = CASE WHEN excluded.status = 'ACTIVE' THEN NULL ELSE debug_sessions.closed_at END
    `
      )
      .run(sessionId, workerId, now, now, closedAt, status);
  }

  public appendStreamChunk(opts: {
    sessionId: string;
    workerId: string;
    category: "thought" | "message";
    method: string;
    text: string;
    rawFrame: string;
    timestamp?: number;
  }): number {
    if (this.closed) return 0;
    const now = opts.timestamp ?? Date.now();
    const existing = this.openChunks.get(opts.sessionId);

    if (existing && existing.category === opts.category) {
      this.appendToBuffer(existing, opts.text, opts.rawFrame, now);
      return existing.rowId;
    }

    if (existing) {
      this.flushSessionChunk(opts.sessionId);
    }

    this.registerSession(opts.sessionId, opts.workerId, "ACTIVE");

    const initText = truncateUtf8(opts.text, MAX_CONTENT_BYTES);
    const frameBytes = Buffer.byteLength(opts.rawFrame, "utf-8");
    const keepFrame = frameBytes <= MAX_RAW_FRAMES_BYTES;
    const rawFrames = keepFrame ? [opts.rawFrame] : [];
    const rawTruncated = !keepFrame || initText.truncated;
    const summary = makePreviewSummary(initText.text);

    const insertRes = this.db
      .prepare(
        `
      INSERT INTO session_logs (
        session_id, worker_id, timestamp, updated_at, direction, category, method, summary, content, raw_frames, chunk_count, raw_truncated
      ) VALUES (?, ?, ?, ?, 'stdout', ?, ?, ?, ?, ?, 1, ?)
    `
      )
      .run(
        opts.sessionId,
        opts.workerId,
        now,
        now,
        opts.category,
        opts.method,
        summary,
        initText.text,
        JSON.stringify(rawFrames),
        rawTruncated ? 1 : 0
      );

    const rowId = Number(insertRes.lastInsertRowid);

    this.db
      .prepare(
        `
      UPDATE debug_sessions
      SET last_log_at = ?,
          log_count = log_count + 1,
          raw_line_count = raw_line_count + 1
      WHERE session_id = ?
    `
      )
      .run(now, opts.sessionId);

    const buf: OpenChunkBuffer = {
      rowId,
      sessionId: opts.sessionId,
      workerId: opts.workerId,
      category: opts.category,
      method: opts.method,
      timestamp: now,
      updatedAt: now,
      content: initText.text,
      contentTruncated: initText.truncated,
      rawFrames,
      rawFramesBytes: keepFrame ? frameBytes : 0,
      rawTruncated,
      chunkCount: 1,
      pendingChunks: 0,
      dirty: false,
      flushTimer: null,
    };

    this.openChunks.set(opts.sessionId, buf);
    return rowId;
  }

  private appendToBuffer(
    buf: OpenChunkBuffer,
    text: string,
    rawFrame: string,
    now: number
  ): void {
    buf.updatedAt = now;
    buf.chunkCount += 1;
    buf.pendingChunks += 1;
    buf.dirty = true;

    if (!buf.contentTruncated && text) {
      const combined = buf.content + text;
      const checked = truncateUtf8(combined, MAX_CONTENT_BYTES);
      buf.content = checked.text;
      if (checked.truncated) {
        buf.contentTruncated = true;
        buf.rawTruncated = true;
      }
    }

    if (!buf.rawTruncated) {
      const frameBytes = Buffer.byteLength(rawFrame, "utf-8");
      if (
        buf.rawFrames.length < MAX_RAW_FRAMES &&
        buf.rawFramesBytes + frameBytes <= MAX_RAW_FRAMES_BYTES
      ) {
        buf.rawFrames.push(rawFrame);
        buf.rawFramesBytes += frameBytes;
      } else {
        buf.rawTruncated = true;
      }
    }

    if (!buf.flushTimer) {
      buf.flushTimer = setTimeout(() => {
        buf.flushTimer = null;
        this.flushBufferToDb(buf);
      }, CHUNK_FLUSH_DELAY_MS);
      buf.flushTimer.unref?.();
    }
  }

  private flushBufferToDb(buf: OpenChunkBuffer): void {
    if (this.closed || !buf.dirty) return;
    const summary = makePreviewSummary(buf.content);
    this.db
      .prepare(
        `
      UPDATE session_logs
      SET updated_at = ?,
          summary = ?,
          content = ?,
          raw_frames = ?,
          chunk_count = ?,
          raw_truncated = ?
      WHERE id = ?
    `
      )
      .run(
        buf.updatedAt,
        summary,
        buf.content,
        JSON.stringify(buf.rawFrames),
        buf.chunkCount,
        buf.rawTruncated ? 1 : 0,
        buf.rowId
      );

    if (buf.pendingChunks > 0) {
      this.db
        .prepare(
          `
        UPDATE debug_sessions
        SET last_log_at = MAX(last_log_at, ?),
            raw_line_count = raw_line_count + ?
        WHERE session_id = ?
      `
        )
        .run(buf.updatedAt, buf.pendingChunks, buf.sessionId);
      buf.pendingChunks = 0;
    }

    buf.dirty = false;
  }

  public flushSessionChunk(sessionId: string): void {
    const buf = this.openChunks.get(sessionId);
    if (!buf) return;
    if (buf.flushTimer) {
      clearTimeout(buf.flushTimer);
      buf.flushTimer = null;
    }
    this.flushBufferToDb(buf);
    this.openChunks.delete(sessionId);
  }

  public flushAll(): void {
    for (const sessionId of Array.from(this.openChunks.keys())) {
      this.flushSessionChunk(sessionId);
    }
  }

  public recordEvent(opts: {
    sessionId: string;
    workerId: string;
    direction: DebugLogDirection;
    category: Exclude<DebugLogCategory, "thought" | "message">;
    method: string;
    summary: string;
    content: string;
    rawFrame: string;
    timestamp?: number;
    sessionStatusOnCreate?: "ACTIVE" | "CLOSED";
  }): number {
    if (this.closed || !opts.sessionId) return 0;

    // Preserve strict chronological ordering between stream chunks and discrete events
    this.flushSessionChunk(opts.sessionId);

    const now = opts.timestamp ?? Date.now();
    this.registerSession(
      opts.sessionId,
      opts.workerId,
      opts.sessionStatusOnCreate ?? "ACTIVE"
    );

    const checkedContent = truncateUtf8(opts.content, MAX_CONTENT_BYTES);
    const checkedFrame = truncateUtf8(opts.rawFrame, MAX_RAW_FRAMES_BYTES);
    const rawTruncated = checkedContent.truncated || checkedFrame.truncated ? 1 : 0;
    const isErr = opts.category === "error" || opts.category === "stderr" ? 1 : 0;

    const insertRes = this.db
      .prepare(
        `
      INSERT INTO session_logs (
        session_id, worker_id, timestamp, updated_at, direction, category, method, summary, content, raw_frames, chunk_count, raw_truncated
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
    `
      )
      .run(
        opts.sessionId,
        opts.workerId,
        now,
        now,
        opts.direction,
        opts.category,
        opts.method,
        makePreviewSummary(opts.summary),
        checkedContent.text,
        JSON.stringify([checkedFrame.text]),
        rawTruncated
      );

    this.db
      .prepare(
        `
      UPDATE debug_sessions
      SET last_log_at = MAX(last_log_at, ?),
          log_count = log_count + 1,
          raw_line_count = raw_line_count + 1,
          error_count = error_count + ?
      WHERE session_id = ?
    `
      )
      .run(now, isErr, opts.sessionId);

    return Number(insertRes.lastInsertRowid);
  }

  public markSessionClosed(
    sessionId: string,
    retentionMs: number = config.logRetentionMs
  ): void {
    if (this.closed || !sessionId) return;
    this.flushSessionChunk(sessionId);
    const now = Date.now();
    this.db
      .prepare(
        `
      UPDATE debug_sessions
      SET status = 'CLOSED',
          closed_at = COALESCE(closed_at, ?)
      WHERE session_id = ? AND status != 'CLOSED'
    `
      )
      .run(now, sessionId);

    if (retentionMs <= 0) {
      this.pruneExpired(retentionMs, now);
    }
  }

  public markWorkerSessionsClosed(
    workerId: string,
    retentionMs: number = config.logRetentionMs
  ): void {
    if (this.closed || !workerId) return;
    for (const [sid, buf] of this.openChunks.entries()) {
      if (buf.workerId === workerId) {
        this.flushSessionChunk(sid);
      }
    }
    const now = Date.now();
    this.db
      .prepare(
        `
      UPDATE debug_sessions
      SET status = 'CLOSED',
          closed_at = COALESCE(closed_at, ?)
      WHERE worker_id = ? AND status = 'ACTIVE'
    `
      )
      .run(now, workerId);

    if (retentionMs <= 0) {
      this.pruneExpired(retentionMs, now);
    }
  }

  public pruneExpired(
    retentionMs: number = config.logRetentionMs,
    now: number = Date.now()
  ): number {
    if (this.closed) return 0;
    const cutoff = now - Math.max(0, retentionMs);
    const expiredRows = this.db
      .prepare(
        `
      SELECT session_id FROM debug_sessions
      WHERE status = 'CLOSED' AND closed_at IS NOT NULL AND closed_at <= ?
    `
      )
      .all(cutoff) as Array<{ session_id: string }>;

    if (expiredRows.length === 0) return 0;

    const delLogs = this.db.prepare(`DELETE FROM session_logs WHERE session_id = ?`);
    const delSess = this.db.prepare(`DELETE FROM debug_sessions WHERE session_id = ?`);
    for (const row of expiredRows) {
      this.openChunks.delete(row.session_id);
      delLogs.run(row.session_id);
      delSess.run(row.session_id);
    }
    return expiredRows.length;
  }

  public getSummary(): DebugSummary {
    if (this.closed) {
      return {
        activeSessions: 0,
        closedSessions: 0,
        totalSessions: 0,
        totalLogs: 0,
        totalRawLines: 0,
        totalErrors: 0,
      };
    }

    const row = this.db
      .prepare(
        `
      SELECT
        COALESCE(SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END), 0) AS activeSessions,
        COALESCE(SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END), 0) AS closedSessions,
        COUNT(*) AS totalSessions,
        COALESCE(SUM(log_count), 0) AS totalLogs,
        COALESCE(SUM(raw_line_count), 0) AS totalRawLines,
        COALESCE(SUM(error_count), 0) AS totalErrors
      FROM debug_sessions
    `
      )
      .get() as Record<string, number>;

    let pendingRaw = 0;
    for (const buf of this.openChunks.values()) {
      pendingRaw += buf.pendingChunks;
    }

    return {
      activeSessions: Number(row?.activeSessions ?? 0),
      closedSessions: Number(row?.closedSessions ?? 0),
      totalSessions: Number(row?.totalSessions ?? 0),
      totalLogs: Number(row?.totalLogs ?? 0),
      totalRawLines: Number(row?.totalRawLines ?? 0) + pendingRaw,
      totalErrors: Number(row?.totalErrors ?? 0),
    };
  }

  public getSessions(limit: number = 100): DebugSessionSummary[] {
    if (this.closed) return [];
    const safeLimit = Math.max(1, Math.min(500, Math.floor(limit)));
    const rows = this.db
      .prepare(
        `
      SELECT
        ds.session_id,
        ds.worker_id,
        ds.created_at,
        ds.last_log_at,
        ds.closed_at,
        ds.status,
        ds.log_count,
        ds.raw_line_count,
        ds.error_count,
        s.base_model_id,
        s.last_thinking_level,
        s.last_actual_model_version
      FROM debug_sessions ds
      LEFT JOIN sessions s ON ds.session_id = s.session_id
      ORDER BY ds.last_log_at DESC
      LIMIT ?
    `
      )
      .all(safeLimit) as Array<Record<string, any>>;

    return rows.map((row) => {
      const sid = String(row.session_id);
      const buf = this.openChunks.get(sid);
      return {
        sessionId: sid,
        workerId: String(row.worker_id),
        createdAt: Number(row.created_at),
        lastLogAt: buf
          ? Math.max(Number(row.last_log_at), buf.updatedAt)
          : Number(row.last_log_at),
        closedAt: row.closed_at != null ? Number(row.closed_at) : null,
        status: row.status as "ACTIVE" | "CLOSED",
        logCount: Number(row.log_count),
        rawLineCount: Number(row.raw_line_count) + (buf ? buf.pendingChunks : 0),
        errorCount: Number(row.error_count),
        ...(row.base_model_id ? { baseModelId: String(row.base_model_id) } : {}),
        ...(row.last_thinking_level
          ? { lastThinkingLevel: String(row.last_thinking_level) }
          : {}),
        ...(row.last_actual_model_version
          ? { lastActualModelVersion: String(row.last_actual_model_version) }
          : {}),
      };
    });
  }

  public getLogs(opts?: {
    sessionId?: string;
    q?: string;
    limit?: number;
  }): DebugLogItem[] {
    if (this.closed) return [];
    const safeLimit = Math.max(1, Math.min(500, Math.floor(opts?.limit ?? 200)));
    const sessionId = opts?.sessionId?.trim() || "";
    const q = opts?.q?.trim().toLowerCase() || "";

    const whereClauses: string[] = [];
    const params: Array<string | number> = [];

    if (sessionId) {
      whereClauses.push("session_id = ?");
      params.push(sessionId);
    }
    if (q) {
      const like = `%${q}%`;
      whereClauses.push(
        "(LOWER(session_id) LIKE ? OR LOWER(worker_id) LIKE ? OR LOWER(category) LIKE ? OR LOWER(method) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(content) LIKE ?)"
      );
      params.push(like, like, like, like, like, like);
    }

    const whereSql =
      whereClauses.length > 0 ? `WHERE ${whereClauses.join(" AND ")}` : "";

    const rows = this.db
      .prepare(
        `
      SELECT
        id, session_id, worker_id, timestamp, updated_at, direction, category, method, summary,
        LENGTH(content) AS content_length, chunk_count, raw_truncated
      FROM session_logs
      ${whereSql}
      ORDER BY id DESC
      LIMIT ?
    `
      )
      .all(...params, safeLimit) as Array<Record<string, any>>;

    const openById = new Map<number, OpenChunkBuffer>();
    for (const buf of this.openChunks.values()) {
      openById.set(buf.rowId, buf);
    }

    return rows.map((row) => {
      const id = Number(row.id);
      const buf = openById.get(id);
      return {
        id,
        sessionId: String(row.session_id),
        workerId: String(row.worker_id),
        timestamp: Number(row.timestamp),
        updatedAt: buf ? buf.updatedAt : Number(row.updated_at),
        direction: row.direction as DebugLogDirection,
        category: row.category as DebugLogCategory,
        method: String(row.method),
        summary: buf ? makePreviewSummary(buf.content) : String(row.summary),
        chunkCount: buf ? buf.chunkCount : Number(row.chunk_count),
        contentLength: buf ? buf.content.length : Number(row.content_length),
        rawTruncated: buf ? buf.rawTruncated : Boolean(row.raw_truncated),
      };
    });
  }

  public getLogDetail(id: number): DebugLogDetail | null {
    if (this.closed || !Number.isFinite(id)) return null;
    const row = this.db
      .prepare(`SELECT * FROM session_logs WHERE id = ?`)
      .get(id) as Record<string, any> | undefined;
    if (!row) return null;

    let openBuf: OpenChunkBuffer | undefined;
    for (const buf of this.openChunks.values()) {
      if (buf.rowId === id) {
        openBuf = buf;
        break;
      }
    }

    if (openBuf) {
      return {
        id,
        sessionId: openBuf.sessionId,
        workerId: openBuf.workerId,
        timestamp: openBuf.timestamp,
        updatedAt: openBuf.updatedAt,
        direction: "stdout",
        category: openBuf.category,
        method: openBuf.method,
        summary: makePreviewSummary(openBuf.content),
        chunkCount: openBuf.chunkCount,
        contentLength: openBuf.content.length,
        rawTruncated: openBuf.rawTruncated,
        content: openBuf.content,
        rawFrames: [...openBuf.rawFrames],
      };
    }

    let rawFrames: string[] = [];
    try {
      const parsed = JSON.parse(String(row.raw_frames || "[]"));
      if (Array.isArray(parsed)) {
        rawFrames = parsed.map((x) => String(x));
      }
    } catch {}

    const content = String(row.content || "");
    return {
      id: Number(row.id),
      sessionId: String(row.session_id),
      workerId: String(row.worker_id),
      timestamp: Number(row.timestamp),
      updatedAt: Number(row.updated_at),
      direction: row.direction as DebugLogDirection,
      category: row.category as DebugLogCategory,
      method: String(row.method),
      summary: String(row.summary),
      chunkCount: Number(row.chunk_count),
      contentLength: content.length,
      rawTruncated: Boolean(row.raw_truncated),
      content,
      rawFrames,
    };
  }

  public clearLogs(sessionId?: string): number {
    if (this.closed) return 0;
    const sid = sessionId?.trim();
    if (sid) {
      this.openChunks.delete(sid);
      const res = this.db
        .prepare(`DELETE FROM session_logs WHERE session_id = ?`)
        .run(sid);
      this.db.prepare(`DELETE FROM debug_sessions WHERE session_id = ?`).run(sid);
      return Number(res.changes || 0);
    }

    this.openChunks.clear();
    const res = this.db.prepare(`DELETE FROM session_logs`).run();
    this.db.prepare(`DELETE FROM debug_sessions`).run();
    return Number(res.changes || 0);
  }
}

export const debugStore = new DebugLogStore(telemetryStore);
