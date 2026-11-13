import crypto from "node:crypto";
import { AcpPromptContentBlock } from "../types/acp.js";
import { OpenAIMessage, OpenAIMessageContentPart } from "../types/openai.js";

export type ParsedImageSource =
  | {
      kind: "inline";
      mimeType: string;
      data: string;
    }
  | {
      kind: "remote";
      url: string;
    };

export type MessageContentSegment =
  | {
      kind: "text";
      text: string;
    }
  | ParsedImageSource;

const MAX_REMOTE_IMAGE_BYTES = 25 * 1024 * 1024; // 25 MiB safety cap
const REMOTE_IMAGE_TIMEOUT_MS = 30_000;

function sha256Hex(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

/**
 * Sniffs standard IANA image MIME type from raw binary header bytes.
 */
export function sniffImageMimeFromBuffer(buf: Buffer): string | null {
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) {
    return "image/jpeg";
  }
  if (
    buf.length >= 8 &&
    buf[0] === 0x89 &&
    buf[1] === 0x50 &&
    buf[2] === 0x4e &&
    buf[3] === 0x47 &&
    buf[4] === 0x0d &&
    buf[5] === 0x0a &&
    buf[6] === 0x1a &&
    buf[7] === 0x0a
  ) {
    return "image/png";
  }
  if (
    buf.length >= 6 &&
    buf[0] === 0x47 &&
    buf[1] === 0x49 &&
    buf[2] === 0x46 &&
    buf[3] === 0x38
  ) {
    return "image/gif";
  }
  if (
    buf.length >= 12 &&
    buf[0] === 0x52 &&
    buf[1] === 0x49 &&
    buf[2] === 0x46 &&
    buf[3] === 0x46 &&
    buf[8] === 0x57 &&
    buf[9] === 0x45 &&
    buf[10] === 0x42 &&
    buf[11] === 0x50
  ) {
    return "image/webp";
  }
  if (buf.length >= 2 && buf[0] === 0x42 && buf[1] === 0x4d) {
    return "image/bmp";
  }
  if (buf.length >= 12 && buf.subarray(4, 8).toString("ascii") === "ftyp") {
    const brand = buf.subarray(8, 12).toString("ascii").toLowerCase();
    if (brand.startsWith("hei") || brand.startsWith("mif1")) {
      return "image/heic";
    }
    if (brand.startsWith("avi")) {
      return "image/avif";
    }
  }
  return null;
}

/**
 * Normalizes non-standard MIME aliases (such as Android/client `image/jpg`)
 * and falls back to magic-byte inspection when MIME is missing or generic.
 */
export function normalizeImageMimeType(
  rawMime: string | undefined,
  base64Data?: string
): string {
  const cleaned = (rawMime || "").split(";")[0].trim().toLowerCase();

  if (cleaned === "image/jpg" || cleaned === "image/pjpeg") {
    return "image/jpeg";
  }
  if (cleaned === "image/x-png") {
    return "image/png";
  }
  if (cleaned === "image/svg") {
    return "image/svg+xml";
  }

  if (cleaned.startsWith("image/")) {
    return cleaned;
  }

  if (base64Data) {
    try {
      const head = Buffer.from(base64Data.slice(0, 64), "base64");
      const sniffed = sniffImageMimeFromBuffer(head);
      if (sniffed) return sniffed;
    } catch {
      // Fall through to default
    }
  }

  return "image/png";
}

function cleanBase64Payload(raw: string): string {
  // Strip any whitespace/newlines (e.g. Android Base64.DEFAULT 76-char line wraps)
  // and normalize URL-safe base64 characters if present.
  return raw.replace(/\s+/g, "").replace(/-/g, "+").replace(/_/g, "/");
}

function parseUrlOrDataUri(
  rawUrl: string,
  hintMime?: string
): ParsedImageSource | null {
  const trimmed = rawUrl.trim();
  if (!trimmed) return null;

  // 1. Standard Data URI: `data:[<mediatype>][;param=...];base64,<data>`
  if (/^data:/i.test(trimmed)) {
    const commaIdx = trimmed.indexOf(",");
    if (commaIdx === -1) return null;
    const header = trimmed.slice(5, commaIdx);
    const body = trimmed.slice(commaIdx + 1);
    if (!/;base64/i.test(header)) return null;

    const rawMime = header.split(";")[0]?.trim() || hintMime;
    const data = cleanBase64Payload(body);
    if (!data) return null;
    return {
      kind: "inline",
      mimeType: normalizeImageMimeType(rawMime, data),
      data,
    };
  }

  // 2. Remote HTTP(S) URL
  if (/^https?:\/\//i.test(trimmed)) {
    return {
      kind: "remote",
      url: trimmed,
    };
  }

  // 3. Bare base64 payload passed directly without `data:` prefix
  const compact = cleanBase64Payload(trimmed);
  if (compact.length >= 16 && /^[A-Za-z0-9+/=]+$/.test(compact)) {
    return {
      kind: "inline",
      mimeType: normalizeImageMimeType(hintMime, compact),
      data: compact,
    };
  }

  return null;
}

/**
 * Parses an OpenAI / multimodal message content part into a normalized image source.
 * Supports:
 *  - `{ type: "image_url", image_url: { url: "data:image/...;base64,..." | "https://..." } }`
 *  - `{ type: "image_url", image_url: "data:image/...;base64,..." }`
 *  - `{ type: "input_image", image_url: ... | url: ... }`
 *  - `{ type: "image", data: "...", mimeType / media_type: "..." }`
 *  - `{ type: "image", source: { type: "base64" | "url", media_type: "...", data / url: "..." } }`
 */
export function parsePartAsImage(
  part: OpenAIMessageContentPart
): ParsedImageSource | null {
  if (!part || typeof part !== "object") return null;

  const t = typeof part.type === "string" ? part.type.toLowerCase() : "";
  if (t !== "image_url" && t !== "input_image" && t !== "image") {
    return null;
  }

  const hintMime =
    part.mimeType ||
    part.media_type ||
    part.source?.media_type ||
    part.source?.mimeType;

  // Explicit inline base64 fields (`part.source.data` or `part.data`)
  const explicitData =
    typeof part.source?.data === "string"
      ? part.source.data
      : typeof part.data === "string"
        ? part.data
        : null;

  if (explicitData) {
    if (/^data:/i.test(explicitData.trim())) {
      return parseUrlOrDataUri(explicitData, hintMime);
    }
    const data = cleanBase64Payload(explicitData);
    if (data) {
      return {
        kind: "inline",
        mimeType: normalizeImageMimeType(hintMime, data),
        data,
      };
    }
  }

  const rawUrl =
    typeof part.image_url === "string"
      ? part.image_url
      : typeof part.image_url?.url === "string"
        ? part.image_url.url
        : typeof part.url === "string"
          ? part.url
          : typeof part.source?.url === "string"
            ? part.source.url
            : null;

  if (rawUrl) {
    return parseUrlOrDataUri(rawUrl, hintMime);
  }

  return null;
}

/**
 * Extracts ordered text & image segments from `OpenAIMessage["content"]`.
 * Consecutive text parts within the same message array are joined with `\n`
 * to match `extractTextContent` behavior on text-only arrays.
 */
export function extractContentSegments(
  content: OpenAIMessage["content"]
): MessageContentSegment[] {
  if (!content) return [];
  if (typeof content === "string") {
    return content ? [{ kind: "text", text: content }] : [];
  }
  if (!Array.isArray(content)) {
    const str = String(content);
    return str ? [{ kind: "text", text: str }] : [];
  }

  const segments: MessageContentSegment[] = [];
  let pendingTextParts: string[] = [];

  const flushPendingText = () => {
    if (pendingTextParts.length === 0) return;
    const joined = pendingTextParts.join("\n");
    pendingTextParts = [];
    if (joined) {
      segments.push({ kind: "text", text: joined });
    }
  };

  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "text" || part.type === "input_text") {
      pendingTextParts.push(part.text || "");
      continue;
    }

    const img = parsePartAsImage(part);
    if (img) {
      flushPendingText();
      segments.push(img);
    }
  }

  flushPendingText();
  return segments;
}

/**
 * Computes deterministic, synchronous SHA-256 fingerprints for all images in `content`.
 * Returns `[]` when no images are present so text-only message hashes remain unchanged.
 */
export function extractImageFingerprints(
  content: OpenAIMessage["content"]
): string[] {
  if (!content || typeof content === "string" || !Array.isArray(content)) {
    return [];
  }

  const fps: string[] = [];
  for (const part of content) {
    const img = parsePartAsImage(part);
    if (!img) continue;
    if (img.kind === "inline") {
      fps.push(`img:${sha256Hex(`${img.mimeType}:${img.data}`)}`);
    } else {
      fps.push(`url:${sha256Hex(img.url)}`);
    }
  }
  return fps;
}

/**
 * Resolves a `ParsedImageSource` into an ACP `{ type: "image", mimeType, data }` block,
 * downloading remote `http(s)://` URLs if necessary.
 */
export async function resolveImageSourceToBlock(
  source: ParsedImageSource,
  abortSignal?: AbortSignal
): Promise<AcpPromptContentBlock> {
  if (source.kind === "inline") {
    return {
      type: "image",
      mimeType: source.mimeType,
      data: source.data,
    };
  }

  const timeoutSignal = AbortSignal.timeout(REMOTE_IMAGE_TIMEOUT_MS);
  const combinedSignal = abortSignal
    ? AbortSignal.any([abortSignal, timeoutSignal])
    : timeoutSignal;

  const res = await fetch(source.url, {
    signal: combinedSignal,
    headers: {
      Accept: "image/*,*/*;q=0.8",
    },
  });

  if (!res.ok) {
    throw new Error(
      `Failed to fetch remote image (${res.status} ${res.statusText}): ${source.url}`
    );
  }

  const contentLength = Number(res.headers.get("content-length") || "0");
  if (contentLength > MAX_REMOTE_IMAGE_BYTES) {
    throw new Error(
      `Remote image exceeds maximum size of ${MAX_REMOTE_IMAGE_BYTES} bytes`
    );
  }

  const arrayBuf = await res.arrayBuffer();
  if (arrayBuf.byteLength > MAX_REMOTE_IMAGE_BYTES) {
    throw new Error(
      `Remote image exceeds maximum size of ${MAX_REMOTE_IMAGE_BYTES} bytes`
    );
  }

  const buf = Buffer.from(arrayBuf);
  const data = buf.toString("base64");
  const headerMime = res.headers.get("content-type") || undefined;
  const sniffed = sniffImageMimeFromBuffer(buf);
  const mimeType = normalizeImageMimeType(sniffed || headerMime, data);

  return {
    type: "image",
    mimeType,
    data,
  };
}
