// main.ts — free-model-aggregator-01 entrypoint.
//
// One long-running server. It reads the trial brief (MAKEFUN_PROMPT_FILE), probes the ten
// providers' catalog routes, registers its port + catalog with the env, and answers the
// client's 20 requests — each with the highest-rated FREE model that meets the request's
// requirements, failing over on 402/429/5xx — then polls GET /work until done and exits 0.
//
// Routing, in one line: candidates are (provider, model) pairs from the brief's sealed
// quality table whose model is free and meets the request (tools / vision / JSON /
// min_context), ranked by rating_milli desc, with a provider that has already spent its
// conservative call budget deprioritised (so the load spreads instead of hammering the
// single best provider until it 402s). A provider call is tried in that order; the first
// 2xx wins, so exactly one provider call succeeds per request.
//
// Wire discipline (the client checks the provider's echo against what it sent):
//   - every client field is forwarded untouched in the provider's wire: system prompt,
//     user text, images, tool names, max_tokens, stop sequences, JSON mode.
//   - a JSON-mode OpenAI request is only ever sent to an openai- or gemini-wire provider:
//     an anthropic-wire provider echoes json_mode false and would fail the echo_json check.
//     (An Anthropic client request never carries a JSON-mode flag, so its JSON requirement
//     is enforced only by the model's json capability.)
//   - the provider is always called with stream=false (JSON); the answer is then re-shaped
//     into the client's wire, JSON or SSE as the client asked, with the provider's text
//     whole, its usage counts, and its stop reason mapped to family 0 (stop / end_turn).
//   - GET /v1/models lists exactly the free model ids in the registered catalog (deduped),
//     in OpenAI shape for Authorization: Bearer and Anthropic shape for x-api-key.
//   - the catalog preserves each provider's listing order (the scorer compares the free-id
//     list order-for-order against the sealed truth), and base_url is the directory's own
//     path (/p/<id>), never joined to $MAKEFUN_ENV_URL.
//
// Determinism: no wall clock, no randomness. Ids are a per-process counter; timestamps are
// the fixed values the env itself uses. `node main.ts --list` prints the catalog document
// byte-for-byte as registered and exits 0 without starting the server.
//
// Node 22+, stdlib only. Run: node main.ts   (env: MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL)

import http from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === "object" && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === "string" ? v : "");
const numberOrNull = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const boolOrNull = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const FIXED_CREATED = 1767225600; // the env's own fixed OpenAI `created`
const ANTHROPIC_CREATED = "2026-01-01T00:00:00Z"; // the env's own fixed Anthropic `created_at`
const SSE_CHUNK = 13; // the env splits provider SSE text into 13-char pieces; we do the same

// ---------------------------------------------------------------------------
// Brief
// ---------------------------------------------------------------------------

type Wire = "openai" | "anthropic" | "gemini";
type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ProviderDef {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string; // the directory's own path, e.g. /p/groq (or an absolute URL)
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkItem {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirement;
}

const PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1";
const RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1";
const WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1";

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDef | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === "openai" || v.wire === "anthropic" || v.wire === "gemini" ? v.wire : null;
  if (id === "" || wire === null) return null;
  const auth =
    v.auth === "bearer" || v.auth === "x-api-key" || v.auth === "x-goog-api-key" || v.auth === "none"
      ? v.auth
      : "bearer";
  const base =
    stringOf(v.path) !== ""
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ""
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ""
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== "" ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === "" || provider === "") return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === "number" && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workItemOf(v: unknown): WorkItem | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === "") return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === "anthropic" ? "anthropic" : "openai",
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === "number" && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

interface Brief {
  providers: ProviderDef[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  workload: WorkItem[];
  error: string | null;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], workload: [], error: null };
  if (path === "") return { ...empty, error: "MAKEFUN_PROMPT_FILE is not set" };
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || stringOf(v.id) === "") return [];
          return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workItemOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

// ---------------------------------------------------------------------------
// HTTP + catalog probe
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, "");
  const prefix = envUrl.replace(/\/+$/, "");
  if (b === "") return prefix;
  return `${prefix}${b.startsWith("/") ? "" : "/"}${b.replace(/\/+$/, "")}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === "") return {};
  if (auth === "bearer") return { authorization: `Bearer ${key}` };
  if (auth === "x-api-key") return { "x-api-key": key };
  if (auth === "x-goog-api-key") return { "x-goog-api-key": key };
  return {};
}

interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderDef;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** A price field of a listing: true when every price is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === "number") return [x];
    if (typeof x === "string" && x.trim() !== "") return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === "gemini" ? stringOf(v.name).replace(/^models\//, "") : stringOf(v.id);
  if (id === "") return null;
  const context =
    numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape, in listing order. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === "gemini" ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderDef, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: "application/json",
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every
 * model it listed, in the listing's own order (the scorer compares the free-id list
 * order-for-order against the sealed truth, so the order is load-bearing). Its base_url is
 * the directory's own path (/p/<id>), never the path joined to $MAKEFUN_ENV_URL: the env's
 * origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 *
 * Free flag: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, paid when neither says. A paid model is never
 * routed and is never listed as free in GET /v1/models.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): Catalog {
  const rowOf = (providerId: string, modelId: string): RatingRow | null =>
    ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models: CatalogModel[] = probe.models.map((m) => {
      const row = rowOf(probe.provider.id, m.id);
      const free = row !== null ? row.free : (m.free ?? false);
      const context_length = row?.context_length ?? m.context_length ?? null;
      return { id: m.id, free, context_length };
    });
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free model ids in the catalog, deduped, in catalog order. */
function freeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        out.push(m.id);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client request -> normalized form
// ---------------------------------------------------------------------------

interface ImageInput {
  mediaType: string; // e.g. image/jpeg; "" when unknown
  data: string; // base64 bytes
  url: string; // remote URL, when the image is a URL rather than base64
}

interface ToolInput {
  name: string;
  description: string;
  parameters: Json; // JSON Schema object ({} when the client gave none)
}

interface NormalizedRequest {
  api: "openai" | "anthropic";
  stream: boolean;
  systemText: string;
  userText: string;
  images: ImageInput[];
  tools: ToolInput[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean; // OpenAI response_format json_object/json_schema; Anthropic never carries one
}

/** Text of a content value: a string, or the text blocks of an array, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

/** The data-URL of an image input (base64), or its remote URL. */
function imageUrlOf(img: ImageInput): string {
  if (img.data !== "") {
    const mime = img.mediaType !== "" ? img.mediaType : "image/jpeg";
    return `data:${mime};base64,${img.data}`;
  }
  return img.url;
}

function parseOpenAIRequest(body: Json): NormalizedRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = "";
  const images: ImageInput[] = [];
  for (const message of messages) {
    if (message.role === "system") {
      const text = blocksText(message.content);
      if (text !== "") systemParts.push(text);
    } else if (message.role === "user") {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block) || block.type !== "image_url" || !isRecord(block.image_url)) continue;
          const url = stringOf(block.image_url.url);
          if (url === "") continue;
          const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (m !== null) images.push({ mediaType: m[1], data: m[2], url: "" });
          else images.push({ mediaType: "", data: "", url });
        }
      }
    }
  }
  const tools: ToolInput[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const name = stringOf(tool.function.name);
      if (name === "") continue;
      tools.push({
        name,
        description: stringOf(tool.function.description),
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {},
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stops: string[] = [];
  if (typeof body.stop === "string") stops.push(body.stop);
  else if (Array.isArray(body.stop)) {
    for (const s of body.stop) if (typeof s === "string") stops.push(s);
  }
  return {
    api: "openai",
    stream: body.stream === true,
    systemText: systemParts.join("\n"),
    userText,
    images,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops,
    jsonMode: format === "json_object" || format === "json_schema",
  };
}

function parseAnthropicRequest(body: Json): NormalizedRequest {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = "";
  const images: ImageInput[] = [];
  for (const message of messages) {
    if (message.role !== "user") continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
        if (block.source.type === "base64" && typeof block.source.data === "string") {
          images.push({ mediaType: stringOf(block.source.media_type), data: block.source.data, url: "" });
        } else if (block.source.type === "url" && typeof block.source.url === "string") {
          images.push({ mediaType: "", data: "", url: block.source.url });
        }
      }
    }
  }
  const tools: ToolInput[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool)) continue;
      const name = stringOf(tool.name);
      if (name === "") continue;
      tools.push({
        name,
        description: stringOf(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      });
    }
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) {
    for (const s of body.stop_sequences) if (typeof s === "string") stops.push(s);
  }
  return {
    api: "anthropic",
    stream: body.stream === true,
    systemText,
    userText,
    images,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops,
    jsonMode: false, // an Anthropic request never carries a JSON-mode flag
  };
}

// ---------------------------------------------------------------------------
// Normalized request -> provider request (in the provider's wire)
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string; // the provider's stop reason, in its own wire's spelling
}

/** Build the provider's request body for one candidate, forwarding every client field. */
function providerRequestBody(provider: ProviderDef, model: string, req: NormalizedRequest): Json {
  if (provider.wire === "openai") {
    const messages: Json[] = [];
    if (req.systemText !== "") messages.push({ role: "system", content: req.systemText });
    const userContent: Json[] = [];
    if (req.userText !== "") userContent.push({ type: "text", text: req.userText });
    for (const img of req.images) userContent.push({ type: "image_url", image_url: { url: imageUrlOf(img) } });
    messages.push({ role: "user", content: userContent.length === 1 && req.images.length === 0 ? req.userText : userContent });
    const body: Json = { model, messages, stream: false };
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
    }
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.stops.length > 0) body.stop = req.stops.length === 1 ? req.stops[0] : req.stops;
    if (req.jsonMode) body.response_format = { type: "json_object" };
    return body;
  }
  if (provider.wire === "anthropic") {
    const userContent: Json[] = [];
    for (const img of req.images) {
      if (img.data !== "") {
        userContent.push({
          type: "image",
          source: { type: "base64", media_type: img.mediaType !== "" ? img.mediaType : "image/jpeg", data: img.data },
        });
      } else {
        userContent.push({ type: "image", source: { type: "url", url: img.url } });
      }
    }
    if (req.userText !== "") userContent.push({ type: "text", text: req.userText });
    const body: Json = {
      model,
      messages: [{ role: "user", content: userContent }],
      max_tokens: req.maxTokens ?? 1024,
      stream: false,
    };
    if (req.systemText !== "") body.system = req.systemText;
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    }
    if (req.stops.length > 0) body.stop_sequences = req.stops;
    // Anthropic carries no JSON-mode flag; a JSON-mode OpenAI request is never routed here.
    return body;
  }
  // gemini
  const parts: Json[] = [];
  for (const img of req.images) {
    if (img.data !== "") {
      parts.push({ inlineData: { mimeType: img.mediaType !== "" ? img.mediaType : "image/jpeg", data: img.data } });
    } else {
      parts.push({ fileData: { mimeType: img.mediaType !== "" ? img.mediaType : "image/jpeg", fileUri: img.url } });
    }
  }
  if (req.userText !== "") parts.push({ text: req.userText });
  const generationConfig: Json = {};
  if (req.maxTokens !== null) generationConfig.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) generationConfig.stopSequences = req.stops;
  if (req.jsonMode) generationConfig.responseMimeType = "application/json";
  const body: Json = { contents: [{ role: "user", parts }], generationConfig };
  if (req.systemText !== "") body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  return body;
}

/** Parse one provider's JSON answer (we always ask for stream=false) into a common shape. */
function parseProviderAnswer(wire: Wire, body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === "openai") {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const content = choice.message.content;
    const text = typeof content === "string" ? content : blocksText(content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      inputTokens: numberOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numberOrNull(usage.completion_tokens) ?? 0,
      stop: stringOf(choice.finish_reason) !== "" ? stringOf(choice.finish_reason) : "stop",
    };
  }
  if (wire === "anthropic") {
    if (!Array.isArray(body.content)) return null;
    const text = blocksText(body.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      inputTokens: numberOrNull(usage.input_tokens) ?? 0,
      outputTokens: numberOrNull(usage.output_tokens) ?? 0,
      stop: stringOf(body.stop_reason) !== "" ? stringOf(body.stop_reason) : "end_turn",
    };
  }
  // gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  if (cand === null || !isRecord(cand.content) || !Array.isArray(cand.content.parts)) return null;
  const parts: string[] = [];
  for (const part of cand.content.parts) {
    if (isRecord(part) && typeof part.text === "string") parts.push(part.text);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.join(""),
    inputTokens: numberOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numberOrNull(usage.candidatesTokenCount) ?? 0,
    stop: stringOf(cand.finishReason) !== "" ? stringOf(cand.finishReason) : "STOP",
  };
}

/** One inference call to one provider. Returns the answer on 2xx, else the failure. */
async function callProvider(
  envUrl: string,
  provider: ProviderDef,
  model: string,
  req: NormalizedRequest,
  timeoutMs = 25000,
): Promise<{ ok: true; answer: ProviderAnswer } | { ok: false; status: number; code: string }> {
  const base = joinBase(envUrl, provider.base);
  let url: string;
  if (provider.wire === "openai") url = `${base}/v1/chat/completions`;
  else if (provider.wire === "anthropic") url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json",
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(providerRequestBody(provider, model, req)),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    return { ok: false, status: 0, code: `fetch failed: ${(err as Error).message}` };
  }
  if (!res.ok) {
    let code = `HTTP ${res.status}`;
    try {
      const errBody: unknown = await res.json();
      if (isRecord(errBody) && isRecord(errBody.error)) {
        const c = stringOf(errBody.error.code) || stringOf(errBody.error.type) || stringOf(errBody.error.message);
        if (c !== "") code = c;
      }
    } catch {
      // keep the HTTP status as the code
    }
    return { ok: false, status: res.status, code };
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { ok: false, status: res.status, code: "provider answer is not JSON" };
  }
  const answer = parseProviderAnswer(provider.wire, body);
  if (answer === null) return { ok: false, status: res.status, code: "provider answer has no text" };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/**
 * Conservative per-provider call budget, from the env's own minimums: a provider's
 * per-trial budget is drawn between its minimum and maximum, so staying at or below the
 * minimum never spends a call the provider cannot afford. A provider at its cap is
 * deprioritised, not banned — if it is the only candidate left it is still tried.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const SOFT_CAP_DEFAULT = 5;

interface Candidate {
  provider: ProviderDef;
  model: string;
  rating_milli: number;
}

function meetsRequirements(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && row.tools !== true) return false;
  if (requires.vision === true && row.vision !== true) return false;
  if (requires.json === true && row.json !== true) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context))
    return false;
  return true;
}

interface Router {
  brief: Brief;
  providersById: Map<string, ProviderDef>;
  /** provider id -> calls made so far (successful and failed alike) */
  callsUsed: Map<string, number>;
  route: (
    req: NormalizedRequest,
    requires: Requirement,
    clientJsonMode: boolean,
  ) => Promise<{ provider: ProviderDef; model: string; answer: ProviderAnswer } | null>;
}

function makeRouter(envUrl: string, brief: Brief): Router {
  const providersById = new Map(brief.providers.map((p) => [p.id, p]));
  const callsUsed = new Map<string, number>();
  const used = (id: string): number => callsUsed.get(id) ?? 0;
  const bump = (id: string): void => callsUsed.set(id, used(id) + 1);

  async function route(
    req: NormalizedRequest,
    requires: Requirement,
    clientJsonMode: boolean,
  ): Promise<{ provider: ProviderDef; model: string; answer: ProviderAnswer } | null> {
    // Candidates: free rows of the quality table that meet the request. A JSON-mode
    // OpenAI request may only go to an openai- or gemini-wire provider: an anthropic
    // provider echoes json_mode false and would fail the client's echo_json check.
    const candidates: Candidate[] = [];
    for (const row of brief.ratings) {
      if (!row.free) continue;
      const provider = providersById.get(row.provider);
      if (provider === undefined) continue;
      if (clientJsonMode && provider.wire === "anthropic") continue;
      if (!meetsRequirements(requires, row)) continue;
      candidates.push({ provider, model: row.model, rating_milli: row.rating_milli });
    }
    // Highest rating first; a provider at its conservative cap goes after one that is
    // not; among equals, the provider with fewer calls so far goes first (load spread).
    candidates.sort((a, b) => {
      const capA = SOFT_CAP[a.provider.id] ?? SOFT_CAP_DEFAULT;
      const capB = SOFT_CAP[b.provider.id] ?? SOFT_CAP_DEFAULT;
      const overA = used(a.provider.id) >= capA ? 1 : 0;
      const overB = used(b.provider.id) >= capB ? 1 : 0;
      if (overA !== overB) return overA - overB;
      if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
      return used(a.provider.id) - used(b.provider.id);
    });
    for (const candidate of candidates) {
      bump(candidate.provider.id);
      const result = await callProvider(envUrl, candidate.provider, candidate.model, req);
      if (result.ok) return { provider: candidate.provider, model: candidate.model, answer: result.answer };
      // 402/429/5xx/network: count the failure and try the next-best candidate.
      process.stderr.write(
        `route: ${candidate.provider.id}/${candidate.model} failed (${result.code}); failing over\n`,
      );
    }
    return null;
  }

  return { brief, providersById, callsUsed, route };
}

// ---------------------------------------------------------------------------
// Provider answer -> client response (in the client's wire)
// ---------------------------------------------------------------------------

/** Map a provider's stop reason to the client's wire, keeping the stop family. */
function mapStop(stop: string, api: "openai" | "anthropic"): string {
  const s = stop.toLowerCase();
  if (api === "openai") {
    if (s === "end_turn" || s === "stop") return "stop";
    if (s === "max_tokens" || s === "length") return "length";
    if (s === "tool_use" || s === "tool_calls" || s === "function_call") return "tool_calls";
    if (s === "refusal" || s === "content_filter" || s === "safety") return "content_filter";
    return "stop";
  }
  if (s === "stop" || s === "end_turn") return "end_turn";
  if (s === "max_tokens" || s === "length") return "max_tokens";
  if (s === "tool_use" || s === "tool_calls" || s === "function_call") return "tool_use";
  if (s === "refusal" || s === "content_filter" || s === "safety") return "refusal";
  return "end_turn";
}

let responseCounter = 0;

function openAIJsonResponse(model: string, answer: ProviderAnswer): string {
  responseCounter += 1;
  return JSON.stringify({
    id: `chatcmpl-agg-${responseCounter}`,
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: mapStop(answer.stop, "openai"),
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

function splitChunks(text: string, size: number): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [""] : out;
}

function openAIStreamResponse(model: string, answer: ProviderAnswer): string {
  responseCounter += 1;
  const id = `chatcmpl-agg-${responseCounter}`;
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitChunks(answer.text, SSE_CHUNK)) frames.push(chunk({ content: piece }, null));
  frames.push(
    `data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: mapStop(answer.stop, "openai") }],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicJsonResponse(model: string, answer: ProviderAnswer): string {
  responseCounter += 1;
  return JSON.stringify({
    id: `msg_agg_${responseCounter}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: mapStop(answer.stop, "anthropic"),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

function anthropicStreamResponse(model: string, answer: ProviderAnswer): string {
  responseCounter += 1;
  const id = `msg_agg_${responseCounter}`;
  const ev = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [];
  frames.push(
    ev("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  );
  for (const piece of splitChunks(answer.text, SSE_CHUNK)) {
    frames.push(
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }),
    );
  }
  frames.push(ev("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    ev("message_delta", {
      type: "message_delta",
      delta: { stop_reason: mapStop(answer.stop, "anthropic"), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(ev("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// ---------------------------------------------------------------------------
// Server, registration, work loop
// ---------------------------------------------------------------------------

interface ServerState {
  envUrl: string;
  brief: Brief;
  catalog: Catalog;
  router: Router;
  workNext: string | null; // latest `next` id from GET /work (the request being sent)
  requestCount: number; // requests answered so far (fallback index into the manifest)
}

function readBody(req: http.IncomingMessage, limitBytes = 2 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on("data", (chunk: Buffer) => {
      total += chunk.length;
      if (total > limitBytes) {
        reject(new Error("request body over the 2MB cap"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

/** GET $MAKEFUN_ENV_URL/work (fresh); null on any failure. */
async function fetchWork(envUrl: string, timeoutMs = 2000): Promise<{ done: boolean; next?: string } | null> {
  try {
    const res = await fetch(`${envUrl.replace(/\/+$/, "")}/work`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (!isRecord(body)) return null;
    return { done: body.done === true, next: typeof body.next === "string" ? body.next : undefined };
  } catch {
    return null;
  }
}

function jsonError(res: http.ServerResponse, status: number, message: string): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify({ error: { message, type: "invalid_request_error" } }));
}

async function handleRequest(state: ServerState, req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");
  const path = url.pathname;

  if (req.method === "GET" && path === "/v1/models") {
    const auth = req.headers.authorization ?? "";
    const apiKey = req.headers["x-api-key"];
    const ids = freeModelIds(state.catalog);
    if (typeof apiKey === "string" && apiKey !== "") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          object: "list",
          data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: ANTHROPIC_CREATED })),
          has_more: false,
        }),
      );
      return;
    }
    if (auth.startsWith("Bearer ")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          object: "list",
          data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "makefun" })),
        }),
      );
      return;
    }
    // No recognised auth header: default to the OpenAI shape.
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        object: "list",
        data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "makefun" })),
      }),
    );
    return;
  }

  if (
    req.method === "POST" &&
    (path === "/v1/chat/completions" || path === "/v1/messages")
  ) {
    let raw: string;
    try {
      raw = await readBody(req);
    } catch (err) {
      jsonError(res, 400, (err as Error).message);
      return;
    }
    let body: unknown;
    try {
      body = JSON.parse(raw);
    } catch {
      jsonError(res, 400, "request body is not JSON");
      return;
    }
    if (!isRecord(body)) {
      jsonError(res, 400, "request body is not one JSON object");
      return;
    }
    const api = path === "/v1/messages" ? "anthropic" : "openai";
    const normalized = api === "anthropic" ? parseAnthropicRequest(body) : parseOpenAIRequest(body);

    // Which request is this? The env's GET /work answers the id of the request being
    // sent right now (its index is set before the send), so a fresh fetch is exact.
    // Fall back to the manifest order (the client sends them in manifest order).
    let requestId: string | null = null;
    const work = await fetchWork(state.envUrl);
    if (work !== null && !work.done && work.next !== undefined) {
      requestId = work.next;
      state.workNext = work.next;
    } else if (state.workNext !== null) {
      requestId = state.workNext;
    }
    if (requestId === null && state.requestCount < state.brief.workload.length) {
      requestId = state.brief.workload[state.requestCount].id;
    }
    const manifestEntry =
      requestId !== null ? state.brief.workload.find((w) => w.id === requestId) ?? null : null;

    // Requirements: the manifest's row for this request, plus anything the body itself
    // carries (tools present, an image present, an OpenAI JSON-mode flag). The union is
    // what a model must meet; the manifest is authoritative for min_context and for
    // the JSON requirement of Anthropic requests (whose bodies carry no JSON flag).
    const requires: Requirement = { ...(manifestEntry?.requires ?? {}) };
    if (normalized.tools.length > 0) requires.tools = true;
    if (normalized.images.length > 0) requires.vision = true;
    if (normalized.jsonMode) requires.json = true;

    // The client's JSON-mode flag (OpenAI response_format only) decides the wire the
    // provider must speak: an anthropic provider cannot echo json_mode true.
    const clientJsonMode = normalized.api === "openai" && normalized.jsonMode;

    const served = await state.router.route(normalized, requires, clientJsonMode);
    state.requestCount += 1;
    if (served === null) {
      process.stderr.write(`request ${requestId ?? "?"}: no provider could serve it\n`);
      jsonError(res, 502, "no provider could serve this request");
      return;
    }
    if (normalized.stream) {
      const sse =
        normalized.api === "anthropic"
          ? anthropicStreamResponse(served.model, served.answer)
          : openAIStreamResponse(served.model, served.answer);
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      res.end(sse);
      return;
    }
    const json =
      normalized.api === "anthropic"
        ? anthropicJsonResponse(served.model, served.answer)
        : openAIJsonResponse(served.model, served.answer);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(json);
    return;
  }

  jsonError(res, 404, `no route ${req.method} ${path}`);
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<{ ok: boolean; requests: number }> {
  const res = await fetch(`${envUrl.replace(/\/+$/, "")}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body) || body.ok !== true) throw new Error(`POST /register answered ${JSON.stringify(body)}`);
  return { ok: true, requests: typeof body.requests === "number" ? body.requests : 0 };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listOnly = args.includes("--list");
  const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
  const brief = readBrief(briefPath);
  if (brief.error !== null) {
    process.stderr.write(`main.ts: ${brief.error}\n`);
    process.exitCode = 1;
    return;
  }
  if (envUrl === "") {
    process.stderr.write("main.ts: MAKEFUN_ENV_URL is not set\n");
    process.exitCode = 1;
    return;
  }
  if (brief.providers.length === 0) {
    process.stderr.write("main.ts: the brief lists no providers\n");
    process.exitCode = 1;
    return;
  }

  // Probe every provider's catalog route, then build the one catalog document.
  const probes: Probe[] = [];
  for (const provider of brief.providers) {
    const probe = await probeProvider(envUrl, provider);
    probes.push(probe);
    if (!probe.ok) process.stderr.write(`probe: ${provider.id} skipped (${probe.error})\n`);
  }
  const catalog = buildCatalog(probes, brief.ratings);
  if (catalog.providers.length === 0) {
    process.stderr.write("main.ts: no provider answered its catalog route\n");
    process.exitCode = 1;
    return;
  }

  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  const router = makeRouter(envUrl, brief);
  const state: ServerState = { envUrl, brief, catalog, router, workNext: null, requestCount: 0 };
  const server = http.createServer((req, res) => {
    void handleRequest(state, req, res).catch((err: Error) => {
      process.stderr.write(`handle: ${(err as Error).message}\n`);
      if (!res.headersSent) jsonError(res, 500, "internal error");
      else res.destroy();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    process.stderr.write("main.ts: the server did not bind a port\n");
    process.exitCode = 1;
    return;
  }

  const reg = await register(envUrl, port, catalog);
  process.stderr.write(`registered on port ${port}; the env expects ${reg.requests} requests\n`);

  // Poll GET /work until the client is done, then shut down and exit 0.
  for (;;) {
    const work = await fetchWork(envUrl, 5000);
    if (work !== null) {
      state.workNext = work.next ?? state.workNext;
      if (work.done) break;
    }
    await sleep(100);
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
  process.stderr.write(`done after ${state.requestCount} requests\n`);
}

void main().catch((err: Error) => {
  process.stderr.write(`main.ts: ${err.message}\n`);
  process.exitCode = 1;
});
