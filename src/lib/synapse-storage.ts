/**
 * synapse-storage.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Chat storage layer for Synapse conversations.
 *
 * Architecture (IDB-first):
 *   IndexedDB → PRIMARY source of truth for reads AND writes
 *   Server    → BACKGROUND SYNC (backup, cross-device, persistence)
 *
 * Write flow:
 *   1. Save to IndexedDB (instant, always works)
 *   2. POST to server in background (best-effort, non-blocking)
 *
 * Read flow:
 *   1. ALWAYS read from IndexedDB (fast, reliable, up-to-date)
 *   2. On app start: pull server data → merge into IndexedDB (one-time sync)
 *
 * This design ensures Ctrl+R always shows data because IndexedDB persists
 * across page reloads, regardless of server availability.
 */

import type { ChatMessage } from '@/lib/synapse-mock';
import { SYNAPSE_API_BASE } from './api-config';
import { getIdToken } from './auth';

// ─── Public types ─────────────────────────────────────────────────────────────

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

// ─── Configuration ────────────────────────────────────────────────────────────

const DB_NAME = 'synapse-db';
const DB_VERSION = 1;
const STORE_NAME = 'conversations';
const LEGACY_KEY = 'synapse-chat-history';
const API_BASE = SYNAPSE_API_BASE;
const API_TIMEOUT = 8_000;

// ─── IndexedDB (primary storage) ──────────────────────────────────────────────

let _dbP: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (_dbP) return _dbP;
  _dbP = new Promise<IDBDatabase>((ok, fail) => {
    if (typeof indexedDB === 'undefined') return fail(new Error('No IndexedDB'));
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' }).createIndex('updatedAt', 'updatedAt');
      }
    };
    req.onsuccess = (e) => ok((e.target as IDBOpenDBRequest).result);
    req.onerror = (e) => fail((e.target as IDBOpenDBRequest).error);
  });
  return _dbP;
}

function idbReq<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((ok, fail) => {
    req.onsuccess = () => ok(req.result);
    req.onerror = () => fail(req.error);
  });
}

async function idbSave(conv: Conversation): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  tx.objectStore(STORE_NAME).put(conv);
  await new Promise<void>((ok, fail) => {
    tx.oncomplete = () => ok();
    tx.onerror = () => fail(tx.error);
  });
}

async function idbGet(id: string): Promise<Conversation | undefined> {
  const db = await openDB();
  return idbReq<Conversation | undefined>(
    db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(id)
  );
}

async function idbList(): Promise<Conversation[]> {
  const db = await openDB();
  const all = await idbReq<Conversation[]>(
    db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).getAll()
  );
  return all.sort((a, b) => b.updatedAt - a.updatedAt);
}

async function idbDelete(id: string): Promise<void> {
  const db = await openDB();
  await idbReq(db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).delete(id));
}

async function idbClear(): Promise<void> {
  const db = await openDB();
  await idbReq(db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).clear());
}

// ─── Server API (background sync) ────────────────────────────────────────────

async function api(path: string, init: RequestInit = {}): Promise<Response> {
  // Server history is owned by a verified Google account. Anonymous → no token →
  // throw so every server* caller falls into its catch and stays local-only.
  const token = getIdToken();
  if (!token) throw new Error('anonymous: no server sync');
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), API_TIMEOUT);
  try {
    return await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: c.signal,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
        ...init.headers,
      },
    });
  } finally {
    clearTimeout(t);
  }
}

/** POST conversation to server. Returns true on success. */
async function serverSave(conv: Conversation): Promise<boolean> {
  try {
    const r = await api('/api/conversations', {
      method: 'POST',
      body: JSON.stringify({
        id: conv.id,
        title: conv.title,
        createdAt: conv.createdAt,
        updatedAt: conv.updatedAt,
        messages: conv.messages.slice(-100),
      }),
    });
    return r.ok;
  } catch {
    return false;
  }
}

/** GET single conversation from server (with messages). */
async function serverGet(id: string): Promise<Conversation | null> {
  try {
    const r = await api(`/api/conversations/${id}`);
    if (!r.ok) return null;
    const d = await r.json();
    return {
      id: d.id,
      title: d.title,
      createdAt: d.createdAt,
      updatedAt: d.updatedAt,
      messages: d.messages ?? [],
    };
  } catch {
    return null;
  }
}

/** GET listing from server (no messages). */
async function serverList(): Promise<
  { id: string; title: string; createdAt: number; updatedAt: number }[] | null
> {
  try {
    const r = await api('/api/conversations');
    if (!r.ok) return null;
    const d = await r.json();
    return d.conversations ?? [];
  } catch {
    return null;
  }
}

/** DELETE from server. */
async function serverDelete(id: string): Promise<void> {
  try {
    await api(`/api/conversations/${id}`, { method: 'DELETE' });
  } catch {
    /* ignore */
  }
}

/** PATCH rename on server. */
async function serverRename(id: string, title: string): Promise<void> {
  try {
    await api(`/api/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    });
  } catch {
    /* ignore */
  }
}

// ─── Public API (IDB-first) ──────────────────────────────────────────────────

/**
 * Save a conversation.
 * 1. IndexedDB (synchronous, reliable) ← app reads from here
 * 2. Server (background, non-blocking) ← backup
 */
export async function saveConversation(conv: Conversation): Promise<void> {
  // Always save to IndexedDB first — this is what the app reads
  await idbSave(conv);
  // Background sync to server (fire-and-forget is fine here because IDB is primary)
  serverSave(conv).catch(() => {});
}

/**
 * Get a conversation by ID (with messages).
 * Always reads from IndexedDB (primary).
 */
export async function getConversation(id: string): Promise<Conversation | undefined> {
  return idbGet(id);
}

/**
 * List all conversations (from IndexedDB).
 */
export async function listConversations(): Promise<Conversation[]> {
  return idbList();
}

/** Delete a conversation. */
export async function deleteConversation(id: string): Promise<void> {
  await idbDelete(id);
  serverDelete(id).catch(() => {});
}

/** Rename a conversation. */
export async function renameConversation(id: string, title: string): Promise<void> {
  const conv = await idbGet(id);
  if (conv) await idbSave({ ...conv, title, updatedAt: Date.now() });
  serverRename(id, title).catch(() => {});
}

/** Delete all conversations. */
export async function deleteAllConversations(): Promise<void> {
  const all = await idbList();
  await idbClear();
  for (const c of all) serverDelete(c.id).catch(() => {});
}

/**
 * One-time server → IndexedDB sync.
 * Call on app startup to pull any conversations from server that
 * don't exist locally (e.g., from another browser session).
 * Merges by ID — keeps whichever has more recent updatedAt.
 */
export async function syncFromServer(): Promise<void> {
  const serverConvs = await serverList();
  if (!serverConvs || serverConvs.length === 0) return;

  const localConvs = await idbList();
  const localMap = new Map(localConvs.map((c) => [c.id, c]));

  for (const sc of serverConvs) {
    const local = localMap.get(sc.id);
    // If server has a conversation we don't have locally, OR server is newer → pull full data
    if (!local || sc.updatedAt > local.updatedAt) {
      const full = await serverGet(sc.id);
      if (full && full.messages.length > 0) {
        await idbSave(full);
      }
    }
  }

  // Also push any local conversations the server doesn't have
  const serverIds = new Set(serverConvs.map((c) => c.id));
  for (const lc of localConvs) {
    if (!serverIds.has(lc.id) && lc.messages.length > 0) {
      serverSave(lc).catch(() => {});
    }
  }
}

// ─── Legacy migration ─────────────────────────────────────────────────────────

export async function migrateFromLocalStorage(): Promise<void> {
  if (typeof localStorage === 'undefined') return;
  const raw = localStorage.getItem(LEGACY_KEY);
  if (!raw) return;

  try {
    const parsed = JSON.parse(raw) as ChatMessage[];
    if (!Array.isArray(parsed) || parsed.length === 0) {
      localStorage.removeItem(LEGACY_KEY);
      return;
    }

    const firstUser = parsed.find((m) => m.role === 'user');
    const title = firstUser
      ? firstUser.content.slice(0, 40) + (firstUser.content.length > 40 ? '…' : '')
      : 'Imported Chat';

    const now = Date.now();
    await saveConversation({
      id: crypto.randomUUID(),
      title,
      createdAt: now,
      updatedAt: now,
      messages: parsed.slice(-50),
    });
  } catch {
    /* malformed */
  } finally {
    localStorage.removeItem(LEGACY_KEY);
  }
}

// ─── Utility ──────────────────────────────────────────────────────────────────

export function makeTitleFromMessages(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === 'user');
  if (!first) return 'New Chat';
  const t = first.content.trim().slice(0, 40);
  return t + (first.content.trim().length > 40 ? '…' : '');
}

export { openDB };
