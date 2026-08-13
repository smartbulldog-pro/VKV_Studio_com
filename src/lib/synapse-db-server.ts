/**
 * synapse-db-server.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Background server sync for Synapse chat data.
 *
 * This module is entirely fire-and-forget. It NEVER affects local UI or state.
 * If the server is down, nothing breaks — data is already safe in IndexedDB.
 *
 * Used for:
 *  • Backing up conversation data to the server
 *  • Sending a final beacon on page visibility change
 */

import {
  db,
  getMessages,
  upsertServerConversation,
  type Conversation,
  type StoredMessage,
} from './synapse-db';
import { SYNAPSE_API_BASE } from './api-config';
import { getIdToken } from './auth';

const API_BASE = SYNAPSE_API_BASE;
const TIMEOUT = 6_000;

// Server-side history belongs to a VERIFIED Google account. Anonymous visitors
// have no server owner, so every sync call below is a no-op for them — their
// chats stay in IndexedDB only. `null` ⇒ not signed in ⇒ skip the request.
function authHeaders(base: Record<string, string> = {}): Record<string, string> | null {
  const token = getIdToken();
  if (!token) return null;
  return { ...base, Authorization: `Bearer ${token}` };
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface ServerConversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>;
}

// ─── Server push (background, non-blocking) ───────────────────────────────────

/**
 * Push a conversation + messages to the server.
 * Fire-and-forget: errors are silently logged.
 */
export function pushToServer(conv: Conversation, messages: StoredMessage[]): void {
  const headers = authHeaders({ 'Content-Type': 'application/json' });
  if (!headers) return; // anonymous → local-only, nothing to sync
  const payload: ServerConversation = {
    id: conv.id,
    title: conv.title,
    createdAt: conv.createdAt,
    updatedAt: conv.updatedAt,
    messages: messages.map((m) => ({
      id: m.msgId,
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
    })),
  };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT);

  fetch(`${API_BASE}/api/conversations`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: ctrl.signal,
  })
    .catch(() => {}) // silently ignore
    .finally(() => clearTimeout(timer));
}

/**
 * Push conversation data via sendBeacon (for visibilitychange / unload).
 * Uses text/plain to avoid CORS preflight.
 */
export function beaconSave(conv: Conversation, messages: StoredMessage[]): void {
  const token = getIdToken();
  if (!token) return; // anonymous → nothing to persist server-side
  const payload = JSON.stringify({
    id: conv.id,
    title: conv.title,
    createdAt: conv.createdAt,
    updatedAt: conv.updatedAt,
    // sendBeacon can't set headers → carry the ID token in the body; the backend
    // verifies it exactly like the Authorization header on the other endpoints.
    idToken: token,
    messages: messages.slice(-100).map((m) => ({
      id: m.msgId,
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
    })),
  });

  navigator.sendBeacon(
    `${API_BASE}/api/conversations/beacon`,
    new Blob([payload], { type: 'text/plain' })
  );
}

/**
 * Rename a conversation on the server.
 */
export function serverRename(convId: string, title: string): void {
  const headers = authHeaders({ 'Content-Type': 'application/json' });
  if (!headers) return;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT);

  fetch(`${API_BASE}/api/conversations/${convId}`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify({ title }),
    signal: ctrl.signal,
  })
    .catch(() => {})
    .finally(() => clearTimeout(timer));
}

/**
 * Delete a conversation on the server.
 */
export function serverDelete(convId: string): void {
  const headers = authHeaders();
  if (!headers) return;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT);

  fetch(`${API_BASE}/api/conversations/${convId}`, {
    method: 'DELETE',
    headers,
    signal: ctrl.signal,
  })
    .catch(() => {})
    .finally(() => clearTimeout(timer));
}

// ─── Server → local pull (cross-device sync) ──────────────────────────────────

interface ServerConvSummary {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
}

async function authedGet(path: string): Promise<Response | null> {
  const headers = authHeaders();
  if (!headers) return null;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT);
  try {
    return await fetch(`${API_BASE}${path}`, { headers, signal: ctrl.signal });
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Pull the signed-in account's conversations DOWN from the server into local
 * Dexie — the missing half of cross-device sync. For each server conversation
 * that is absent locally OR newer than the local copy, fetch it in full and
 * upsert it. Locally-newer conversations are left untouched (the push path syncs
 * those up). Anonymous → no-op. Returns true if anything local changed, so the
 * caller can refresh the UI.
 */
export async function pullFromServer(): Promise<boolean> {
  const listRes = await authedGet('/api/conversations');
  if (!listRes || !listRes.ok) return false;

  let serverConvs: ServerConvSummary[];
  try {
    serverConvs = (await listRes.json()).conversations ?? [];
  } catch {
    return false;
  }
  if (serverConvs.length === 0) return false;

  const local = await db.conversations.toArray();
  const localById = new Map(local.map((c) => [c.id, c]));

  let changed = false;
  for (const sc of serverConvs) {
    const lc = localById.get(sc.id);
    if (lc && sc.updatedAt <= lc.updatedAt) continue; // local is same-or-newer → keep it

    const fullRes = await authedGet(`/api/conversations/${sc.id}`);
    if (!fullRes || !fullRes.ok) continue;
    try {
      const full = await fullRes.json();
      const messages = (full.messages ?? []).map(
        (m: { id: string; role: 'user' | 'assistant'; content: string; timestamp: number }) => ({
          msgId: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
        })
      );
      if (messages.length === 0) continue;
      await upsertServerConversation(
        { id: full.id, title: full.title, createdAt: full.createdAt, updatedAt: full.updatedAt },
        messages
      );
      changed = true;
    } catch {
      // skip this conversation, keep going
    }
  }
  return changed;
}

// ─── Visibility change handler ────────────────────────────────────────────────

/**
 * Setup visibilitychange listener that sends a beacon when user
 * switches tabs or minimizes. More reliable than beforeunload.
 *
 * Call once from the component's onMount.
 * Returns cleanup function for onDestroy.
 */
export function setupVisibilitySync(getCurrentConvId: () => string): () => void {
  const handler = async () => {
    if (document.visibilityState !== 'hidden') return;

    const convId = getCurrentConvId();
    if (!convId) return;

    try {
      const conv = await db.conversations.get(convId);
      if (!conv) return;
      const msgs = await getMessages(convId);
      if (msgs.length === 0) return;

      beaconSave(conv, msgs);
    } catch {
      // Best-effort, don't throw
    }
  };

  document.addEventListener('visibilitychange', handler);
  return () => document.removeEventListener('visibilitychange', handler);
}
