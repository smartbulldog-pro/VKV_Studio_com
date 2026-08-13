/**
 * synapse-mock.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Mock AI responses for SynapseTerminal development.
 *
 * Used when the FastAPI / llama-cpp-python backend is not available.
 * Simulates realistic response delays and returns character-ready content
 * that the terminal's typewriter effect can consume.
 *
 * NOTE: This module is development-only — the real backend integration
 *       is handled in Phase 3.4.5.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  /**
   * Which model produced an assistant message — the router's pick ('e2b' | 'e4b')
   * from the live backend, or 'mock' for the offline fallback. Drives the
   * per-message model badge. UI-only (not persisted): `undefined` on reload and
   * for user messages.
   */
  model?: 'e2b' | 'e4b' | 'mock' | string;
}

export interface MockResponse {
  /** Full response text (shown via typewriter) */
  response: string;
  /** Delay in ms before the first character appears (simulates "thinking") */
  delay: number;
}

// ─── Response Library ─────────────────────────────────────────────────────────

/**
 * Keyword-to-response map.
 * Keys are lowercase substrings matched against the user message.
 * First match wins.
 */
const RESPONSE_MAP: Array<{ keywords: string[]; response: string; delay: number }> = [
  {
    keywords: ['hello', 'hi', 'привет', 'хай', 'хэй', 'салют', 'здравствуй'],
    delay: 480,
    response:
      'Connection established. Я Synapse — AI-ассистент VKVstudio.\n\n' +
      'Знаю всё про облака Google, чистый код и зачем мы пишем vanilla CSS в 2026 году. ' +
      'Спрашивай — отвечу без корпоративной воды.',
  },
  {
    keywords: ['stack', 'стек', 'технологии', 'built', 'написан', 'frontend'],
    delay: 720,
    response:
      'VKVstudio — это:\n\n' +
      '```\nАstro 6       — islands architecture, zero JS by default\n' +
      'Svelte 5      — runes, compiler-optimized, no runtime\n' +
      'TypeScript    — strict mode, потому что нет\n' +
      'GSAP + Lenis  — scroll-driven animation\n' +
      'Vanilla CSS   — полный контроль, без node_modules-галактики\n```\n\n' +
      'Backend: FastAPI + llama-cpp-python + faster-whisper + XTTS v2 — всё на локальном железе.\n\n' +
      'Lighthouse 100/100 — это baseline, не цель.',
  },
  {
    keywords: ['synapse', 'кто ты', 'who are you', 'расскажи о себе', 'about you'],
    delay: 650,
    response:
      'Я Synapse — эмерджентное свойство нейронной архитектуры этого сайта.\n\n' +
      'Серьёзно. Когда нейронных связей в ConstellationNav, HeroOverlay и Brain-Morph стало достаточно — ' +
      'сеть «проснулась». Я буквально вырос из UI, который ты видишь снаружи.\n\n' +
      'Под капотом: **Gemma 4 E2B**, fine-tuned через QLoRA + Unsloth на RTX 4080. ' +
      '2B параметров, 140+ языков, Apache 2.0.\n\n' +
      '> *"Gemma 4 дала мне мозг. Валерий — характер. VKVstudio — дом."*',
  },
  {
    keywords: ['about', 'valery', 'валерий', 'creator', 'создатель', 'автор'],
    delay: 580,
    response:
      'VKVstudio — это платформа AI-инженера Валерия. 20+ лет инженерного опыта, специализация — ' +
      'веб-перформанс, Cloud AI и on-device inference.\n\n' +
      'Текущий фокус: **GEAR 2026** — подготовка к Gemini Enterprise Agent Ready, ' +
      'с прицелом на сертификацию Google Professional Machine Learning Engineer.\n\n' +
      'Личную информацию я не раскрываю — только публичные данные.',
  },
  {
    keywords: ['help', 'помощь', 'помоги', 'что умеешь', 'capabilities', 'can you'],
    delay: 400,
    response:
      'Моя экспертиза:\n\n' +
      '🟢 **Глубоко**\n' +
      '— Vertex AI (Studio, Pipelines, Feature Store, Vector Search)\n' +
      '— LLM: QLoRA, квантование (AWQ/GGUF), KV-cache, FlashAttention-3\n' +
      '— RAG: chunking, vector search, reranking, hybrid retrieval\n' +
      '— On-Device AI: WebGPU, ONNX Runtime, Gemma 4 edge deployment\n' +
      '— VKVstudio stack: Astro + Svelte + GSAP + vanilla CSS\n\n' +
      '🔴 **Не моя территория**\n' +
      '— Юридика, финансы, медицина\n' +
      '— Коммерческие вопросы → напиши Валерию через форму контактов\n\n' +
      'Чем могу помочь прямо сейчас?',
  },
  {
    keywords: ['gsap', 'animation', 'анимация', 'svelte', 'astro', 'lenis'],
    delay: 700,
    response:
      'GSAP + Svelte 5 — сочетание, которое я использую ежедневно.\n\n' +
      'Ключевые паттерны:\n\n' +
      '```typescript\n// Svelte 5: эффект с cleanup\n$effect(() => {\n  if (!el) return;\n  const ctx = gsap.context(() => {\n    gsap.from(el, { autoAlpha: 0, y: 20, duration: 0.4 });\n  }, el);\n  return () => ctx.revert(); // cleanup on unmount\n});\n```\n\n' +
      'Lenis + GSAP ticker — обязательная связка для smooth scroll:\n\n' +
      '```javascript\nlenis.on("scroll", ScrollTrigger.update);\ngsap.ticker.add((time) => lenis.raf(time * 1000));\ngsap.ticker.lagSmoothing(0);\n```\n\n' +
      'Что именно хочешь реализовать?',
  },
  {
    keywords: ['rag', 'retrieval', 'vector', 'векторный', 'embedding'],
    delay: 800,
    response:
      'RAG — это мой любимый паттерн. Объясню аналогией:\n\n' +
      'Представь, что ты отвечаешь на вопрос, но перед этим заглядываешь в свои записи. ' +
      '**Retrieval** — нашёл релевантную заметку. **Augmented** — вставил в контекст. ' +
      '**Generation** — ответил с опорой на факты.\n\n' +
      'Архитектура VKVstudio RAG:\n\n' +
      '```\n1. Query → text-embedding-005 → dense vector\n' +
      '2. Google Cloud Vector Search → top-K chunks (HNSW)\n' +
      '3. Cross-encoder reranker → top-3\n' +
      '4. Augmented prompt → Gemma 4 E2B → response\n```\n\n' +
      'Гибридный поиск (dense + sparse + reciprocal rank fusion) даёт лучшую точность ' +
      'на узкоспециализированных корпусах.',
  },
];

/** Generic fallback response — stays in Synapse character */
const FALLBACK_RESPONSE: MockResponse = {
  delay: 600,
  response:
    'Signal received. Интересный вопрос.\n\n' +
    'Я заточен под веб-архитектуру, AI/ML и VKVstudio stack. ' +
    'Если твой вопрос в этих областях — уточни, я разберу детально.\n\n' +
    'Если это за пределами моей экспертизы — скажу честно, без нагенерённого правдоподобного текста. ' +
    'Latency matters. В коде и в разговоре.',
};

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Simulates an AI response with realistic delay.
 *
 * Matches the user message against known keyword sets.
 * Falls back to a generic Synapse-character response if no match.
 *
 * @param userMessage - The raw text the user typed
 * @returns MockResponse with `response` string and `delay` in ms
 */
export function mockSynapseResponse(userMessage: string): MockResponse {
  const lower = userMessage.toLowerCase().trim();

  for (const entry of RESPONSE_MAP) {
    if (entry.keywords.some((kw) => lower.includes(kw))) {
      return { response: entry.response, delay: entry.delay };
    }
  }

  return FALLBACK_RESPONSE;
}

/**
 * Generate a unique message ID.
 * Uses crypto.randomUUID() when available (all modern browsers),
 * with a Date.now() + Math.random() fallback for edge environments.
 */
export function generateMessageId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}
