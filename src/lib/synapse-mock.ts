/**
 * synapse-mock.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * The scripted fallback that answers when the inference backend cannot.
 *
 * This is NOT development-only, which is what this header used to claim. The
 * backend is a single self-hosted box; when it is asleep, cold-starting, rate
 * limiting or simply down, these are the words a visitor reads, under a MOCK
 * badge. For a stranger evaluating the site that makes this file part of the
 * product, and it has to obey the same two rules the rest of the site does:
 *
 *   1. ANSWER IN THE VISITOR'S LANGUAGE. Every response here was Russian —
 *      including the catch-all — so an English visitor who arrived while the
 *      box was down got a wall of Cyrillic from a site that had just addressed
 *      them in English. Each entry now carries both, chosen per message.
 *
 *   2. DESCRIBE THE STACK THAT EXISTS. The previous text advertised
 *      llama-cpp-python, faster-whisper and XTTS v2 running "on local hardware",
 *      and a RAG pipeline built on text-embedding-005 and Google Cloud Vector
 *      Search. None of that is true any more: serving moved to llama.cpp's
 *      llama-server on a CPU-only ARM box, speech-to-text was removed, speech
 *      output is Google Chirp 3 HD, and retrieval is a local corpus scored with
 *      EmbeddingGemma-300M. A fallback that invents an architecture is worse
 *      than no fallback — it is the site itself confabulating, on the one page
 *      that exists to argue the author does not.
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

/** The two languages the site speaks. */
export type MockLang = 'en' | 'ru';

// ─── Language detection ───────────────────────────────────────────────────────

/**
 * Which language to answer in, decided from the message alone.
 *
 * The client has no access to the UI locale — and should not answer from it
 * anyway: a Russian speaker reading the English pages still types Russian, and
 * a reviewer typing English on /ru/ still wants English back. The live backend
 * makes the same call the same way (`rag/retriever.py` classifies by Cyrillic
 * ratio), so the fallback behaves like the thing it is standing in for.
 *
 * Any Cyrillic at all means Russian. A Russian sentence rarely contains no
 * Cyrillic; an English one containing some is almost always a quotation of a
 * Russian word, and a Russian reply to that is the safer miss.
 */
export function detectMockLang(message: string): MockLang {
  return /[Ѐ-ӿ]/.test(message) ? 'ru' : 'en';
}

// ─── Response Library ─────────────────────────────────────────────────────────

interface MockEntry {
  /** Lowercase substrings matched against the user message. First match wins. */
  keywords: string[];
  delay: number;
  en: string;
  ru: string;
}

const RESPONSE_MAP: MockEntry[] = [
  {
    keywords: ['hello', 'hi', 'привет', 'хай', 'хэй', 'салют', 'здравствуй'],
    delay: 480,
    en:
      "Connection established. I'm Synapse — VKVstudio's assistant.\n\n" +
      'I know this studio: how it is built, what runs where, and why it still ships ' +
      'vanilla CSS in 2026. Ask — you will get an answer, not a brochure.',
    ru:
      'Connection established. Я Synapse — ассистент VKVstudio.\n\n' +
      'Я знаю эту студию: как она устроена, что где работает и почему здесь до сих пор ' +
      'пишут vanilla CSS в 2026 году. Спрашивай — отвечу без корпоративной воды.',
  },
  {
    keywords: ['stack', 'стек', 'технологии', 'built', 'написан', 'frontend'],
    delay: 720,
    en:
      'VKVstudio is:\n\n' +
      '```\nAstro 6       — islands architecture, zero JS by default\n' +
      'Svelte 5      — runes, compiled, no runtime\n' +
      'TypeScript    — strict, with noUncheckedIndexedAccess\n' +
      'GSAP + Lenis  — scroll-driven animation\n' +
      'Vanilla CSS   — design tokens, no framework\n```\n\n' +
      'Backend: Python FastAPI in front of llama.cpp — a fine-tuned Gemma 4 E2B for chat, ' +
      'EmbeddingGemma-300M for the Embedding Explorer, SQLite for history, ' +
      'Google Chirp 3 HD for speech. It runs on one self-hosted ARM server, on CPU.\n\n' +
      'That last detail is the interesting one: there is no GPU anywhere in production, so ' +
      'an answer takes tens of seconds rather than milliseconds. Slow was the accepted price ' +
      'of running it all on one small rented box instead of a rented accelerator.',
    ru:
      'VKVstudio — это:\n\n' +
      '```\nAstro 6       — islands architecture, zero JS by default\n' +
      'Svelte 5      — руны, компиляция, без рантайма\n' +
      'TypeScript    — strict, с noUncheckedIndexedAccess\n' +
      'GSAP + Lenis  — scroll-driven анимация\n' +
      'Vanilla CSS   — дизайн-токены, без фреймворка\n```\n\n' +
      'Бэкенд: Python FastAPI поверх llama.cpp — дообученная Gemma 4 E2B для чата, ' +
      'EmbeddingGemma-300M для Embedding Explorer, SQLite для истории, ' +
      'Google Chirp 3 HD для речи. Всё это крутится на одном своём ARM-сервере, на CPU.\n\n' +
      'Последняя деталь и есть самая интересная: GPU в продакшене нет вообще, поэтому ответ ' +
      'занимает десятки секунд, а не миллисекунды. Медленно — осознанная цена за то, что всё ' +
      'крутится на одной небольшой арендованной машине, а не на арендованном ускорителе.',
  },
  {
    keywords: ['synapse', 'кто ты', 'who are you', 'расскажи о себе', 'about you'],
    delay: 650,
    en:
      "I'm Synapse — VKVstudio's assistant, and I am small on purpose.\n\n" +
      'Under the hood: **Gemma 4 E2B**, fine-tuned with QLoRA on a single RTX 4080, ' +
      'quantised to GGUF and served by llama.cpp. Two billion parameters, open weights ' +
      'under Google’s Gemma license, no API bill behind it.\n\n' +
      'What that costs me: I am confidently wrong more often than a frontier model, ' +
      'so my factual claims are worth checking. What it buys: every word you get from me is ' +
      'generated on a server the studio runs itself, by a model it fine-tuned itself.\n\n' +
      '> *"Gemma 4 gave me a brain. Valerii gave me a character. VKVstudio is where I live."*',
    ru:
      'Я Synapse — ассистент VKVstudio, и я намеренно маленький.\n\n' +
      'Под капотом: **Gemma 4 E2B**, дообученная через QLoRA на одной RTX 4080, ' +
      'квантованная в GGUF и обслуживаемая llama.cpp. Два миллиарда параметров, открытые ' +
      'веса под лицензией Google Gemma, без счёта за API.\n\n' +
      'Чем я за это плачу: я уверенно ошибаюсь чаще, чем фронтирная модель, так что мои факты ' +
      'стоит перепроверять. Что я за это получаю: каждое слово генерируется на сервере, ' +
      'который студия держит сама, моделью, которую она сама дообучила.\n\n' +
      '> *"Gemma 4 дала мне мозг. Валерий — характер. VKVstudio — дом."*',
  },
  {
    // Both Latin spellings stay: 'Valerii' is the passport transliteration the
    // site now uses everywhere, but a visitor typing the older 'valery' should
    // still be understood. Keywords are for matching what someone types, not
    // for declaring the canonical name — dropping one only loses matches.
    keywords: ['about', 'valerii', 'valery', 'валерий', 'creator', 'создатель', 'автор'],
    delay: 580,
    en:
      'VKVstudio is the studio of Valerii, a full-stack AI/web engineer with 20 years ' +
      'behind him — photography, then web, then marketing, now AI engineering.\n\n' +
      'Current focus: **GEAR 2026** — Gemini Enterprise Agent Ready — and the Google ' +
      'Professional Machine Learning Engineer certification.\n\n' +
      'For anything commercial, write to hello@vkvstudio.com. Personal details about him ' +
      'I do not hand out; what is on this site is what is public.',
    ru:
      'VKVstudio — студия Валерия, full-stack инженера по AI и вебу с 20 годами за плечами: ' +
      'фотография, потом веб, потом маркетинг, теперь AI-инженерия.\n\n' +
      'Текущий фокус: **GEAR 2026** — Gemini Enterprise Agent Ready — и сертификация ' +
      'Google Professional Machine Learning Engineer.\n\n' +
      'По коммерческим вопросам пиши на hello@vkvstudio.com. Личные данные о нём я не выдаю: ' +
      'что есть на сайте — то и публично.',
  },
  {
    keywords: ['help', 'помощь', 'помоги', 'что умеешь', 'capabilities', 'can you'],
    delay: 400,
    en:
      'What I am for:\n\n' +
      '🟢 **Solid ground**\n' +
      '— How this site is built: Astro, Svelte 5 runes, GSAP, vanilla CSS\n' +
      '— How it is served: FastAPI, llama.cpp, GGUF quantisation, self-hosting\n' +
      '— LLM work: QLoRA fine-tuning, embeddings, retrieval, evaluation\n' +
      '— Google Cloud AI: Vertex AI, Gemma, the deployment trade-offs behind this box\n\n' +
      '🔴 **Not my territory**\n' +
      '— Law, finance, medicine — I will say so rather than improvise\n' +
      '— Hiring and pricing → hello@vkvstudio.com\n\n' +
      'What would you like to look at?',
    ru:
      'Чем я полезен:\n\n' +
      '🟢 **Твёрдая почва**\n' +
      '— Как устроен этот сайт: Astro, руны Svelte 5, GSAP, vanilla CSS\n' +
      '— Как он обслуживается: FastAPI, llama.cpp, квантование GGUF, самохостинг\n' +
      '— Работа с LLM: дообучение QLoRA, эмбеддинги, поиск, оценка качества\n' +
      '— Google Cloud AI: Vertex AI, Gemma и компромиссы, из которых вырос этот сервер\n\n' +
      '🔴 **Не моя территория**\n' +
      '— Право, финансы, медицина — скажу об этом прямо, а не начну сочинять\n' +
      '— Наём и цены → hello@vkvstudio.com\n\n' +
      'С чем разбираемся?',
  },
  {
    keywords: ['gsap', 'animation', 'анимация', 'svelte', 'astro', 'lenis'],
    delay: 700,
    en:
      'GSAP with Svelte 5 — the pairing this site runs on.\n\n' +
      'The pattern that matters is cleanup: an effect that creates a timeline must also ' +
      'revert it, or you leak one per remount.\n\n' +
      '```typescript\n// Svelte 5: effect with cleanup\n$effect(() => {\n  if (!el) return;\n  const ctx = gsap.context(() => {\n    gsap.from(el, { autoAlpha: 0, y: 20, duration: 0.4 });\n  }, el);\n  return () => ctx.revert(); // on unmount\n});\n```\n\n' +
      'And Lenis has to drive ScrollTrigger from a single ticker, or the two fight ' +
      'over the frame:\n\n' +
      '```javascript\nlenis.on("scroll", ScrollTrigger.update);\ngsap.ticker.add((time) => lenis.raf(time * 1000));\ngsap.ticker.lagSmoothing(0);\n```\n\n' +
      'What are you trying to build?',
    ru:
      'GSAP вместе со Svelte 5 — связка, на которой держится этот сайт.\n\n' +
      'Главный паттерн — уборка за собой: эффект, создающий таймлайн, обязан его же и ' +
      'откатить, иначе на каждый ремаунт остаётся утечка.\n\n' +
      '```typescript\n// Svelte 5: эффект с cleanup\n$effect(() => {\n  if (!el) return;\n  const ctx = gsap.context(() => {\n    gsap.from(el, { autoAlpha: 0, y: 20, duration: 0.4 });\n  }, el);\n  return () => ctx.revert(); // при размонтировании\n});\n```\n\n' +
      'И Lenis должен двигать ScrollTrigger одним тикером, иначе они дерутся за кадр:\n\n' +
      '```javascript\nlenis.on("scroll", ScrollTrigger.update);\ngsap.ticker.add((time) => lenis.raf(time * 1000));\ngsap.ticker.lagSmoothing(0);\n```\n\n' +
      'Что именно хочешь сделать?',
  },
  {
    keywords: ['rag', 'retrieval', 'vector', 'векторный', 'embedding'],
    delay: 800,
    en:
      'Retrieval is what keeps a 2B model honest, and this site runs the small version of it.\n\n' +
      'The idea in one line: look at your notes before answering. **Retrieval** finds the ' +
      'relevant note, **Augmented** puts it in the context window, **Generation** answers ' +
      'from it instead of from memory.\n\n' +
      'Here that is deliberately modest — no vector database, because the corpus is small ' +
      'enough not to need one:\n\n' +
      '```\n1. Question → EmbeddingGemma-300M → dense vector\n' +
      '2. Cosine similarity over a curated local corpus of studio facts\n' +
      '3. Top-3 above a score floor, with a relevance margin\n' +
      '4. Facts prepended to the prompt → Gemma 4 E2B → answer\n```\n\n' +
      'Everything above 3 facts made the small model paraphrase rather than answer. ' +
      'Choosing an index you do not need is the more common mistake.',
    ru:
      'Retrieval — то, что удерживает модель на 2B от вранья, и здесь работает его малая версия.\n\n' +
      'Идея одной строкой: загляни в свои записи, прежде чем отвечать. **Retrieval** находит ' +
      'нужную заметку, **Augmented** кладёт её в контекст, **Generation** отвечает по ней, ' +
      'а не по памяти.\n\n' +
      'Здесь это сделано намеренно скромно — без векторной базы, потому что корпус ' +
      'достаточно мал, чтобы она была не нужна:\n\n' +
      '```\n1. Вопрос → EmbeddingGemma-300M → плотный вектор\n' +
      '2. Косинусная близость по выверенному локальному корпусу фактов о студии\n' +
      '3. Топ-3 выше порога, с запасом по релевантности\n' +
      '4. Факты в начало промпта → Gemma 4 E2B → ответ\n```\n\n' +
      'Всё, что больше трёх фактов, заставляло маленькую модель пересказывать вместо ответа. ' +
      'Взять индекс, который тебе не нужен, — ошибка куда более частая.',
  },
];

/**
 * The catch-all, and the one that mattered most: an unmatched message is the
 * common case, so this text is what most visitors who meet the fallback read.
 */
const FALLBACK_RESPONSE: Record<MockLang, MockResponse> = {
  en: {
    delay: 600,
    response:
      'Signal received — but not from the model.\n\n' +
      'The inference server is not answering right now, so you are reading a scripted ' +
      'reply rather than a generated one. The badge above says MOCK for exactly this reason; ' +
      'the site would rather admit that than fake a live connection.\n\n' +
      'It runs on one self-hosted box and it does sleep. Ask again in a minute, or ' +
      'write to hello@vkvstudio.com if it is something a person should answer.',
  },
  ru: {
    delay: 600,
    response:
      'Signal received — но не от модели.\n\n' +
      'Сервер инференса сейчас не отвечает, так что ты читаешь заготовленный ответ, ' +
      'а не сгенерированный. Значок сверху написал MOCK именно поэтому: сайт лучше ' +
      'признается, чем изобразит живое соединение.\n\n' +
      'Он работает на одном своём сервере, и тот действительно засыпает. Спроси через ' +
      'минуту — или напиши на hello@vkvstudio.com, если вопрос к человеку.',
  },
};

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Simulates an AI response with realistic delay.
 *
 * Matches the user message against known keyword sets, then answers in the
 * language that message was written in. Falls back to a generic response that
 * says plainly it is not the model talking.
 *
 * @param userMessage - The raw text the user typed
 * @param lang - Override the detected language (the UI knows better in tests)
 * @returns MockResponse with `response` string and `delay` in ms
 */
export function mockSynapseResponse(userMessage: string, lang?: MockLang): MockResponse {
  const target = lang ?? detectMockLang(userMessage);
  const lower = userMessage.toLowerCase().trim();

  for (const entry of RESPONSE_MAP) {
    if (entry.keywords.some((kw) => lower.includes(kw))) {
      return { response: entry[target], delay: entry.delay };
    }
  }

  return FALLBACK_RESPONSE[target];
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
