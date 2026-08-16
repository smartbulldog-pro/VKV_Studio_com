/**
 * The site's FAQ — ONE source for both the visible section and the JSON-LD.
 *
 * It used to live only inside BaseLayout's FAQPage markup, which meant the
 * answers existed for machines and not for people: the same six questions were
 * emitted on all 12 pages while appearing in the body of none, including a
 * "can I hire you" pitch on the privacy policy. That breaks Google's rule that
 * marked-up content must be visible, and — more to the point — wasted six
 * genuinely good answers by hiding them.
 *
 * Keeping one array here is what stops the visible copy and the structured data
 * drifting apart later, which is the usual way this defect comes back.
 */
import { MODEL_LIST } from '@/lib/tokenizer/engine';

export interface FaqItem {
  q: { en: string; ru: string };
  a: { en: string; ru: string };
}

/**
 * Counted, never typed in. This answer said "18 models" and was correct on the
 * day it was written; adding two rows to the tokenizer roster made it wrong
 * without touching this file, which is exactly how a number in prose goes
 * stale on a site whose whole argument is that its numbers are checked. The
 * prompt page's meta description had the identical bug and got the identical
 * fix. Build-time only — this never reaches the client bundle.
 */
const TOKENIZER_MODEL_COUNT = MODEL_LIST.length;

export const FAQ: FaqItem[] = [
  {
    q: {
      en: 'What is Synapse?',
      ru: 'Что такое Synapse?',
    },
    a: {
      en: "Synapse is the AI assistant built into this site. It runs a Gemma 4 E2B model that was fine-tuned with QLoRA on a purpose-written dataset, served from a self-hosted inference server — not in your browser, and not through anyone else's API. It answers about machine learning, cloud AI architecture and engineering, and it is still in training, so it can be wrong about facts.",
      ru: 'Synapse — это AI-ассистент, встроенный в этот сайт. Под ним модель Gemma 4 E2B, дообученная методом QLoRA на специально собранном датасете и работающая на собственном сервере инференса — не в браузере и не через чужой API. Отвечает про машинное обучение, облачную AI-архитектуру и разработку; всё ещё дообучается, поэтому может ошибаться в фактах.',
    },
  },
  {
    q: {
      en: 'Where does the model actually run?',
      ru: 'Где на самом деле работает модель?',
    },
    a: {
      en: 'On a server, through llama.cpp — a four-core ARM machine with no GPU. Expect roughly thirty to forty-five seconds for a full first answer. Most of that is the model reading the system prompt and the facts it looks up before it writes a word; the first words appear sooner, around twenty seconds in, and the rest arrives as it is written. Later questions in the same conversation reuse that reading and are noticeably faster. The browser only sends the question and renders the answer — nothing about the model runs on your device.',
      ru: 'На сервере, через llama.cpp — это четырёхъядерная ARM-машина без GPU. На полный первый ответ закладывайте секунд тридцать-сорок пять. Большая часть этого времени — модель читает системный промпт и подтянутые факты, прежде чем написать хоть слово; первые слова появляются раньше, секунде на двадцатой, а дальше ответ дописывается на глазах. Следующие вопросы в том же диалоге переиспользуют прочитанное и идут заметно быстрее. Браузер только отправляет вопрос и показывает ответ — на вашем устройстве не выполняется ничего.',
    },
  },
  {
    q: {
      en: 'What is the Edge Tokenizer Profiler?',
      ru: 'Что такое Edge Tokenizer Profiler?',
    },
    a: {
      en: `A tool that shows how a language model actually splits your text, and what that costs. It compares token counts, price and context-window usage across ${TOKENIZER_MODEL_COUNT} models from OpenAI, Google, Anthropic, Meta, DeepSeek, Alibaba, Mistral and Zhipu. Counting happens in your browser where a public tokenizer exists; for models that publish none, the count is either fetched from the provider or clearly labelled as an approximation — it is never guessed silently.`,
      ru: `Инструмент, который показывает, как языковая модель на самом деле режет ваш текст и во что это обходится. Сравнивает количество токенов, стоимость и заполнение контекстного окна для ${TOKENIZER_MODEL_COUNT} моделей от OpenAI, Google, Anthropic, Meta, DeepSeek, Alibaba, Mistral и Zhipu. Там, где есть публичный токенизатор, подсчёт идёт прямо в браузере; где его нет — число либо берётся у провайдера, либо явно помечается как приблизительное. Молча угадывать оно не будет.`,
    },
  },
  {
    q: {
      en: 'What does the Embedding Space Explorer do?',
      ru: 'Что делает Embedding Space Explorer?',
    },
    a: {
      en: "It turns text into vectors with Google EmbeddingGemma on the studio's own server, projects them into a 3D map you can rotate, and lets you search that map by meaning instead of by keyword. It also shows cross-lingual similarity (English and Russian words for the same thing land close together), vector analogies, and how different chunking and retrieval strategies change what comes back — which is the mechanism behind RAG.",
      ru: 'Превращает текст в векторы с помощью Google EmbeddingGemma на собственном сервере студии, проецирует их в трёхмерную карту, которую можно вращать, и позволяет искать по смыслу, а не по ключевым словам. Ещё показывает межъязыковое сходство (английское и русское слово об одном и том же оказываются рядом), векторные аналогии и то, как разные стратегии нарезки и поиска меняют выдачу — а это и есть механизм RAG.',
    },
  },
  {
    q: {
      en: 'What is this site built with?',
      ru: 'На чём построен этот сайт?',
    },
    a: {
      en: 'Astro 6 with Svelte 5 islands, plain CSS with no framework, and GSAP for motion. It is a static site on Cloudflare Pages: the pages are HTML files, and JavaScript only loads for the parts that are genuinely interactive. That is why it scores 100 on desktop performance while still carrying a scroll-driven video hero.',
      ru: 'Astro 6 со Svelte 5 в виде островов, обычный CSS без фреймворка и GSAP для анимации. Это статический сайт на Cloudflare Pages: страницы — это HTML-файлы, а JavaScript подгружается только там, где что-то действительно интерактивно. Поэтому он держит 100 баллов производительности на десктопе, имея при этом видео-hero, управляемое прокруткой.',
    },
  },
  {
    q: {
      en: 'Who is behind VKVstudio, and is it a real business?',
      ru: 'Кто стоит за VKVstudio и это настоящий бизнес?',
    },
    a: {
      en: 'Valerii Karpov — a full-stack engineer and AI architect with 20 years of building things that had to work. VKVstudio is a sole proprietorship registered in Armenia with the Ministry of Justice State Register since February 2026, under activity code 62.01.0 (computer programming). It is one person, not an agency — but it is a registered business, and the registration details are on the privacy page.',
      ru: 'Валерий Карпов — фулстек-разработчик и AI-архитектор, двадцать лет делающий вещи, которые обязаны работать. VKVstudio — индивидуальное предприятие, зарегистрированное в Армении в Государственном регистре Министерства юстиции с февраля 2026 года, код деятельности 62.01.0 (разработка программного обеспечения). Это один человек, а не агентство, — но это зарегистрированный бизнес, и реквизиты указаны на странице политики конфиденциальности.',
    },
  },
];
