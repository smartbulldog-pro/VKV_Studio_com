<script lang="ts">
  import { t } from '../../../i18n/utils';
  import { MODELS } from '../../../lib/prompt/builder';
  import { SYNAPSE_API_BASE } from '@/lib/api-config';

  let { lang = 'en', onGenerated } = $props<{
    lang?: 'en' | 'ru';
    onGenerated: (blocks: { role: string; content: string }[]) => void;
  }>();

  let userInput = $state('');
  let selectedModel = $state(MODELS[0].id);
  let selectedStyle = $state('structured');
  let isGenerating = $state(false);

  /**
   * Builds the generation request text in the current UI locale.
   *
   * This isn't just cosmetic: the backend (`inference/llm.py`) has no
   * explicit language parameter — it infers reply language from the
   * Cyrillic-character ratio of the *request text itself*. A Russian-only
   * scaffold (as this used to be, hardcoded) made every request look
   * Russian to that heuristic regardless of the site locale, so `/en/`
   * always got a Russian reply. Keeping the scaffold's majority script
   * aligned with `lang`, plus an explicit "respond only in X" directive,
   * fixes that at the source. The `[SYSTEM]`/`[USER]`/`[ASSISTANT]` tags
   * are deliberately kept as literal, untranslated ASCII in both locales —
   * the parsing below matches on those exact tags regardless of language.
   */
  function buildGenerationPrompt(uiLang: 'en' | 'ru', task: string, model: string, style: string): string {
    if (uiLang === 'ru') {
      return `Ты генератор промптов. Сгенерируй промпт для следующей задачи.
Отвечай ТОЛЬКО на русском языке — не используй английский или другой язык.
Задача: ${task}
Целевая модель: ${model}
Стиль: ${style}
Ответ дай в формате (метки [SYSTEM]/[USER]/[ASSISTANT] оставь как есть, латиницей):
[SYSTEM]
текст системного промпта
[USER]
текст пользовательского сообщения
Не добавляй пояснений. Только промпт.`;
    }
    return `You are a prompt generator. Generate a prompt for the following task.
Respond ONLY in English — do not use Russian or any other language.
Task: ${task}
Target model: ${model}
Style: ${style}
Format your answer exactly like this (keep the [SYSTEM]/[USER]/[ASSISTANT] tags in English as-is):
[SYSTEM]
system prompt text
[USER]
user message text
Do not add explanations. Output only the prompt.`;
  }

  async function handleGenerate() {
    if (!userInput.trim() || isGenerating) return;

    isGenerating = true;
    try {
      const promptText = buildGenerationPrompt(lang, userInput, selectedModel, selectedStyle);

      const res = await fetch(`${SYNAPSE_API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // `lang` is passed alongside the directive baked into `promptText`
        // above so a future backend revision can key off it directly —
        // the current `/api/chat` (ChatRequest, extra="ignore") simply
        // drops unknown fields, so this is inert until then, not a break.
        body: JSON.stringify({ message: promptText, history: [], lang })
      });

      if (!res.ok) throw new Error('Generation failed');

      const data = await res.json();
      const text = data.response;

      // Parse blocks
      const blocks: { role: string; content: string }[] = [];
      
      const systemMatch = text.match(/\[SYSTEM\]\n?([\s\S]*?)(?=\[USER\]|\[ASSISTANT\]|$)/i);
      const userMatch = text.match(/\[USER\]\n?([\s\S]*?)(?=\[SYSTEM\]|\[ASSISTANT\]|$)/i);
      const assistantMatch = text.match(/\[ASSISTANT\]\n?([\s\S]*?)(?=\[SYSTEM\]|\[USER\]|$)/i);

      if (systemMatch && systemMatch[1].trim()) {
        blocks.push({ role: 'system', content: systemMatch[1].trim() });
      }
      if (userMatch && userMatch[1].trim()) {
        blocks.push({ role: 'user', content: userMatch[1].trim() });
      }
      if (assistantMatch && assistantMatch[1].trim()) {
        blocks.push({ role: 'assistant', content: assistantMatch[1].trim() });
      }

      if (blocks.length > 0) {
        onGenerated(blocks);
      } else {
        // Fallback if no tags found
        onGenerated([{ role: 'user', content: text.trim() }]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      isGenerating = false;
    }
  }
</script>

<div class="generator-card glass-panel">
  <div class="generator-heading">
    <h2 class="generator-title">{t(lang, 'prompt.generateTitle')}</h2>
    <p class="generator-hint">{t(lang, 'prompt.generateTabHint')}</p>
    <p class="generator-disclaimer" role="note">{t(lang, 'prompt.generateDisclaimer')}</p>
  </div>

  <div class="generator-field">
    <textarea
      class="generator-textarea"
      bind:value={userInput}
      placeholder={t(lang, 'prompt.generatePlaceholder')}
    ></textarea>
  </div>

  <div class="generator-controls">
    <div class="generator-select-group">
      <label for="model-select" class="generator-label">{t(lang, 'prompt.targetModel')}</label>
      <select id="model-select" bind:value={selectedModel} class="generator-select">
        {#each MODELS as model}
          <option value={model.id}>{model.name}</option>
        {/each}
      </select>
    </div>

    <div class="generator-select-group">
      <label for="style-select" class="generator-label">{t(lang, 'prompt.promptStyle')}</label>
      <select id="style-select" bind:value={selectedStyle} class="generator-select">
        <option value="structured">{t(lang, 'prompt.styleStructured')}</option>
        <option value="markdown">{t(lang, 'prompt.styleMarkdown')}</option>
        <option value="cot">{t(lang, 'prompt.styleCot')}</option>
        <option value="minimal">{t(lang, 'prompt.styleMinimal')}</option>
      </select>
    </div>
  </div>

  <button
    class="generator-btn"
    class:generating={isGenerating}
    disabled={!userInput.trim() || isGenerating}
    onclick={handleGenerate}
  >
    {#if isGenerating}
      <span class="pulse-dot"></span> {t(lang, 'prompt.generating')}
    {:else}
      {t(lang, 'prompt.generateButton')}
    {/if}
  </button>
</div>

<style>
  .generator-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    gap: var(--space-6);
    backdrop-filter: blur(20px);
  }

  .generator-heading {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .generator-title {
    font-size: var(--text-h3);
    color: var(--text-primary);
    margin: 0;
    font-weight: var(--weight-medium);
  }

  .generator-hint {
    font-size: var(--text-sm);
    color: var(--text-muted);
    margin: 0;
    line-height: var(--leading-normal);
  }

  /* Honesty disclaimer — drafted by a model still in training (warning-amber convention) */
  .generator-disclaimer {
    margin: var(--space-2) 0 0;
    padding: var(--space-2) var(--space-3);
    border: 1px solid hsla(40, 90%, 55%, 0.3);
    border-radius: var(--radius-md);
    background: hsla(40, 90%, 55%, 0.08);
    color: hsl(40, 90%, 68%);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    line-height: var(--leading-normal);
  }

  .generator-field {
    display: flex;
    flex-direction: column;
  }

  .generator-textarea {
    width: 100%;
    min-height: 120px;
    background: var(--bg-void);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: var(--text-base);
    resize: vertical;
    outline: none;
    transition: var(--transition-colors);
  }

  .generator-textarea:focus {
    border-color: var(--accent-cyan-400);
  }

  .generator-controls {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--space-4);
  }

  @media (min-width: 640px) {
    .generator-controls {
      grid-template-columns: 1fr 1fr;
    }
  }

  .generator-select-group {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .generator-label {
    font-size: var(--text-sm);
    color: var(--text-muted);
  }

  .generator-select {
    width: 100%;
    background: var(--bg-void);
    border: 1px solid var(--border-default);
    color: var(--text-primary);
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    font-family: var(--font-sans);
    font-size: var(--text-sm);
    outline: none;
    cursor: pointer;
  }

  .generator-select:focus {
    border-color: var(--accent-cyan-400);
  }

  @media (max-width: 767px) {
    /* Prevent iOS Safari auto-zoom on focus (requires >=16px font-size) */
    .generator-select {
      font-size: 16px;
    }
  }

  .generator-btn {
    align-self: flex-start;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    background: var(--accent-green-400);
    color: var(--bg-obsidian);
    border: none;
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-8);
    font-family: var(--font-mono);
    font-weight: var(--weight-semibold);
    font-size: var(--text-sm);
    cursor: pointer;
    transition: var(--transition-all);
  }

  .generator-btn:hover:not(:disabled) {
    box-shadow: 0 0 16px hsla(155, 70%, 50%, 0.4);
    transform: translateY(-1px);
  }

  .generator-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .generator-btn.generating {
    animation: pulse 1.5s infinite;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: var(--bg-obsidian);
    border-radius: 50%;
    animation: blink 1s infinite;
  }

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
  }

  @keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
  }
</style>
