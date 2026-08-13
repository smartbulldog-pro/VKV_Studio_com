// @ts-check
import { execFileSync } from 'node:child_process';
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import sitemap from '@astrojs/sitemap';

/**
 * `lastmod` per URL, taken from the last commit that touched the page's source.
 *
 * Google only uses lastmod when a site is "consistently accurate" about it, so
 * stamping every page with the build time — the tempting one-liner — is worse
 * than omitting it: each deploy would claim all 14 pages changed, and the
 * signal gets discounted. Git already knows which page actually changed and
 * when, so that is what ships.
 *
 * The map is keyed by the path *after* the locale segment, because both
 * locales are generated from one `[lang]` source file and therefore share a
 * modification date.
 */
const PAGE_SOURCES = {
  '/': 'src/pages/[lang]/index.astro',
  '/lab/tokenizer/': 'src/pages/[lang]/lab/tokenizer/index.astro',
  '/lab/prompt/': 'src/pages/[lang]/lab/prompt/index.astro',
  '/lab/embeddings/': 'src/pages/[lang]/lab/embeddings/index.astro',
  '/log/': 'src/pages/[lang]/log/index.astro',
  '/privacy/': 'src/pages/[lang]/privacy/index.astro',
};

/** ISO commit date of the newest commit touching `file`, or null. */
function lastCommitISO(file) {
  try {
    const out = execFileSync('git', ['log', '-1', '--format=%cI', '--', file], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    return out || null;
  } catch {
    // No git (a tarball build, a CI checkout without history) — omit lastmod
    // rather than invent one. An absent date is honest; a wrong one is not.
    return null;
  }
}

const LASTMOD = Object.fromEntries(
  Object.entries(PAGE_SOURCES)
    .map(([route, file]) => [route, lastCommitISO(file)])
    .filter(([, date]) => date !== null)
);

// https://astro.build/config
export default defineConfig({
  site: 'https://vkvstudio.com',
  integrations: [
    svelte(),
    sitemap({
      i18n: {
        defaultLocale: 'en',
        locales: { en: 'en', ru: 'ru' },
      },
      // Exclude the bare "/" redirect shell (src/pages/index.astro — no
      // content of its own, just forwards to /en/ or /ru/ by detected
      // locale). @astrojs/sitemap's i18n grouping parses a URL with no
      // locale prefix as { locale: defaultLocale, path: '/' } — identical to
      // "/en/"'s { locale: 'en', path: '/' } — so without this filter "/"
      // and "/en/" land in the same alternates group and both claim
      // hreflang="en", which is invalid (one hreflang value must resolve to
      // exactly one URL). Excluding "/" here (before the i18n grouping runs)
      // leaves "/en/" and "/ru/" as the only real, indexable locale pages.
      filter: (page) => new URL(page).pathname !== '/',
      serialize(item) {
        const route = new URL(item.url).pathname.replace(/^\/(en|ru)/, '') || '/';
        const lastmod = LASTMOD[route];
        return lastmod ? { ...item, lastmod } : item;
      },
    }),
  ],
  output: 'static',
  // NOT enabling Astro's hash-based CSP (`security.csp`). It looks like the
  // obvious way to drop 'unsafe-inline' from script-src, and it was tried and
  // measured on a preview deploy 2026-08-09 — it does not fit this site:
  //
  //  1. `strictDynamic: true` disables host-based allowlisting INCLUDING
  //     'self'. Astro hydrates islands through dynamic import(), which
  //     strict-dynamic does not extend trust to, so all 10 islands were
  //     blocked and the page shipped dead.
  //  2. With strictDynamic off, scripts work — but Astro also emits hashes in
  //     style-src, and per the CSP spec ANY hash there makes the browser
  //     ignore 'unsafe-inline'. That blocks the runtime inline style
  //     attributes Svelte emits for the hero parallax (`style:transform`),
  //     whose values are computed per frame and so cannot be hashed at build
  //     time. Omitting styleDirective is worse still: Astro then emits
  //     style-src with hashes and no 'unsafe-inline' at all.
  //
  // So the choice is between a working hero and a stricter script-src, and
  // 'unsafe-inline' stays for now. Revisit if Astro exposes style-src-attr,
  // or if the parallax stops using inline style attributes.
  compressHTML: true,
  server: {
    port: 4173,
    host: 'localhost',
  },
  vite: {
    build: {
      assetsInlineLimit: 0,
    },
  },
});
