/**
 * FE Iter 1 — XSS regression battery for sanitizeResponse (standalone sanitizer).
 * ==============================================================================
 * sanitizeResponse is the secondary text-sanitizer used by the non-streaming
 * chat()/sendVoice() helpers + mock fallback. It must be robust ON ITS OWN — safe
 * even if its output is ever placed in a raw-HTML sink. In particular it must
 * survive the mutation-XSS (mXSS) bypass that a single-pass tag strip allowed.
 */
import { describe, it, expect } from 'vitest';
import { sanitizeResponse } from '@/lib/synapse-client';

describe('sanitizeResponse — XSS battery', () => {
  it('mXSS: nested <<script>script> must NOT reconstruct a live tag', () => {
    const out = sanitizeResponse('<<script>script>alert(1)<</script>/script>');
    expect(out).not.toContain('<script>');
    expect(out).not.toContain('</script>');
    expect(out).not.toMatch(/<script/i);
  });

  it('mXSS: split handler <img src=x o<script>nerror> collapses', () => {
    const out = sanitizeResponse('<img src=x o<script>nerror=alert(1)>');
    expect(out).not.toMatch(/<img/i);
    expect(out).not.toMatch(/<script/i);
    expect(out).not.toMatch(/\bonerror\s*=/i);
  });

  const strippedPayloads = [
    '<img src=x onerror=alert(1)>',
    '<svg/onload=alert(1)>',
    '<ScRiPt>alert(1)</ScRiPt>',
    '<iframe srcdoc="x">',
    '<!-- comment -->',
  ];
  for (const p of strippedPayloads) {
    it(`strips tags/handlers from: ${p.slice(0, 30)}`, () => {
      const out = sanitizeResponse(p);
      expect(out).not.toMatch(/<[a-z]/i);
      expect(out).not.toMatch(/\son\w+\s*=/i);
    });
  }

  it('removes javascript:/data:/vbscript: schemes', () => {
    expect(sanitizeResponse('go to javascript:alert(1)')).not.toMatch(/javascript:/i);
    expect(sanitizeResponse('data:text/html,x')).not.toMatch(/data:/i);
    expect(sanitizeResponse('vbscript:msgbox')).not.toMatch(/vbscript:/i);
  });

  it('removes numeric entity obfuscation', () => {
    // &#106; = 'j' — must not survive to rebuild javascript:
    const out = sanitizeResponse('&#106;avascript:alert(1)');
    expect(out).not.toMatch(/javascript:/i);
  });

  it('handles empty / non-string input', () => {
    expect(sanitizeResponse('')).toBe('');
    // @ts-expect-error deliberately wrong type
    expect(sanitizeResponse(null)).toBe('');
  });

  it('leaves normal prose untouched', () => {
    const text = 'Lighthouse 100 is our baseline, not the goal.';
    expect(sanitizeResponse(text)).toBe(text);
  });

  // ── Comment-splice mutation: the gap the battery above did NOT cover ────────
  // The cases above nest TAGS (`<<script>script>`), which the original
  // fixed-point tag loop already collapsed. The real hole was a dangerous token
  // reconstructed by a LATER stage: removing an HTML comment splices its
  // neighbours together, and only the tag stage looped, so the splice was never
  // re-scanned. Twelve of these were fed to the staged version and ten came out
  // live. Each is pinned so the staged structure cannot return.
  const splicePayloads = [
    '<<!-- -->script>alert(1)<<!-- -->/script>',
    '<<![CDATA[x]]>script>alert(1)<<![CDATA[x]]>/script>',
    '<<!-- -->img src=x o<!--y-->nerror=alert(1)>',
    '<<!-- -->svg o<!--y-->nload=alert(1)>',
    '<<!-- -->a href="java<!--y-->script:alert(1)">x</a>',
    '<<!-- -->iframe src=x>',
    '<<!--<!-- -->-->script>alert(1)</script>',
    '<scr<!-- -->ipt>alert(1)</scr<!-- -->ipt>',
    'javajavascript:script:alert(1)', // \b once made this match nothing
    'o<!--a-->n<!--b-->error=alert(1)',
  ];
  for (const p of splicePayloads) {
    it(`comment-splice leaves nothing live: ${p.slice(0, 32)}`, () => {
      const out = sanitizeResponse(p);
      expect(out, `tag survived: ${out}`).not.toMatch(/<\/?[a-zA-Z]/);
      expect(out, `handler survived: ${out}`).not.toMatch(/\bon[a-zA-Z]{2,20}\s*=/i);
      expect(out, `scheme survived: ${out}`).not.toMatch(/(?:javascript|vbscript)\s*:/i);
    });
  }

  it('is idempotent: a second pass finds nothing the first missed', () => {
    for (const p of splicePayloads) {
      const once = sanitizeResponse(p);
      expect(sanitizeResponse(once)).toBe(once);
    }
  });
});
