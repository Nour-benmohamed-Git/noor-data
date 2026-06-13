#!/usr/bin/env node
/**
 * Build tajweed/uthmani.json — Uthmani text with inline tajweed markup
 * (<tajweed class="...">…</tajweed>), for the tajweed-colored mushaf feature.
 *
 * Source: quran.com API v4 (/quran/verses/uthmani_tajweed). Reciter-independent
 * (it annotates the text, not audio), so it's safe to host as-is.
 *
 * Usage:  node scripts/fetch-tajweed.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BASE = 'https://api.quran.com/api/v4';
const CONCURRENCY = 6;

async function getJson(url, tries = 3) {
  for (let t = 1; ; t++) {
    try {
      const r = await fetch(url, { headers: { accept: 'application/json' } });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      if (t >= tries) throw e;
      await new Promise((res) => setTimeout(res, 400 * t));
    }
  }
}

const out = {};
let i = 0;
let done = 0;
const chapters = Array.from({ length: 114 }, (_, k) => k + 1);
await Promise.all(
  Array.from({ length: CONCURRENCY }, async () => {
    while (i < chapters.length) {
      const ch = chapters[i++];
      const j = await getJson(`${BASE}/quran/verses/uthmani_tajweed?chapter_number=${ch}`);
      for (const v of j.verses) out[v.verse_key] = v.text_uthmani_tajweed;
      done++;
      if (done % 20 === 0 || done === 114) process.stdout.write(`  ${done}/114\n`);
    }
  })
);

await mkdir(resolve(ROOT, 'tajweed'), { recursive: true });
const path = resolve(ROOT, 'tajweed/uthmani.json');
await writeFile(path, JSON.stringify(out));
console.log(`wrote ${path} — ${Object.keys(out).length} verses`);
