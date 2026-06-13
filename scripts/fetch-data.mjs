#!/usr/bin/env node
/**
 * Populate noor-data from TarteelAI/quran-assets (public).
 *
 *   - translations/  : the translations the Noor app offers, transformed from
 *                      quran-assets' array-of-arrays into the app's wire format
 *                      ({ "surah:verse": "text" }) and named by the app's ids.
 *   - tafsir/        : English tafsir JSON (as-is; staging for the tafsir feature).
 *   - indopak/       : per-word IndoPak Arabic text (as-is; staging for IndoPak).
 *
 * Re-run any time to refresh. Word-by-word (en_wbw.json) is generated separately
 * via the Noor repo's scripts/build-wbw.mjs against a QUL source — see
 * word-by-word/README.md.
 *
 * Usage:  node scripts/fetch-data.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = 'https://raw.githubusercontent.com/TarteelAI/quran-assets/main';
const CONCURRENCY = 12;
const TOTAL_VERSES = 6236;

// Noor app translation id  ->  quran-assets tanzil filename stem.
const TRANSLATIONS = {
  en_sahih: 'en-sahih',
  en_yusufali: 'en-yusufali',
  en_pickthall: 'en-pickthall',
  fr_hamidullah: 'fr-hamidullah',
  id_indonesian: 'id-indonesian',
  bn_bengali: 'bn-bengali',
  ru_kuliev: 'ru-kuliev',
  tr_diyanet: 'tr-diyanet',
  ur_maududi: 'ur-maududi',
};

const TAFSIRS = ['en-tafsir-ibn-kathir', 'en-tafsir-maarif-ul-quran', 'en-tafsir-mokhtasar'];

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
const ensureDir = (d) => mkdir(d, { recursive: true });

async function buildTranslations() {
  await ensureDir(resolve(ROOT, 'translations'));
  for (const [appId, stem] of Object.entries(TRANSLATIONS)) {
    try {
      const data = await getJson(`${SRC}/translations/tanzil/${stem}.json`);
      if (!Array.isArray(data) || data.length !== 114) {
        console.warn(`! ${appId}: unexpected shape — skipped`);
        continue;
      }
      const out = {};
      let total = 0;
      data.forEach((verses, si) => {
        verses.forEach((text, vi) => {
          out[`${si + 1}:${vi + 1}`] = text;
          total++;
        });
      });
      if (total !== TOTAL_VERSES) {
        console.warn(`! ${appId}: ${total} verses (expected ${TOTAL_VERSES}) — skipped`);
        continue;
      }
      await writeFile(resolve(ROOT, 'translations', `${appId}.json`), JSON.stringify(out));
      console.log(`✓ translations/${appId}.json (${total})`);
    } catch (e) {
      console.warn(`! ${appId}: ${e.message}`);
    }
  }
}

async function buildTafsir() {
  await ensureDir(resolve(ROOT, 'tafsir'));
  for (const name of TAFSIRS) {
    try {
      const data = await getJson(`${SRC}/tafsir/${name}.json`);
      await writeFile(resolve(ROOT, 'tafsir', `${name}.json`), JSON.stringify(data));
      console.log(`✓ tafsir/${name}.json`);
    } catch (e) {
      console.warn(`! ${name}: ${e.message}`);
    }
  }
}

async function buildIndopak() {
  await ensureDir(resolve(ROOT, 'indopak'));
  const tasks = [];
  for (let s = 1; s <= 114; s++) {
    tasks.push(async () => {
      const data = await getJson(`${SRC}/indopak-word-by-word/${s}.json`);
      await writeFile(resolve(ROOT, 'indopak', `${s}.json`), JSON.stringify(data));
    });
  }
  let i = 0;
  let ok = 0;
  let fail = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (i < tasks.length) {
        const idx = i++;
        try {
          await tasks[idx]();
          ok++;
        } catch (e) {
          fail++;
          console.warn(`! indopak ${idx + 1}: ${e.message}`);
        }
      }
    })
  );
  console.log(`✓ indopak/ ${ok}/114${fail ? ` (${fail} failed)` : ''}`);
}

const t0 = Date.now();
console.log('Populating noor-data from quran-assets…');
await buildTranslations();
await buildTafsir();
await buildIndopak();
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
