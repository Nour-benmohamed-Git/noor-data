#!/usr/bin/env node
/**
 * Fetch ALL tanzil translations from TarteelAI/quran-assets, transform each to
 * the app wire format ({ "surah:verse": "text" }), and emit translations/index.json
 * — the catalog manifest the app loads to populate its translation picker.
 *
 *   translations/<id>.json   (id = slug with '-' -> '_', e.g. en_sahih)
 *   translations/index.json  [{ id, name, language, author, isDefault }]
 *
 * Names: a curated OVERRIDES map for well-known translations, else a prettified
 * slug. Language from a fixed code map. Only translations validating to 6236
 * verses are written/listed. Re-runs skip files that already exist (just the
 * manifest is rebuilt), so refreshing names is instant.
 *
 * Usage:  node scripts/fetch-translations.mjs
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const GH_API = 'https://api.github.com/repos/TarteelAI/quran-assets/contents/translations/tanzil';
const RAW = 'https://raw.githubusercontent.com/TarteelAI/quran-assets/main/translations/tanzil';
const CONCURRENCY = 10;
const TOTAL_VERSES = 6236;
const DEFAULT_ID = 'en_sahih';

const LANG = {
  am: 'Amharic', ar: 'Arabic', az: 'Azerbaijani', ber: 'Berber', bg: 'Bulgarian',
  bn: 'Bengali', bs: 'Bosnian', cs: 'Czech', de: 'German', dv: 'Divehi',
  en: 'English', es: 'Spanish', fa: 'Persian', fr: 'French', ha: 'Hausa',
  hi: 'Hindi', id: 'Indonesian', it: 'Italian', ja: 'Japanese', ko: 'Korean',
  ku: 'Kurdish', ml: 'Malayalam', ms: 'Malay', nl: 'Dutch', no: 'Norwegian',
  pl: 'Polish', pt: 'Portuguese', ro: 'Romanian', ru: 'Russian', sd: 'Sindhi',
  so: 'Somali', sq: 'Albanian', sv: 'Swedish', sw: 'Swahili', ta: 'Tamil',
  tg: 'Tajik', th: 'Thai', tr: 'Turkish', tt: 'Tatar', ug: 'Uyghur', ur: 'Urdu',
  uz: 'Uzbek', zh: 'Chinese',
};

// Polished names for well-known translations; everything else uses the
// prettified slug (which is already a clean surname for most).
const OVERRIDES = {
  en_sahih: 'Saheeh International',
  en_yusufali: 'Yusuf Ali',
  en_ahmedali: 'Ahmed Ali',
  en_ahmedraza: 'Ahmed Raza Khan',
  en_hilali: 'Hilali & Khan',
  en_qaribullah: 'Qaribullah & Darwish',
  en_wahiduddin: 'Wahiduddin Khan',
  ur_ahmedali: 'Ahmed Ali',
  ur_kanzuliman: 'Kanzul Iman',
  de_aburida: 'Abu Rida',
  de_bubenheim: 'Bubenheim & Elyas',
  ru_abuadel: 'Abu Adel',
  ru_kuliev: 'Kuliev',
  ru_kuliev_alsaadi: 'Kuliev & as-Saadi',
  ru_muntahab: 'Al-Muntakhab',
  am_sadiq: 'Sadiq & Sani',
  ml_karakunnu: 'Karakunnu & Elayavoor',
  uz_sodik: 'Muhammad Sodiq',
  zh_majian: 'Ma Jian',
  zh_jian: 'Ma Jian',
  bn_bengali: 'Muhiuddin Khan',
  id_indonesian: 'Indonesian Ministry of Religious Affairs',
};

async function getJson(url, headers = {}) {
  const r = await fetch(url, { headers: { accept: 'application/json', ...headers } });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  return r.json();
}

const prettify = (s) =>
  s.split(/[-_]/).filter(Boolean).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

async function listSlugs() {
  const entries = await getJson(GH_API, { 'user-agent': 'noor-data-build' });
  return entries
    .filter((e) => e.type === 'file' && e.name.endsWith('.json'))
    .map((e) => e.name.replace(/\.json$/, ''));
}

function transform(data) {
  if (!Array.isArray(data) || data.length !== 114) return null;
  const out = {};
  let total = 0;
  data.forEach((verses, si) => {
    if (!Array.isArray(verses)) return;
    verses.forEach((text, vi) => {
      out[`${si + 1}:${vi + 1}`] = text;
      total++;
    });
  });
  return total === TOTAL_VERSES ? out : null;
}

const slugs = await listSlugs();
console.log(`Found ${slugs.length} tanzil translations`);

await mkdir(resolve(ROOT, 'translations'), { recursive: true });

const manifest = [];
const skipped = [];
let i = 0;
let done = 0;
await Promise.all(
  Array.from({ length: CONCURRENCY }, async () => {
    while (i < slugs.length) {
      const slug = slugs[i++];
      const id = slug.replace(/-/g, '_');
      const file = resolve(ROOT, 'translations', `${id}.json`);
      try {
        if (!existsSync(file)) {
          const data = await getJson(`${RAW}/${slug}.json`);
          const wire = transform(data);
          if (!wire) {
            skipped.push(slug);
            done++;
            continue;
          }
          await writeFile(file, JSON.stringify(wire));
        }
        const code = slug.split('-')[0];
        const translator = slug.slice(code.length + 1);
        manifest.push({
          id,
          name: OVERRIDES[id] || prettify(translator),
          language: LANG[code] || prettify(code),
          author: OVERRIDES[id] || prettify(translator),
          isDefault: id === DEFAULT_ID,
        });
      } catch (e) {
        skipped.push(`${slug} (${e.message})`);
      }
      done++;
      if (done % 25 === 0 || done === slugs.length) process.stdout.write(`  ${done}/${slugs.length}\n`);
    }
  })
);

manifest.sort(
  (a, b) =>
    a.language.localeCompare(b.language) ||
    Number(b.isDefault) - Number(a.isDefault) ||
    a.name.localeCompare(b.name)
);
await writeFile(resolve(ROOT, 'translations/index.json'), JSON.stringify(manifest));

const langs = new Set(manifest.map((m) => m.language));
console.log(`\n✓ ${manifest.length} translations across ${langs.size} languages + index.json`);
if (skipped.length) console.warn(`! skipped ${skipped.length}: ${skipped.join(', ')}`);
