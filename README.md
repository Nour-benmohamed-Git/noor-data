# noor-data

Public, data-only companion to the Noor Quran app. Holds the JSON datasets the
app downloads on demand, served free via the [jsDelivr](https://www.jsdelivr.com/)
CDN. The app source lives in a separate **private** repo.

> ⚠️ This repo **must stay public** — jsDelivr only serves public GitHub repos.

## CDN base

```
https://cdn.jsdelivr.net/gh/Nour-benmohamed-Git/noor-data@main/<path>
```

## Contents

| Folder | What | Format | Used by |
|--------|------|--------|---------|
| `translations/` | 113 verse translations (43 languages) + `index.json` catalog | `{ "surah:verse": "text" }`; manifest `[{id,name,language,author,isDefault}]` | Translations (live) |
| `word-by-word/` | Per-word gloss + transliteration | `{ "surah:verse": [{ "g", "t" }] }` | Word-by-word (live) — see folder README |
| `tafsir/` | English verse commentary | source JSON | Tafsir (planned) |
| `indopak/` | IndoPak-script word text (`1.json`…`114.json`) | `{ "<verse>": { total_words, words[] } }` | IndoPak script (planned) |
| `tajweed/` | Uthmani text with tajweed markup | `{ "surah:verse": "…<tajweed class=…>…" }` | Tajweed-colored mushaf (planned) |

## Refreshing

`translations/`, `tafsir/`, and `indopak/` are pulled and transformed from
[TarteelAI/quran-assets](https://github.com/TarteelAI/quran-assets):

```bash
node scripts/fetch-translations.mjs  # all 113 translations + index.json manifest
node scripts/fetch-data.mjs          # tafsir, indopak
node scripts/fetch-tajweed.mjs       # tajweed/uthmani.json (from quran.com API)
```

`word-by-word/en_wbw.json` is generated separately — see
[`word-by-word/README.md`](word-by-word/README.md).

## Licensing

Each dataset retains its original source and license — see [SOURCES.md](SOURCES.md).
**Verify per-resource terms before commercial use**, especially tafsir and translations.
