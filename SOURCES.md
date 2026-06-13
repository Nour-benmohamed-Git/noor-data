# Sources & licenses

All datasets here are redistributed from upstream open sources. **Noor is a
commercial app — verify each resource's terms before relying on it in
production.** This file records provenance and the attribution shown in-app.

## Translations — `translations/`
- **Source:** [Tanzil.net](https://tanzil.net) translations, via
  [TarteelAI/quran-assets](https://github.com/TarteelAI/quran-assets) (`translations/tanzil/`),
  transformed to `{ "surah:verse": "text" }`.
- **License:** Each translation is individually licensed by Tanzil. Many permit
  free use **with attribution**; some restrict commercial use. Check
  [tanzil.net/docs/Quran_Translations](https://tanzil.net/trans/) per translation.
- **Files:** `en_sahih`, `en_yusufali`, `en_pickthall`, `fr_hamidullah`,
  `id_indonesian`, `bn_bengali`, `ru_kuliev`, `tr_diyanet`, `ur_maududi`.

## Word-by-word — `word-by-word/`
- **Source:** [Quranic Universal Library](https://qul.tarteel.ai/) (English
  word-by-word + transliteration), derived from
  [Quranic Arabic Corpus](https://corpus.quran.com).
- **License:** GNU — usable in any app **provided the source is credited and a
  link to corpus.quran.com is shown**. (Already credited in the app's About screen.)
- **Generated** via the Noor repo's `scripts/build-wbw.mjs` (alignment-audited).

## Tafsir — `tafsir/`
- **Source:** [TarteelAI/quran-assets](https://github.com/TarteelAI/quran-assets) (`tafsir/`).
- **Files:** Ibn Kathir (abridged, EN), Maarif-ul-Quran (EN), Al-Mukhtasar/Mokhtasar (EN).
- **License:** ⚠️ **Varies and not all are commercial-friendly.** Al-Mukhtasar is
  generally free to distribute; Ibn Kathir (abridged) and Maarif-ul-Quran English
  translations may carry publisher copyright. **Confirm before shipping commercially.**

## Tajweed — `tajweed/`
- **Source:** [quran.com API v4](https://api.quran.com/api/v4/quran/verses/uthmani_tajweed)
  (`text_uthmani_tajweed`) — Uthmani text with inline `<tajweed class="…">` markup.
- **Content:** rule annotations over the text (reciter-independent).
- **License:** quran.com / KFGQPC Uthmani text — confirm redistribution terms before
  commercial use.

## IndoPak word text — `indopak/`
- **Source:** [TarteelAI/quran-assets](https://github.com/TarteelAI/quran-assets) (`indopak-word-by-word/`).
- **Content:** IndoPak-script Arabic, segmented per word (no translation).
- **License:** ⚠️ Quranic script data — confirm redistribution terms with QUL/upstream.
