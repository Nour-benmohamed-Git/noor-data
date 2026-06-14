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
- **Source:** [TarteelAI/quran-assets](https://github.com/TarteelAI/quran-assets) (`tafsir/`),
  normalized to `{ "surah:verse": text }` by the Noor repo's `scripts/build-tafsir.mjs`.
- **Shipped (in the app catalog):**
  - `en_mokhtasar` — **Al-Mukhtasar** (Markaz Tafsir, Riyadh). Explicitly free to redistribute.
  - `ar_qurtubi` — **Al-Qurtubi** (Abu Abdullah al-Qurtubi, d. 671 AH / 1273 CE).
    Classical → **public domain** (author died > 700 years ago). One empty ayah (28:34).
  - `ar_baghawi` — **Al-Baghawi / Ma'alim al-Tanzil** (al-Husayn al-Baghawi, d. 516 AH /
    1122 CE). Classical → **public domain**. Complete (0 empty ayat).
- **Generated but NOT shipped (gitignored, copyright risk):** `en_ibnkathir`
  (Darussalam abridged translation ©) and `en_maarif` (Ma'arif-ul-Quran ©). The raw
  English inputs (`en-tafsir-*.json`) remain in the repo only as build inputs.
- **Intentionally excluded:** quran-assets' `ar-tafsir-tanweer` is **Ibn Ashur's
  at-Tahrir wa't-Tanwir** (Muhammad al-Tahir ibn Ashur, d. 1973) — modern and
  **copyrighted**, *not* the classical (public-domain) Tanwir al-Miqbas its slug
  suggests. Verified by content, so it is not fetched or built.
- **License note:** QUL/quran.com disclaim uniform licensing — each resource carries
  its own copyright and you must verify per-resource ([QUL FAQ](https://qul.tarteel.ai/faq)).
  Only the three shipped above are cleared for this commercial app (one explicit free
  license + two public-domain classical works). The other quran-assets tafsirs (modern
  English/Urdu/Bengali/Russian translations) carry live publisher copyright and are not shipped.

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
