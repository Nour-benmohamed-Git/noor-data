# word-by-word

Per-word English gloss + transliteration the Noor app downloads for the
word-by-word reading feature.

**Wire format** (`en_wbw.json`): keyed by `"surah:verse"`, value is the verse's
words in order — array index + 1 is the word position.

```json
{ "1:1": [ { "g": "In (the) name", "t": "bismi" }, { "g": "(of) Allah", "t": "l-lahi" } ] }
```

## Generating `en_wbw.json`

The data isn't a raw file upstream — it lives in the
[QUL](https://qul.tarteel.ai/) database. Generate the aligned wire file from the
**Noor app repo** (which has the verse text + the alignment audit):

1. On [qul.tarteel.ai](https://qul.tarteel.ai/) → Resources → **Word by word
   translation**, download the English resource (JSON).
2. In the Noor repo:
   ```bash
   node scripts/build-wbw.mjs --source <downloaded-file>
   ```
   The script audits alignment across all 6236 verses and only emits on success.
3. Copy the emitted `scripts/wbw_en_wbw.json` here as `word-by-word/en_wbw.json`,
   then commit + push.

Until `en_wbw.json` exists here, the in-app download fails gracefully (the
feature stays off).
