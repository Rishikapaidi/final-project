# Tests

End-to-end checks of the app. They are plain scripts, not pytest tests (the
names start with `e2e_` so pytest does not collect them): each one prints
lines starting with `RESULT` to read through. Run them from the project root
with the project's virtual environment active, for example:

```
python tests/e2e_logic.py
```

None of them uses the real webcam or calls the iTunes API. Logs, screenshots
and the pictures the upload test generates go to `tests/output/` (not
committed).

## What they need

- The app's own files, plus three local, uncommitted data files used to make
  fake camera frames and test sentences: `data/face_emotion/raw/privateTest-0000.parquet`,
  `data/fusion/face_test.npz` and `data/fusion/text_test.npz` (rebuild them
  as described in `data/README.md`), and the trained text model under
  `models/text_emotion_distilbert/model/`.
- For the browser tests: Playwright (`pip install playwright`) and Microsoft
  Edge, which they drive headless (`channel="msedge"`; no browser download is
  needed when Edge is installed).

## The scripts

| Script | What it checks | Time |
|---|---|---|
| `e2e_logic.py` | The app logic with Streamlit's `AppTest`, a fake camera and a fake preview service: the mode chosen for each emotion (text only, forced modes, faces, face + text), Uplift journeys rising, every language's queue holding only that language, Telugu and Malayalam falling back to Resonance with a note, and the camera opening only on load and Re-scan (not on setting or language changes). | ~6 min |
| `e2e_page.py` | The real page in headless Edge with saved previews: header, mood card and headline rule, emotion bars and their numbers, scan picture, track list, playing and pausing, the settings popover not stopping the music, and the next song starting on its own. Four scenarios: sad face, happy face, happy face + sad sentence, no face + sad sentence. | ~4 min |
| `e2e_upload.py` | "Upload a photo instead": a happy photo, a sideways phone photo (EXIF rotation), a file that is not a picture, a photo with no face, then Re-scan; the camera is not opened for uploads, and no image file is written by the app. | ~3 min |
| `e2e_language.py` | "Songs in": for each language (sad face, plus a happy face in Hindi), switching from English while a song plays rebuilds the list with only that language, leaves one player with nothing still playing, and shows Spotify links, artwork, the mood card and the Telugu/Malayalam note; then playback and the next song starting on its own. | ~5 min |
| `e2e_coldstart.py [run]` | A cold start: what the page shows while the models load (no "Running load_..." function names), and how long until the player appears. | ~2 min |
| `e2e_warmup.py wait30` / `immediate` | The background text-model warm-up: the page appears without waiting for the text model, and a sentence typed 30 s later (or straight away) is read once the model is ready. | ~2 min each |
| `fake_camera_server.py` | Not a test: starts the real app with a fake camera, used by the browser tests. Settings are environment variables: `TEST_FRAME` (`sad`, `happy`, `mixed`, `blank`), `TEST_PREVIEWS` (`real` = the saved previews, `fake` = generated tones), `TEST_PORT`, and optionally `TEST_CAM_LOG`, `TEST_MODULE_LOG`, `TEST_APP`, `TEST_WATCHER`. | |

Options:

- `e2e_page.py` and `e2e_language.py` take scenario names to run only some,
  e.g. `python tests/e2e_language.py telugu korean`.
- `e2e_logic.py` builds queues with live lookups through its fake preview
  service, so run it with `LIVE_LOOKUPS=1` (PowerShell:
  `$env:LIVE_LOOKUPS=1; python tests/e2e_logic.py`); without it the queues
  only use songs whose fake previews are already saved and come out empty.
  `TEST_SEED=<n>` makes the app's random picks repeatable, and
  `TEST_DUMP=<file.json>` saves the English queues, so two versions of the
  code can be compared song by song (`TEST_ENGLISH_ONLY=1` stops after the
  English scenarios).
- The browser tests start their own app servers on ports 8561–8615, so they
  do not touch an app you already have running on 8501.
