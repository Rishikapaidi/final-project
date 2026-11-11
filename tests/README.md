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
committed). The page's screenshots in `docs/screenshots/` are copies of the
ones the browser tests save there (`page_<name>.png`).

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
| `e2e_logic.py` | The app logic with Streamlit's `AppTest`, a fake camera and a fake preview service: the mode chosen for each emotion (text only, forced modes, faces, face + text), Uplift journeys rising, the mood card (the "X% low" gauge, its band and the Face / Text "leans low" values checked against probabilities computed in the test, the blend note, the headline rule and the Combined bars), the choice cards ("Recommended", rebuilding the queue without the camera, "Let MOOD TUNE choose", the Telugu note; a chosen card kept when the text or language changes and reset to Auto by "Tap to scan again"), every language's queue holding only that language, Telugu and Malayalam falling back to Resonance with a note, and the camera opening only on load and "Tap to scan again". | ~7 min |
| `e2e_page.py` | The real page in headless Edge with saved previews: top bar, date and greeting, fonts, hero card, text box (180-character limit, read on Enter), mood card (gauge = the Combined bars' sad + angry + fear + disgust, band, leans, headline rule), the bars in "See all 7 emotions", scan picture, choice cards, path card and player bar. Scenario 1 also checks row click, Play pill, the bar's play/pause, the expander and the upload popover not stopping the music, no ⚙ settings button, the next song starting on its own, the journey chart, a choice card rebuilding the list without opening the camera, and the narrow (390 px) layout. Four scenarios: sad face, happy face, happy face + sad sentence, no face + sad sentence. Also watches the page for "Running ..." or a function name, and checks no image file is written. | ~5 min |
| `e2e_controls.py` | The music-app controls in headless Edge (sad face, so an Uplift queue), each used while a song plays: shuffle (greyed out during the path, never moving the path's songs), repeat one / all / off at the end of a song and of the list, seeking by click and drag, mute, the shortcuts (Space, Right, Left, M, L, and nothing while typing), the queue panel ("Added by you", remove, Escape), the "..." menu (Play next, Add to queue, Add to My playlist, Open on ..., Copy link via the clipboard), search, hearts, My playlist (play, Play all, remove, kept after a reload, Clear), Settings (autoplay, repeat, show all 7 emotions, larger text: no app rerun, same queue, music still playing) and About. Checks the list order and Uplift's path after every change, then repeats the playlist checks in a browser where localStorage throws, and the narrow (390 px) layout. Saves the queue panel, My playlist, song menu, Settings and About screenshots. | ~4 min |
| `e2e_upload.py` | Uploading a photo through "Upload a photo instead" in the face tile (the page's only uploader): a happy photo, "Lift my mood" chosen, then a sideways phone photo (EXIF rotation) that puts the choice back to Auto, a file that is not a picture, a photo with no face, then "Tap to scan again"; the camera is not opened for uploads, the music keeps playing while the upload popover opens, and no image file is written by the app. | ~3 min |
| `e2e_language.py` | "Songs in": for each language (sad face, plus a happy face in Hindi), switching from English while a song plays rebuilds the list with only that language, leaves one player with nothing still playing, and shows Spotify links, artwork, the mood card and the Telugu/Malayalam note (on the "Lift my mood" card and in the song list); then playback and the next song starting on its own. | ~5 min |
| `e2e_coldstart.py [run]` | A cold start: what the page shows while the models load (no "Running load_..." function names), and how long until the player appears. | ~2 min |
| `e2e_warmup.py wait30` / `immediate` | The background text-model warm-up: the page appears without waiting for the text model, and a sentence typed 30 s later (or straight away) is read once the model is ready. | ~2 min each |
| `pagekit.py` | Not a test: helpers the browser tests share (starting the app, finding the path card, the player bar and the top-bar buttons, counting app reruns, reading the mood card). | |
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
- The browser tests start their own app servers on ports 8561–8620, so they
  do not touch an app you already have running on 8501.
