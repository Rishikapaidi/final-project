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
| `e2e_logic.py` | The app logic with Streamlit's `AppTest`, a fake camera and a fake preview service: the mode chosen for each emotion (text only, forced modes, faces, face + text), Uplift journeys rising, the mood card (the "X% low" gauge, its band and the Face / Text "leans low" values checked against probabilities computed in the test, the blend note, the headline rule and the Combined bars), the choice cards ("Recommended", rebuilding the queue without the camera, "Let MOOD TUNE choose", the Telugu note; a chosen card kept when the text or language changes and reset to Auto by "Tap to scan again"), every language's queue holding only that language, Telugu and Malayalam falling back to Resonance with a note, the first load (no camera, the idle face tile, "Tap to scan"; words alone give a text-only reading with working choice cards, the same as after a scan that found no face), and the camera opening only on "Tap to scan" / "Tap to scan again". Faces are read after pressing the scan button. | ~7 min |
| `e2e_page.py` | The real page in headless Edge with saved previews: top bar (logo, menu button, search, "Songs in", avatar), sidebar, date and greeting, fonts, hero card, text box (180-character limit, read on Enter), mood card (gauge = the Combined bars' sad + angry + fear + disgust, band, leans, headline rule), the bars in "See all 7 emotions", scan picture, choice cards, path card and player bar. Scenario 1 also checks row click, Play pill, the bar's play/pause, the expander and the upload popover not stopping the music, no ⚙ settings button, the next song starting on its own, the journey chart, a choice card rebuilding the list without opening the camera, and the narrow (390 px) layout. Four scenarios: sad face, happy face, happy face + sad sentence, no face + sad sentence. Also watches the page for "Running ..." or a function name, and checks no image file is written. | ~5 min |
| `e2e_controls.py` | The music-app controls in headless Edge (sad face, so an Uplift queue), each used while a song plays: shuffle (greyed out during the path, never moving the path's songs), repeat one / all / off at the end of a song and of the list, seeking by click and drag, mute, the shortcuts (Space, Right, Left, M, L, and nothing while typing), the queue panel ("Added by you", remove, Escape), the "..." menu (Play next, Add to queue, Add to playlist, Open on ..., Copy link via the clipboard), the top bar's search, hearts, Liked songs and My playlist (the sidebar's counts, what is stored, the Library: play, Play all, remove; kept after a reload, cleared from Settings), the avatar's menu, Settings (autoplay, repeat, show all 7 emotions, larger text: no app rerun, same queue, music still playing) and About. Checks the list order and Uplift's path after every change, then repeats the liked-songs checks in a browser where localStorage throws, and the narrow (390 px) layout. Saves the queue panel, Library, song menu, Settings and About screenshots. | ~4 min |
| `e2e_shell.py` | The shell in headless Edge (sad face): Home → Explore → Library → Home with a song playing (same queue, same song, time going on, the typed words kept), Explore (seven mood tiles, six language tiles with "Meet me here only" on Telugu and Malayalam, each pick listing only that mood and language, playing a song from it, the search box), the sidebar's New playlist, Rename and Delete in the Library, Recently played (newest first, six fields only), "Songs in" changed on the Library rebuilding the queue, the hamburger (folded sidebar kept after a reload; slide-in sidebar on a 390 px screen) and the avatar's letter. Then, with no face (a second server): words, a song playing, Library and back, words cleared and "Tap to scan again" with still no face, and the player keeps the last mood queue (still playing) until new words replace it. Saves the Explore, folded sidebar, narrow sidebar and New playlist screenshots. | ~3 min |
| `e2e_home.py` | Home in headless Edge: with a sad face, the chips row (the seven moods only, in one row, none selected or focused on a fresh visit), the "Detect my mood" card holding the scan, the words and the reading, Quick picks (12 songs of the top emotion in "Songs in", three columns, Play all, a tile plays its song, the search box, kept through reruns and page changes, differing from Explore's first 12; one column on a 390 px screen), and a mood chip replacing the queue with Explore's pick for that mood in the "Songs in" language (in its order, playing the first, the bar showing its title and art, selected) once (not again after a rerun or a trip to the Library), a song from Home's list bringing the mood queue back, and another "Songs in" clearing the selection. With no face: a chip plays before there is a reading, Quick picks shows a mix, the reading's placeholder is in the card, and the first reading clears the chip's selection. Saves the top of Home and Quick picks screenshots. | ~3 min |
| `e2e_nowplaying.py` | The now-playing view in headless Edge (sad face, so an Uplift queue): it opens from the song's title, its art and the expand arrow, and closes with its close button and Esc (focus goes back), with the music playing on (same song, time going on) and no app rerun; the left side (art, title, artist, language); Up next (the play order, the path's 5 steps marked, a click plays that song, shuffle still locked on the path); Mood (the same headline, "X% low", band and sentence as Home); Related (12 songs of the queue's language and the reading's top emotion, none in the queue, a click plays one); and on a 390 px screen the tabs under the art. Saves the Up next, Mood, Related and narrow screenshots. | ~2 min |
| `e2e_profile.py` | The avatar's menu and Settings in headless Edge (sad face): the menu's items, the name with "Local profile" and the pencil, Send feedback (the project's GitHub Issues page, new tab, no `mailto` anywhere), Theme switched while a song plays (no app rerun, same queue, the page and all six components follow), Default language and Default mode (the cookie; after a reload "Songs in" and the chosen card start with them; a rescan goes back to the default mode), Liked songs and History (open the Library), Keyboard shortcuts, Privacy, About, and Reset profile (Cancel keeps everything; Reset clears the browser's MOOD TUNE data and the cookie). Saves the menu, Settings and Privacy screenshots. | ~3 min |
| `e2e_upload.py` | Uploading a photo through "Upload a photo instead" in the face tile (the page's only uploader): a happy photo, "Lift my mood" chosen, then a sideways phone photo (EXIF rotation) that puts the choice back to Auto, a file that is not a picture, a photo with no face, then "Tap to scan again"; the camera is off on first load (until "Tap to scan") and is not opened for uploads, the music keeps playing while the upload popover opens, and no image file is written by the app. | ~3 min |
| `e2e_language.py` | "Songs in": for each language (sad face, plus a happy face in Hindi), switching from English while a song plays rebuilds the list with only that language, leaves one player with nothing still playing, and shows Spotify links, artwork, the mood card and the Telugu/Malayalam note (on the "Lift my mood" card and in the song list); then playback and the next song starting on its own. | ~5 min |
| `e2e_face_models.py` | The face-model switch (`face_models.py`): each model (the CNN, FER+, HSEmotion) loads and returns seven probabilities in the order fusion expects; the CNN's crop is exactly `face.crop`'s; FER+ reads the box in grayscale and HSEmotion in colour from the original frame; "contempt" counts as disgust; a missing ONNX file or an unknown `FACE_MODEL` falls back to the CNN; the default is HSEmotion. Then the app run with `FACE_MODEL=ferplus` (sad face) reads the scan with FER+, says so on the face tile, and its mood card checks out. Needs the ONNX files (`python scripts/fetch_face_models.py`). | ~1 min |
| `e2e_face_detection.py` | Face detection without the browser (`face.find_faces`): two backlit faces (a darkened FER2013 face on a bright gradient wall) that the plain Haar search misses and the contrast-boosted fallback finds exactly once, one by the default cascade and one by alt2; no face in the blank grey and noise frames; the browser tests' fake-camera and upload frames still give exactly one face with the same box as before; and the crop is cut from the original frame, so the same box gives the same prediction. | ~1 min |
| `e2e_coldstart.py [run]` | A cold start: what the page shows while the models load (no "Running load_..." function names), and how long until the player appears. | ~2 min |
| `e2e_warmup.py wait30` / `immediate` | The background text-model warm-up: the page appears without waiting for the text model, and a sentence typed 30 s later (or straight away) is read once the model is ready. | ~2 min each |
| `pagekit.py` | Not a test: helpers the browser tests share (starting the app, finding the path card, the player bar and the other components, counting app reruns, reading the mood card). | |
| `fake_camera_server.py` | Not a test: starts the real app with a fake camera, used by the browser tests. Settings are environment variables: `TEST_FRAME` (`sad`, `happy`, `mixed`, `blank`), `TEST_PREVIEWS` (`real` = the saved previews, `fake` = generated tones), `TEST_PORT`, and optionally `TEST_CAM_LOG`, `TEST_MODULE_LOG`, `TEST_APP`, `TEST_WATCHER`. | |

The app opens the webcam only when "Tap to scan" is pressed, so every browser test that
needs a face presses it first (`pagekit.tap_scan`).

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
- `TEST_THEME=dark` runs the browser tests with the dark theme saved, as a
  listener's choice would be (PowerShell: `$env:TEST_THEME="dark"`); their
  logs and screenshots go to `tests/output/dark/`. The page, home, shell,
  profile and now-playing tests log which theme the page and its components show.
- The browser tests start their own app servers on ports 8561–8628, so they
  do not touch an app you already have running on 8501.
