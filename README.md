# MOOD TUNE

MOOD TUNE is a Streamlit web app that suggests music for the mood you are in.
It reads your expression in a quick webcam scan (or an uploaded photo) and,
if you like, a sentence about your day, and combines the two into one reading
over seven emotions (angry, disgust, fear, happy, sad, surprise, neutral):

- **Face:** OpenCV finds the face; HSEmotion (the default) reads its expression.
- **Text:** DistilBERT, fine-tuned on GoEmotions mapped to the same seven emotions.
- **Fusion:** a weighted log-linear rule with per-model calibration, fitted on public data (`fusion.py`).

Then it plays 30-second previews in the language you choose (English, Hindi,
Tamil, Telugu, Malayalam or Korean):

- **"Meet me here"** (Resonance) plays songs that match the mood;
- **"Lift my mood"** (Uplift) plays a five-song path from a low mood towards a calmer or brighter one (`uplift.py`, `UPLIFT.md`).

Nothing about your mood, scan or time is stored, and pictures stay in memory.

MOOD TUNE is a research and education project. The default face model's
weights allow non-commercial research and education use only (see
`CREDITS.md`).

## Installation & Run

Tested with Python 3.13 on Windows; the commands below are for PowerShell.

1. **Get the code and install the packages** (from the project root):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Download the face models** (HSEmotion and FER+, checked by SHA-256).
   They go into `models/face/onnx/` and are not in the repository:

   ```powershell
   python scripts/fetch_face_models.py
   ```

   Without them, or without `onnxruntime`, the app uses the CNN (`model.h5`) instead.

3. **The text model (optional).** The fine-tuned DistilBERT model
   (`models/text_emotion_distilbert/model/`) is not in the repository.
   Without it the app reads the face only and says so. To build it, download
   GoEmotions, then fine-tune; on a CPU training takes a few hours and resumes
   if interrupted:

   ```powershell
   python scripts/prepare_go_emotions.py
   python scripts/train_text_emotion_distilbert.py
   ```

4. **Run the app:**

   ```powershell
   streamlit run app.py
   ```

   It opens at http://localhost:8501. The scan uses the webcam of the
   computer running the app; "Upload a photo instead" works without one.

The song list (`data/songs/songs_clean.csv`) and the saved previews
(`data/songs/previews.csv`) are in the repository, so nothing else is needed
to play music. By default the app makes no song lookups; it plays only songs
with a saved preview (see `data/README.md`).

### Data and evaluation scripts (optional)

| Script | What it does |
|---|---|
| `scripts/prepare_fer2013.py` | Downloads FER2013 and checks it against the official split sizes; needed by the face and fusion evaluations |
| `scripts/prepare_go_emotions.py` | Downloads GoEmotions and maps its 28 labels to the 7 emotions |
| `scripts/train_text_emotion.py`, `scripts/train_text_emotion_distilbert.py` | The text-model baseline and DistilBERT |
| `scripts/prepare_songs.py` | Rebuilds `songs_clean.csv` from `muse_v3.csv` and the Spotify tracks file (`data/songs/raw/`, not committed) |
| `scripts/fetch_previews.py` | Fills the preview cache slowly from the iTunes Search API |
| `scripts/evaluate_face.py` | Compares the face models (`models/face/report.txt`) |
| `scripts/evaluate_fusion.py` | Fits and reports the fusion settings (`models/fusion/`) |
| `scripts/evaluate_fusion_conflict.py`, `scripts/evaluate_fusion_missing.py` | Report fusion when signals disagree or one is missing |
| `scripts/evaluate_uplift.py` | Checks the Uplift playlists |

The data sources and the evaluation numbers are in `data/README.md` and
`docs/EVALUATION_FACTS.md`; the tests are described in `tests/README.md`.

## How MOOD TUNE works

### Streamlit version

`requirements.txt` pins Streamlit to **1.64.0**. The player bar, the song
lists and the shell's parts (the top bar's search and avatar, the sidebar's
playlists, the Library) are `st.components.v1.html` components (see
`player.py` and `shell.py`), and Streamlit has announced that
`st.components.v1.html` will be removed in a later version (1.64 already logs
a warning suggesting `st.iframe`). They also rely on being on the same origin
as the page (they share data through it and the sidebar's playlists press the
Library button on it). Moving to a newer Streamlit means porting those
components first; until then, keep the pin.

### Face model

The face is found with OpenCV's Haar cascade (`face.py`) and read by one of
three face-expression models (`face_models.py`), each given the same face
box and building its own crop from the original frame. A scan takes about
3 seconds: the camera opens (DirectShow on Windows, which opens in about 1 s
against about 6 s for the default backend), then half a second to get ready
and a 1-second capture; the app averages the model over the faces in 10
frames spread evenly across the capture.

| `FACE_MODEL` | Model | Input | Trained on | Licence / limits |
|---|---|---|---|---|
| `hsemotion` (default) | HSEmotion `enet_b0_8_best_afew` | 224x224 colour | VGGFace2, then AffectNet (about 450,000 natural colour photos) | code Apache-2.0; AffectNet allows non-commercial research and education only, so these weights carry that limit |
| `ferplus` | FER+ `emotion-ferplus-8` (ONNX model zoo) | 64x64 grayscale | FER2013 training images, FER+ crowd labels (Microsoft) | MIT |
| `cnn` | the CNN inherited from the project this one started from (`model.h5`, see Credits); each crop is averaged with its mirror image | 48x48 grayscale | not documented (its 7 classes and 48x48 grayscale input match FER2013) | no licence stated by its source; see `CREDITS.md` |

The ONNX files are not in the repository: `python scripts/fetch_face_models.py`
downloads HSEmotion and FER+ into `models/face/onnx/` and checks their SHA-256.
If the chosen model's file or `onnxruntime` is missing, the app uses the CNN.
To compare them on your own webcam, start the app with, for example,
`$env:FACE_MODEL="cnn"; streamlit run app.py`; the face tile says which model
read the scan. No model is trained or tuned on anyone's face.

`python scripts/evaluate_face.py` compares them on FER2013 PrivateTest
through the app's own path (detector, box, crop) and writes
`models/face/report.txt`, scoring against the original FER2013 labels and,
when `fer2013new.csv` is there (`fetch_face_models.py --labels`), the FER+
labels (10 crowd votes per face). On the 2,441 faces the detector finds:

| | accuracy (original) | macro-F1 (original) | accuracy (FER+) | macro-F1 (FER+) | angry/disgust/fear read as happy (FER+) | happy read as neutral or sad (FER+) | load / scan of 80 faces |
|---|---|---|---|---|---|---|---|
| HSEmotion | 55.3% | 0.494 | 54.8% | 0.466 | 1.0% | 2.2% | 0.5 s / 1.2 s |
| FER+ | 60.2% | 0.492 | 82.2% | 0.659 | 3.0% | 6.2% | 0.5 s / 0.5 s |
| CNN (+ mirror) | 64.8% | 0.607 | 63.5% | 0.543 | 5.0% | 4.9% | ~5 s (TensorFlow) / 0.5 s |

HSEmotion is the default: chosen after an informal live webcam comparison by the developer, and because it was trained on natural colour photos. On the public benchmark no model was clearly better, and webcam accuracy was not measured.
Its weak spots on this benchmark: on these grayscale 48-pixel FER2013 faces
it is the least accurate overall, it sometimes reads a neutral face as angry
or sad, and it often reads surprise as fear. These numbers come from FER2013
pictures, not from webcam faces.

The face + text fusion settings were fitted for each face model on public
data only (FER2013 PublicTest x GoEmotions validation pairs;
`python scripts/evaluate_fusion.py --face hsemotion`). HSEmotion's fit keeps
the rule (log-linear) and the face weight (0.55) and lowers the face
temperature from 1.931 to 1.448; it scored a little better on the validation
pairs (fused macro-F1 0.799 against 0.789 with the CNN's settings), so the app
uses it with HSEmotion (`models/fusion/hsemotion/fusion_config.json`); on the
test pairs the two are 0.800 and 0.798. The CNN keeps its own settings
(`models/fusion/fusion_config.json`). Fused macro-F1 on the test pairs: CNN
0.816 (face alone 0.615, text alone 0.689), HSEmotion 0.800 (face alone
0.531, text alone 0.689).

### Page design

A light theme (`.streamlit/config.toml`) with bundled Fraunces and Inter fonts
in `static/fonts/` (SIL Open Font License, licence files in the same folder),
served through Streamlit's static file serving, so the page looks the same
offline. The page's CSS and HTML pieces are in `ui.py`. Screenshots are in
`docs/screenshots/`.

### Layout: top bar, sidebar and pages

A music-app layout with MOOD TUNE's own name, logo and colours:

- **Top bar** (pinned): a menu button, the logo, a search box that filters
  the songs on the page by title or artist as you type, "Songs in", and an
  avatar, which shows the first letter of the name you give it (optional,
  kept in this browser) or a person icon. Its menu: the name and "Local
  profile" (a pencil changes the name), Settings, Liked songs and History
  (open the Library), Keyboard shortcuts, Privacy, About MOOD TUNE, Send
  feedback (the project's GitHub Issues page, in a new tab; the address is
  `FEEDBACK_URL` in `shell.py`) and Reset profile (after a confirm, clears
  everything MOOD TUNE keeps in this browser, and the settings cookie).
- **Sidebar:** Home, Explore and Library, then "New playlist" and your
  playlists (Liked songs, My playlist, the ones you made, Recently played).
  The menu button folds it to icons (remembered in this browser); on a
  narrow screen it slides in over the page instead.
- **Home**, from the top:
  - a row of seven **mood chips** (one row on a wide screen; it scrolls
    sideways on a narrow one). A chip replaces the player's queue with up to
    30 songs of that mood in the "Songs in" language (Explore's pick for it)
    and plays the first (the queue panel names the list); picking a song from
    Home's list, or a new reading, brings the mood queue back. A chip looks
    selected only while its songs are in the player (nothing is selected on
    a fresh visit; a new reading or another "Songs in" clears it). The
    language is chosen only with "Songs in";
  - the **Detect my mood** card: the face scan and your words side by side,
    and the reading under them (the "X% low" gauge, the headline, how low
    face and text lean, "See all 7 emotions");
  - **Quick picks**: 12 songs in a grid (3 columns on a wide window) with
    Play all, picked at random from the reading's top emotion in "Songs in"
    (or a mix of that language before there is a reading). The pick is kept
    for the visit, so reruns don't reshuffle it, and it avoids Explore's first
    12 for the same mood and language;
  - the two cards ("Lift my mood", "Meet me here") and the songs or Uplift's
    path, as before.
- **Explore:** browse by mood (the seven song buckets) and by language,
  each tile with how many songs have a preview. Telugu and Malayalam say
  "Meet me here only" (too few songs for "Lift my mood"). A pick shows up to
  30 of those songs at random.
- **Library:** Liked songs, My playlist and your other playlists, and
  Recently played, as cards; each opens as a list with Play all, and
  Rename / Delete (your own playlists) or Clear.

Moving between the pages never stops the music: the player bar keeps its
place on the page, and the words typed on Home stay in the box. If Home has
no reading for a while (no face and no words, say after a scan that found no
face), the player keeps the last mood queue until a new reading replaces it. Songs
played from Explore or the Library start straight away and go in after the
song playing, like "Play next"; the queue from your mood carries on after
them.

### Player, Liked songs, playlists and Settings

Everything below runs in the browser (`player.py`, `shell.py`), so the music
keeps playing while the app reruns, and none of it is sent to the server.

- **Player bar:** previous / play / next, shuffle, repeat (off / all / this
  song), a progress bar to click or drag, mute and volume, and a queue panel
  ("Now playing", "Next up"). Shuffle only reorders the songs still to come;
  Uplift's 5-song path is never shuffled, and shuffle stays greyed out until
  the path has finished.
- **Now playing:** the song's art or title in the player bar (or its arrow)
  opens a view over the page, with the bar still at the bottom: large art,
  title, artist and language, and three tabs. **Up next** is the play order
  (Uplift's 5-song path in order and marked, never shuffled; a click plays
  that song). **Mood** shows the reading's headline, its "X% low" band and
  the mood card's sentence about the songs (the same words as Home).
  **Related** lists up to 12 songs in the queue's language and the reading's
  top emotion that are not in the queue; a click plays one. Esc or the close
  button goes back to the page. It runs in the browser: the music doesn't
  stop and the app doesn't rerun. On a narrow screen the tabs sit under the
  art.
- **Keyboard:** Space = play/pause, Right / Left = next / previous,
  M = mute, L = like the song playing. Keys typed into a text box are
  ignored.
- **Song lists:** a heart per song (Liked songs) and a "..." menu: Play
  next, Add to queue, Add to / Remove from Liked songs, Add to playlist (any
  playlist, or a new one), Open on Spotify / Last.fm, Copy link. Songs added
  with Play next or Add to queue are tagged "Added by you" in the queue and
  go after the current song; the list's order and Uplift's path never
  change.
- **What is kept, and where:** Liked songs, playlists and Recently played
  (the last 50 songs started, newest first) keep only each song's title,
  artist, link, preview URL, artwork URL and language: never a mood, a scan
  or a time. With your name, the settings, the volume and the folded
  sidebar, they live in the browser's localStorage (the settings cookie
  above is the one exception); if localStorage is
  blocked, an in-memory copy is used until the page is reloaded.
- **Settings** (the avatar's menu): Theme (Light, the default, or Dark),
  larger text, showing all 7 emotions by default, Default language and
  Default mode (Auto, "Meet me here" or "Lift my mood"), autoplay next song,
  the starting repeat mode, and clearing Liked songs or Recently played.
  None of these changes the queue, so they never rerun the app or stop the
  music; only "Songs in" and the Lift / Meet cards rebuild the queue. The
  theme switches straight away. The default language and mode apply from the
  next time MOOD TUNE opens ("Songs in" starts with the language, and each
  new reading starts with the mode). **About MOOD TUNE** covers what the app
  does, its data sources and licences, its limits and what it keeps;
  **Privacy** says what is kept where.
- **The settings cookie:** so the app can open with them, the theme, default
  language and default mode are also kept in a cookie (`moodtune_prefs`),
  the only thing the server reads from the browser. It is removed when all
  three are back at their defaults, and by Reset profile.

## Libraries

- Streamlit (pinned to 1.64.0, see above)
- OpenCV (`opencv-python`)
- ONNX Runtime: the HSEmotion and FER+ face models
- PyTorch and Hugging Face Transformers: the DistilBERT text model
- TensorFlow and Keras: only for the CNN (`FACE_MODEL=cnn`, or the fallback)
- NumPy, pandas, scikit-learn, matplotlib, joblib
- requests: the optional iTunes preview lookups

## Demo video

Demo video: coming soon.

## Authors

- Rishika Paidi

## Credits

Started from Hiral Sathwara's Emotion-based music recommendation system (2024), https://github.com/Hiralsathwara/Emotion-based-music-recommendation-system. From it this project still uses the CNN weights model.h5 and a few generic lines of app.py.

Third-party files, data and models, with their licences, are listed in
`CREDITS.md`.
