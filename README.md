
# Emotion-based music recommendation system

This web-based app written in Python will first scan your current emotion with the help of OpenCV & then crop the image of your face from the entire frame once the cropped image is ready it will give this image to a trained MACHINE LEARNING model to predict the emotion of the cropped image. This will happen 30-40 times in 2-3 seconds, now once we have a list of emotions (containing duplicate elements) with us it will first sort the list based on frequency & remove the duplicates. After performing all the above steps we will have a list containing the user's emotions in sorted order, Now we have to iterate over the list & recommend songs based on emotions in the list.


## Installation & Run

Create a new project in Pycharm and add the above files. After that open the terminal and run the following command. This will install all the modules needed to run this app. 

```bash
  pip install -r requirements.txt
```

To run the app, type the following command in the terminal. 
```bash
  streamlit run app.py
```

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
  - a row of **chips**: six languages, then the seven moods. A language chip
    picks the language (it starts as "Songs in"); a mood chip replaces the
    player's queue with up to 30 songs of that mood in that language
    (Explore's pick for it) and plays the first (the queue panel names the
    list); picking a song from Home's list, or a new reading, brings the mood
    queue back;
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

- Streamlit
- Opencv
- Numpy
- Pandas
- Tensorflow
- Keras





## Demo video

 [Emotion-based music recommendation system](

https://github.com/Hiralsathwara/Emotion-based-music-recommendation-system/assets/127468119/5b2626f6-68fe-499c-9c3e-56b37f7c5d26

)
 

## Authors

- [HIRAL SATHWARA](https://github.com/Hiralsathwara)



## Support

For support,hiral sathwara (https://www.linkedin.com/in/hiral-sathwara-873561269/)

