
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

`requirements.txt` pins Streamlit to **1.64.0**. The page's song list, player
bar and top-bar buttons are three `st.components.v1.html` components (see
`player.py`), and Streamlit has announced that `st.components.v1.html` will
be removed in a later version (1.64 already logs a warning suggesting
`st.iframe`). Moving to a newer Streamlit means porting those components
first; until then, keep the pin.

### Page design

A light theme (`.streamlit/config.toml`) with bundled Fraunces and Inter fonts
in `static/fonts/` (SIL Open Font License, licence files in the same folder),
served through Streamlit's static file serving, so the page looks the same
offline. The page's CSS and HTML pieces are in `ui.py`. Screenshots are in
`docs/screenshots/`.

### Player, My playlist and Settings

Everything below runs in the browser (`player.py`), so the music keeps
playing while the app reruns, and none of it is sent to the server.

- **Player bar:** previous / play / next, shuffle, repeat (off / all / this
  song), a progress bar to click or drag, mute and volume, and a queue panel
  ("Now playing", "Next up"). Shuffle only reorders the songs still to come;
  Uplift's 5-song path is never shuffled, and shuffle stays greyed out until
  the path has finished.
- **Keyboard:** Space = play/pause, Right / Left = next / previous,
  M = mute, L = save the song playing. Keys typed into a text box are
  ignored.
- **Song list:** a search box (title or artist), a heart per song, and a
  "..." menu: Play next, Add to queue, Add to / Remove from My playlist,
  Open on Spotify / Last.fm, Copy link. Songs added with Play next or Add to
  queue are tagged "Added by you" in the queue and go after the current song;
  the list's order and Uplift's path never change.
- **My playlist** (top bar, with a count badge): play, remove, Play all,
  Clear. Saved songs keep only title, artist, link, preview URL, artwork URL
  and language, in the browser's localStorage; if localStorage is blocked,
  an in-memory copy is used until the page is reloaded.
- **Settings** (the gear): autoplay next song, the starting repeat mode,
  showing all 7 emotions by default, larger text, and clearing the saved
  songs. None of these changes the queue, so they never rerun the app or
  stop the music; only "Songs in" and the Lift / Meet cards rebuild the
  queue. **About MOOD TUNE** covers what the app does, its data sources and
  licences, its limits and what it keeps.

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

