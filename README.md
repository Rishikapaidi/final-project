
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

`requirements.txt` pins Streamlit to **1.64.0**. The page's song list and
player bar are two `st.components.v1.html` components (see `player.py`), and
Streamlit has announced that `st.components.v1.html` will be removed in a
later version (1.64 already logs a warning suggesting `st.iframe`). Moving to
a newer Streamlit means porting those two components first; until then, keep
the pin.

### Page design

A light theme (`.streamlit/config.toml`) with bundled Fraunces and Inter fonts
in `static/fonts/` (SIL Open Font License, licence files in the same folder),
served through Streamlit's static file serving, so the page looks the same
offline. The page's CSS and HTML pieces are in `ui.py`. Screenshots are in
`docs/screenshots/`.

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

