# Credits

MOOD TUNE is by Rishika Paidi. This file lists the third-party files, data
and models it uses, where they came from, and their licences. "In the
repository" means the file is committed. Files the scripts download are not
committed; they are used on their own terms.

## Inherited from the starting project

This repository started from Hiral Sathwara's
[Emotion-based music recommendation system](https://github.com/Hiralsathwara/Emotion-based-music-recommendation-system)
(2024). Its first three commits are that project's. **That repository states
no licence.**

- **`model.h5`**, the CNN face-expression weights (in the repository, unchanged since the original upload).
  - **Training:** not documented. Its 7 classes and 48×48 grayscale input match FER2013.
  - **Use:** the face model with `FACE_MODEL=cnn`, and the fallback when the ONNX models or `onnxruntime` are missing (`face_models.py`). The evaluation scripts also use it.
- **`app.py`:** a few generic lines remain from the original (imports, `cap.release()`, `else:`, `break`). The rest of the app was written for MOOD TUNE.
- The original upload also brought in `muse_v3.csv` and `haarcascade_frontalface_default.xml`, both third-party; see below.

## Songs

- **Muse: the Musical Sentiment dataset** (`muse_v3.csv`, in the repository, unchanged).
  - **Source:** C. Akiki and M. Burghardt (2021), "MuSe: The Musical Sentiment Dataset", *Journal of Open Humanities Data*, DOI [10.5334/johd.33](https://doi.org/10.5334/johd.33).
  - **Licence:** CC BY 4.0.
  - **Use:** `scripts/prepare_songs.py` builds the English songs of `data/songs/songs_clean.csv` from it. Those songs carry its titles, artists, Last.fm links, mood tags and valence/arousal values.
- **"Spotify Tracks Dataset (Updated Every Week)"** by gauthamvijayaraj on Kaggle.
  - **Licence:** Apache 2.0.
  - **Use:** the Hindi, Tamil, Telugu, Malayalam and Korean songs in `songs_clean.csv` come from it. The raw file is not committed.
- **iTunes Search API** (Apple).
  - **Use:** the preview and artwork links in `data/songs/previews.csv`. The audio and images are played from Apple's servers, not stored here; they are Apple's.

## Face detection

- **OpenCV Haar cascades**: `haarcascade_frontalface_default.xml` and `haarcascade_frontalface_alt2.xml`.
  - **Source:** created by Rainer Lienhart.
  - **Licence:** the "Intel License Agreement For Open Source Computer Vision Library", Copyright (C) 2000, Intel Corporation. It allows redistribution if the copyright notice and conditions are kept; the full text is in the file's header.
  - **Use:** the app loads the copies bundled with the `opencv-python` package (`cv2.data.haarcascades`, `face.py`).
  - **The root copy:** `haarcascade_frontalface_default.xml` in the project root came with the original upload and is **not used** by the app or the scripts.

## Face-expression models (downloaded by `scripts/fetch_face_models.py`, not committed)

- **HSEmotion**, `enet_b0_8_best_afew.onnx`, by Andrey Savchenko ([github.com/HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)).
  - **Licence:** code Apache-2.0. The weights were pretrained on VGGFace2 and fine-tuned on AffectNet, whose licence allows **non-commercial research and education only**, so these weights carry that limit.
  - **Use:** the app's default face model.
- **FER+**, `emotion-ferplus-8.onnx`, from the ONNX model zoo ([github.com/onnx/models](https://github.com/onnx/models)).
  - **Licence:** MIT.
  - **Source:** trained by Microsoft on the FER2013 training images with the FER+ crowd-sourced labels.
  - **Use:** available with `FACE_MODEL=ferplus`.
- **FER+ labels**, `fer2013new.csv`, Microsoft ([github.com/microsoft/FERPlus](https://github.com/microsoft/FERPlus)).
  - **Licence:** MIT.
  - **Use:** `scripts/evaluate_face.py` only (`--labels`).

## Text emotion

- **GoEmotions** (`google-research-datasets/go_emotions` on Hugging Face), downloaded by `scripts/prepare_go_emotions.py`, not committed.
  - **Licence:** its dataset card says the GitHub repository holding it has an Apache License 2.0. The comments themselves are Reddit content.
  - **Committed derived files:** `models/text_emotion*/test_predictions.csv` include test-set comment text.
- **DistilBERT**, `distilbert-base-uncased` (Hugging Face).
  - **Licence:** Apache 2.0.
  - **Use:** the base of the text model. The fine-tuned weights (`models/text_emotion_distilbert/model/`) are not committed; see the README.

## Evaluation data

- **FER2013** (the Kaggle / ICML 2013 challenge data), from the Hugging Face copy `Aaryan333/fer2013_train_publicTest_privateTest`, downloaded by `scripts/prepare_fer2013.py`, not committed.
  - **Licence:** none stated.
  - **Use:** the face and fusion evaluations only.

## Fonts

- **Fraunces and Inter** (`static/fonts/`, in the repository).
  - **Licence:** SIL Open Font License 1.1, with the licence texts beside them (`OFL-Fraunces.txt`, `OFL-Inter.txt`).
