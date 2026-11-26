# Uplift paths by start emotion and language

Paths: the app's own Uplift path (`app.build_queue`, saved previews only) for 200 sampled fused readings per start
emotion (HSEmotion + DistilBERT on FER2013 PrivateTest x GoEmotions test pairs), in each language. Random: 5 playable
songs drawn at random from the same language (`player.playable_songs`); Resonance: the first 5 songs of a Resonance
playlist of the start emotion's bucket (`player.resonance_queue`). Same readings and the same seeded generator for all three.

Intended direction: sad paths should get brighter (valence rises at each step); angry, fear and disgust paths should
get calmer (arousal falls at each step). Steps are strict: a step with no change counts as not in the direction.
Chance for a random path: 50% of steps, 0.8% for all 4 steps (1 in 120 orders), 50% for song 5 vs song 1.
Difference columns are percentage points (valence points for the valence row). The 95% interval is a paired
bootstrap over readings (2000 resamples). Valence and arousal are dataset values (Muse tags; Spotify
features mapped by percentile), not checked by people; this measures playlist structure, not any effect on a listener.

Flags: a cell is too small to read below 30 complete paths (none is: every cell has 200); a small song
pool (fewer than 30 playable songs in the start bucket in that language) means the paths reuse few songs.

| start emotion | language | measure | unit | Uplift | Random | Resonance | Uplift - Random | 95% interval of Uplift - Random | Uplift - Resonance | n paths | n paths (random, Resonance) | n steps | playable songs in start bucket | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sad | English | steps in the intended direction | % | 99.8 | 51.4 | 47.8 | +48.4 | +45.9 to +50.7 | +52.0 | 200 | 200, 200 | 800 | 149 |  |
| sad | English | all 4 steps in the intended direction | % | 99.0 | 0.5 | 0.5 | +98.5 | +96.5 to +100.0 | +98.5 | 200 | 200, 200 |  | 149 |  |
| sad | English | song 5 vs song 1 in the intended direction | % | 100.0 | 56.0 | 48.5 | +44.0 | +37.0 to +51.0 | +51.5 | 200 | 200, 200 |  | 149 |  |
| sad | Hindi | steps in the intended direction | % | 99.8 | 49.1 | 47.1 | +50.6 | +48.2 to +53.1 | +52.6 | 200 | 200, 200 | 800 | 17 | small song pool: 17 playable sad songs |
| sad | Hindi | all 4 steps in the intended direction | % | 99.0 | 2.0 | 0.5 | +97.0 | +94.5 to +99.0 | +98.5 | 200 | 200, 200 |  | 17 | small song pool: 17 playable sad songs |
| sad | Hindi | song 5 vs song 1 in the intended direction | % | 100.0 | 48.5 | 53.5 | +51.5 | +45.5 to +58.5 | +46.5 | 200 | 200, 200 |  | 17 | small song pool: 17 playable sad songs |
| sad | Tamil | steps in the intended direction | % | 98.9 | 49.2 | 42.4 | +49.6 | +47.2 to +52.2 | +56.5 | 200 | 200, 200 | 800 | 23 | small song pool: 23 playable sad songs |
| sad | Tamil | all 4 steps in the intended direction | % | 95.5 | 0.0 | 0.0 | +95.5 | +92.0 to +98.5 | +95.5 | 200 | 200, 200 |  | 23 | small song pool: 23 playable sad songs |
| sad | Tamil | song 5 vs song 1 in the intended direction | % | 100.0 | 54.5 | 43.0 | +45.5 | +39.0 to +52.0 | +57.0 | 200 | 200, 200 |  | 23 | small song pool: 23 playable sad songs |
| sad | Korean | steps in the intended direction | % | 99.4 | 50.6 | 42.4 | +48.8 | +46.4 to +51.2 | +57.0 | 200 | 200, 200 | 800 | 24 | small song pool: 24 playable sad songs |
| sad | Korean | all 4 steps in the intended direction | % | 97.5 | 2.0 | 0.0 | +95.5 | +92.5 to +98.0 | +97.5 | 200 | 200, 200 |  | 24 | small song pool: 24 playable sad songs |
| sad | Korean | song 5 vs song 1 in the intended direction | % | 100.0 | 49.5 | 41.0 | +50.5 | +43.5 to +57.5 | +59.0 | 200 | 200, 200 |  | 24 | small song pool: 24 playable sad songs |
| angry | English | steps in the intended direction | % | 97.9 | 47.4 | 39.4 | +50.5 | +47.8 to +53.2 | +58.5 | 200 | 200, 200 | 800 | 176 |  |
| angry | English | all 4 steps in the intended direction | % | 91.5 | 1.0 | 0.0 | +90.5 | +86.0 to +94.5 | +91.5 | 200 | 200, 200 |  | 176 |  |
| angry | English | song 5 vs song 1 in the intended direction | % | 100.0 | 45.5 | 35.0 | +54.5 | +48.0 to +61.5 | +65.0 | 200 | 200, 200 |  | 176 |  |
| angry | English | valence change, song 5 minus song 1 (calming paths) | valence points | 1.5 | 0.0 | 0.0 | +1.4 | +1.1 to +1.8 | +1.4 | 200 | 200, 200 |  | 176 |  |
| angry | Hindi | steps in the intended direction | % | 96.0 | 49.9 | 44.2 | +46.1 | +43.2 to +48.9 | +51.7 | 200 | 200, 200 | 800 | 25 | small song pool: 25 playable angry songs |
| angry | Hindi | all 4 steps in the intended direction | % | 85.0 | 1.0 | 0.0 | +84.0 | +79.0 to +89.0 | +85.0 | 200 | 200, 200 |  | 25 | small song pool: 25 playable angry songs |
| angry | Hindi | song 5 vs song 1 in the intended direction | % | 100.0 | 50.0 | 43.5 | +50.0 | +43.0 to +56.5 | +56.5 | 200 | 200, 200 |  | 25 | small song pool: 25 playable angry songs |
| angry | Hindi | valence change, song 5 minus song 1 (calming paths) | valence points | 1.5 | -0.2 | 0.0 | +1.7 | +1.5 to +1.9 | +1.4 | 200 | 200, 200 |  | 25 | small song pool: 25 playable angry songs |
| angry | Tamil | steps in the intended direction | % | 92.0 | 53.0 | 45.9 | +39.0 | +36.0 to +42.0 | +46.1 | 200 | 200, 200 | 800 | 24 | small song pool: 24 playable angry songs |
| angry | Tamil | all 4 steps in the intended direction | % | 73.0 | 0.5 | 0.5 | +72.5 | +66.0 to +78.5 | +72.5 | 200 | 200, 200 |  | 24 | small song pool: 24 playable angry songs |
| angry | Tamil | song 5 vs song 1 in the intended direction | % | 100.0 | 51.0 | 48.5 | +49.0 | +42.5 to +55.5 | +51.5 | 200 | 200, 200 |  | 24 | small song pool: 24 playable angry songs |
| angry | Tamil | valence change, song 5 minus song 1 (calming paths) | valence points | 1.5 | -0.1 | -0.0 | +1.5 | +1.3 to +1.8 | +1.5 | 200 | 200, 200 |  | 24 | small song pool: 24 playable angry songs |
| angry | Korean | steps in the intended direction | % | 95.8 | 47.5 | 41.4 | +48.3 | +45.4 to +51.0 | +54.4 | 200 | 200, 200 | 800 | 27 | small song pool: 27 playable angry songs |
| angry | Korean | all 4 steps in the intended direction | % | 84.5 | 0.5 | 0.0 | +84.0 | +79.0 to +89.0 | +84.5 | 200 | 200, 200 |  | 27 | small song pool: 27 playable angry songs |
| angry | Korean | song 5 vs song 1 in the intended direction | % | 100.0 | 43.5 | 42.0 | +56.5 | +49.5 to +63.0 | +58.0 | 200 | 200, 200 |  | 27 | small song pool: 27 playable angry songs |
| angry | Korean | valence change, song 5 minus song 1 (calming paths) | valence points | 1.4 | -0.2 | -0.0 | +1.5 | +1.3 to +1.8 | +1.4 | 200 | 200, 200 |  | 27 | small song pool: 27 playable angry songs |
| fear | English | steps in the intended direction | % | 99.8 | 51.4 | 48.0 | +48.4 | +45.9 to +50.9 | +51.8 | 200 | 200, 200 | 800 | 162 |  |
| fear | English | all 4 steps in the intended direction | % | 99.0 | 1.5 | 1.0 | +97.5 | +95.0 to +99.5 | +98.0 | 200 | 200, 200 |  | 162 |  |
| fear | English | song 5 vs song 1 in the intended direction | % | 100.0 | 52.5 | 46.0 | +47.5 | +40.5 to +54.5 | +54.0 | 200 | 200, 200 |  | 162 |  |
| fear | English | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | 0.0 | -0.0 | +2.0 | +1.7 to +2.3 | +2.0 | 200 | 200, 200 |  | 162 |  |
| fear | Hindi | steps in the intended direction | % | 93.6 | 48.6 | 46.2 | +45.0 | +42.1 to +47.9 | +47.4 | 200 | 200, 200 | 800 | 28 | small song pool: 28 playable fear songs |
| fear | Hindi | all 4 steps in the intended direction | % | 75.5 | 0.0 | 1.0 | +75.5 | +69.5 to +81.5 | +74.5 | 200 | 200, 200 |  | 28 | small song pool: 28 playable fear songs |
| fear | Hindi | song 5 vs song 1 in the intended direction | % | 100.0 | 46.0 | 42.0 | +54.0 | +47.0 to +61.0 | +58.0 | 200 | 200, 200 |  | 28 | small song pool: 28 playable fear songs |
| fear | Hindi | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | -0.0 | -0.0 | +2.1 | +1.8 to +2.3 | +2.1 | 200 | 200, 200 |  | 28 | small song pool: 28 playable fear songs |
| fear | Tamil | steps in the intended direction | % | 94.8 | 50.0 | 49.8 | +44.8 | +41.8 to +47.6 | +45.0 | 200 | 200, 200 | 800 | 26 | small song pool: 26 playable fear songs |
| fear | Tamil | all 4 steps in the intended direction | % | 81.5 | 0.5 | 0.0 | +81.0 | +75.0 to +86.5 | +81.5 | 200 | 200, 200 |  | 26 | small song pool: 26 playable fear songs |
| fear | Tamil | song 5 vs song 1 in the intended direction | % | 100.0 | 51.5 | 53.5 | +48.5 | +41.5 to +55.5 | +46.5 | 200 | 200, 200 |  | 26 | small song pool: 26 playable fear songs |
| fear | Tamil | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | -0.0 | 0.0 | +2.1 | +1.8 to +2.3 | +2.0 | 200 | 200, 200 |  | 26 | small song pool: 26 playable fear songs |
| fear | Korean | steps in the intended direction | % | 96.0 | 48.6 | 47.8 | +47.4 | +44.4 to +50.1 | +48.2 | 200 | 200, 200 | 800 | 18 | small song pool: 18 playable fear songs |
| fear | Korean | all 4 steps in the intended direction | % | 84.5 | 0.5 | 0.5 | +84.0 | +79.0 to +89.0 | +84.0 | 200 | 200, 200 |  | 18 | small song pool: 18 playable fear songs |
| fear | Korean | song 5 vs song 1 in the intended direction | % | 100.0 | 47.5 | 45.5 | +52.5 | +45.5 to +59.0 | +54.5 | 200 | 200, 200 |  | 18 | small song pool: 18 playable fear songs |
| fear | Korean | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | -0.1 | -0.0 | +2.1 | +1.9 to +2.3 | +2.0 | 200 | 200, 200 |  | 18 | small song pool: 18 playable fear songs |
| disgust | English | steps in the intended direction | % | 89.9 | 51.4 | 48.5 | +38.5 | +35.6 to +41.5 | +41.4 | 200 | 200, 200 | 800 | 162 |  |
| disgust | English | all 4 steps in the intended direction | % | 62.0 | 1.0 | 1.0 | +61.0 | +54.0 to +68.0 | +61.0 | 200 | 200, 200 |  | 162 |  |
| disgust | English | song 5 vs song 1 in the intended direction | % | 100.0 | 49.0 | 45.0 | +51.0 | +44.0 to +58.5 | +55.0 | 200 | 200, 200 |  | 162 |  |
| disgust | English | valence change, song 5 minus song 1 (calming paths) | valence points | 2.5 | 0.1 | 0.0 | +2.4 | +2.1 to +2.7 | +2.5 | 200 | 200, 200 |  | 162 |  |
| disgust | Hindi | steps in the intended direction | % | 67.6 | 50.2 | 46.2 | +17.4 | +13.9 to +21.1 | +21.4 | 200 | 200, 200 | 800 | 26 | small song pool: 26 playable disgust songs |
| disgust | Hindi | all 4 steps in the intended direction | % | 16.5 | 0.0 | 0.5 | +16.5 | +11.5 to +21.5 | +16.0 | 200 | 200, 200 |  | 26 | small song pool: 26 playable disgust songs |
| disgust | Hindi | song 5 vs song 1 in the intended direction | % | 99.5 | 51.0 | 51.5 | +48.5 | +42.0 to +55.0 | +48.0 | 200 | 200, 200 |  | 26 | small song pool: 26 playable disgust songs |
| disgust | Hindi | valence change, song 5 minus song 1 (calming paths) | valence points | 2.6 | 0.2 | 0.0 | +2.4 | +2.2 to +2.6 | +2.6 | 200 | 200, 200 |  | 26 | small song pool: 26 playable disgust songs |
| disgust | Tamil | steps in the intended direction | % | 72.1 | 48.9 | 45.9 | +23.2 | +19.8 to +26.8 | +26.2 | 200 | 200, 200 | 800 | 28 | small song pool: 28 playable disgust songs |
| disgust | Tamil | all 4 steps in the intended direction | % | 20.5 | 1.0 | 0.0 | +19.5 | +14.0 to +25.5 | +20.5 | 200 | 200, 200 |  | 28 | small song pool: 28 playable disgust songs |
| disgust | Tamil | song 5 vs song 1 in the intended direction | % | 100.0 | 47.0 | 47.5 | +53.0 | +45.5 to +60.0 | +52.5 | 200 | 200, 200 |  | 28 | small song pool: 28 playable disgust songs |
| disgust | Tamil | valence change, song 5 minus song 1 (calming paths) | valence points | 2.6 | 0.2 | -0.0 | +2.3 | +2.1 to +2.6 | +2.6 | 200 | 200, 200 |  | 28 | small song pool: 28 playable disgust songs |
| disgust | Korean | steps in the intended direction | % | 77.6 | 49.4 | 46.2 | +28.2 | +24.8 to +31.8 | +31.4 | 200 | 200, 200 | 800 | 23 | small song pool: 23 playable disgust songs |
| disgust | Korean | all 4 steps in the intended direction | % | 29.0 | 0.5 | 0.5 | +28.5 | +22.5 to +35.0 | +28.5 | 200 | 200, 200 |  | 23 | small song pool: 23 playable disgust songs |
| disgust | Korean | song 5 vs song 1 in the intended direction | % | 100.0 | 58.0 | 45.5 | +42.0 | +35.0 to +49.0 | +54.5 | 200 | 200, 200 |  | 23 | small song pool: 23 playable disgust songs |
| disgust | Korean | valence change, song 5 minus song 1 (calming paths) | valence points | 2.5 | -0.1 | -0.0 | +2.5 | +2.3 to +2.8 | +2.5 | 200 | 200, 200 |  | 23 | small song pool: 23 playable disgust songs |
