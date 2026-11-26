# Uplift paths by start emotion (English, Hindi, Tamil and Korean together)

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

| start emotion | intended | measure | unit | Uplift | Random | Resonance | Uplift - Random | 95% interval of Uplift - Random | Uplift - Resonance | n paths | n paths (random, Resonance) | n steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sad | brighter (valence up) | steps in the intended direction | % | 99.4 | 50.1 | 44.9 | +49.3 | +48.1 to +50.6 | +54.5 | 800 | 800, 800 | 3200 |
| sad | brighter (valence up) | all 4 steps in the intended direction | % | 97.8 | 1.1 | 0.2 | +96.6 | +95.2 to +97.9 | +97.5 | 800 | 800, 800 |  |
| sad | brighter (valence up) | song 5 vs song 1 in the intended direction | % | 100.0 | 52.1 | 46.5 | +47.9 | +44.5 to +51.5 | +53.5 | 800 | 800, 800 |  |
| angry | calmer (arousal down) | steps in the intended direction | % | 95.4 | 49.4 | 42.7 | +46.0 | +44.5 to +47.4 | +52.7 | 800 | 800, 800 | 3200 |
| angry | calmer (arousal down) | all 4 steps in the intended direction | % | 83.5 | 0.8 | 0.1 | +82.8 | +80.0 to +85.4 | +83.4 | 800 | 800, 800 |  |
| angry | calmer (arousal down) | song 5 vs song 1 in the intended direction | % | 100.0 | 47.5 | 42.2 | +52.5 | +49.0 to +56.0 | +57.8 | 800 | 800, 800 |  |
| angry | calmer (arousal down) | valence change, song 5 minus song 1 (calming paths) | valence points | 1.4 | -0.1 | 0.0 | +1.6 | +1.4 to +1.7 | +1.4 | 800 | 800, 800 |  |
| fear | calmer (arousal down) | steps in the intended direction | % | 96.0 | 49.7 | 47.9 | +46.4 | +45.0 to +47.8 | +48.1 | 800 | 800, 800 | 3200 |
| fear | calmer (arousal down) | all 4 steps in the intended direction | % | 85.1 | 0.6 | 0.6 | +84.5 | +81.8 to +86.9 | +84.5 | 800 | 800, 800 |  |
| fear | calmer (arousal down) | song 5 vs song 1 in the intended direction | % | 100.0 | 49.4 | 46.8 | +50.6 | +47.1 to +54.1 | +53.2 | 800 | 800, 800 |  |
| fear | calmer (arousal down) | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | -0.0 | -0.0 | +2.1 | +1.9 to +2.2 | +2.0 | 800 | 800, 800 |  |
| disgust | calmer (arousal down) | steps in the intended direction | % | 76.8 | 50.0 | 46.7 | +26.8 | +25.1 to +28.6 | +30.1 | 800 | 800, 800 | 3200 |
| disgust | calmer (arousal down) | all 4 steps in the intended direction | % | 32.0 | 0.6 | 0.5 | +31.4 | +28.0 to +34.6 | +31.5 | 800 | 800, 800 |  |
| disgust | calmer (arousal down) | song 5 vs song 1 in the intended direction | % | 99.9 | 51.2 | 47.4 | +48.6 | +45.4 to +52.1 | +52.5 | 800 | 800, 800 |  |
| disgust | calmer (arousal down) | valence change, song 5 minus song 1 (calming paths) | valence points | 2.5 | 0.1 | 0.0 | +2.4 | +2.3 to +2.5 | +2.5 | 800 | 800, 800 |  |
