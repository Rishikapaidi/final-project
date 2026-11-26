# Uplift paths by language (all four start emotions; each path on its own intended axis)

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

| language | measure | unit | Uplift | Random | Resonance | Uplift - Random | 95% interval of Uplift - Random | Uplift - Resonance | n paths | n paths (random, Resonance) | n steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| English | steps in the intended direction | % | 96.8 | 50.4 | 45.9 | +46.4 | +45.0 to +47.8 | +50.9 | 800 | 800, 800 | 3200 |
| English | all 4 steps in the intended direction | % | 87.9 | 1.0 | 0.6 | +86.9 | +84.4 to +89.2 | +87.2 | 800 | 800, 800 |  |
| English | song 5 vs song 1 in the intended direction | % | 100.0 | 50.7 | 43.6 | +49.3 | +45.9 to +52.6 | +56.4 | 800 | 800, 800 |  |
| English | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | 0.0 | 0.0 | +1.9 | +1.8 to +2.1 | +2.0 | 600 | 600, 600 |  |
| Hindi | steps in the intended direction | % | 89.2 | 49.5 | 46.0 | +39.8 | +38.0 to +41.5 | +43.3 | 800 | 800, 800 | 3200 |
| Hindi | all 4 steps in the intended direction | % | 69.0 | 0.8 | 0.5 | +68.2 | +65.1 to +71.4 | +68.5 | 800 | 800, 800 |  |
| Hindi | song 5 vs song 1 in the intended direction | % | 99.9 | 48.9 | 47.6 | +51.0 | +47.4 to +54.4 | +52.2 | 800 | 800, 800 |  |
| Hindi | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | -0.0 | -0.0 | +2.1 | +1.9 to +2.2 | +2.0 | 600 | 600, 600 |  |
| Tamil | steps in the intended direction | % | 89.4 | 50.3 | 46.0 | +39.2 | +37.4 to +40.8 | +43.5 | 800 | 800, 800 | 3200 |
| Tamil | all 4 steps in the intended direction | % | 67.6 | 0.5 | 0.1 | +67.1 | +63.7 to +70.2 | +67.5 | 800 | 800, 800 |  |
| Tamil | song 5 vs song 1 in the intended direction | % | 100.0 | 51.0 | 48.1 | +49.0 | +45.5 to +52.4 | +51.9 | 800 | 800, 800 |  |
| Tamil | valence change, song 5 minus song 1 (calming paths) | valence points | 2.0 | 0.0 | 0.0 | +2.0 | +1.8 to +2.1 | +2.0 | 600 | 600, 600 |  |
| Korean | steps in the intended direction | % | 92.2 | 49.0 | 44.4 | +43.2 | +41.7 to +44.7 | +47.8 | 800 | 800, 800 | 3200 |
| Korean | all 4 steps in the intended direction | % | 73.9 | 0.9 | 0.2 | +73.0 | +69.9 to +76.1 | +73.6 | 800 | 800, 800 |  |
| Korean | song 5 vs song 1 in the intended direction | % | 100.0 | 49.6 | 43.5 | +50.4 | +46.9 to +53.6 | +56.5 | 800 | 800, 800 |  |
| Korean | valence change, song 5 minus song 1 (calming paths) | valence points | 1.9 | -0.1 | -0.0 | +2.1 | +1.9 to +2.2 | +1.9 | 600 | 600, 600 |  |
