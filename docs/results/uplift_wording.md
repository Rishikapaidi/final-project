# What the app says with an Uplift path, by start emotion

From the app's own code on the English Uplift paths of the playlist check (200 readings per start emotion).
Other languages add " in <language>" to the subtitle. Nothing here was changed.

Shown with every Uplift path, whatever the start emotion:

- choice card (button): "Lift my mood", with the text "Start where you are and move, song by song, towards something lighter."
- path card label: "Your path"; list headings "Your path, in play order" and "Keep listening"
- path card play button: no text, labelled "Play" for screen readers

## sad (200 paths; intended: brighter)

- Path card title: "From heavy to hopeful" (200)
- Subtitle: "5 songs in order, then more happy songs" (200)
- Mood card sentence (chosen by the reading's top emotion): "It looks like a heavy moment. The playlist starts with songs that match it, eases into something calmer and ends a little brighter. Skip ahead whenever you like." (190); "You seem fairly steady, with some heavier feelings underneath. The playlist starts there and moves gently on." (8); "You seem caught off guard, with some heavier feelings underneath. The playlist starts there and moves gently towards steadier songs." (2)

## angry (200 paths; intended: calmer)

- Path card title: "From tense to calm" (200)
- Subtitle: "5 songs in order, then more neutral songs" (200)
- Mood card sentence (chosen by the reading's top emotion): "You seem fired up. The playlist starts with songs that have some edge and moves step by step towards calmer ones." (183); "You seem fairly steady, with some heavier feelings underneath. The playlist starts there and moves gently on." (10); "You seem caught off guard, with some heavier feelings underneath. The playlist starts there and moves gently towards steadier songs." (4); "There's some brightness here, with heavier feelings alongside it. The playlist starts with the heavier part and moves gently on from there." (3)

## fear (200 paths; intended: calmer)

- Path card title: "From uneasy to calm" (200)
- Subtitle: "5 songs in order, then more neutral songs" (200)
- Mood card sentence (chosen by the reading's top emotion): "You seem a bit on edge. The playlist starts close to that feeling and moves step by step towards calmer songs." (195); "You seem caught off guard, with some heavier feelings underneath. The playlist starts there and moves gently towards steadier songs." (3); "You seem fairly steady, with some heavier feelings underneath. The playlist starts there and moves gently on." (2)

## disgust (200 paths; intended: calmer)

- Path card title: "From unsettled to calm" (200)
- Subtitle: "5 songs in order, then more neutral songs" (200)
- Mood card sentence (chosen by the reading's top emotion): "Something seems to have left a bad taste. The playlist starts in that mood and drifts towards calmer songs." (198); "You seem fairly steady, with some heavier feelings underneath. The playlist starts there and moves gently on." (2)

## "brighter", "hopeful" or "lighter" on a calming path (angry, fear, disgust)

- none in the path card title, subtitle or mood card sentence
- the choice card's text "Start where you are and move, song by song, towards something lighter." is the same for every start emotion, calming ones included
- About (shell.py): "a five-song path from a low mood towards a lighter one", for every start emotion
