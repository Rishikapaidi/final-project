"""Build a cleaned song file with seven emotion buckets from muse_v3.csv.

muse_v3.csv is read only, never modified. Only its intact rows are used: the
ones that still have a Last.fm link (about 36%). The rest had their title,
artist and link overwritten, so their mood values belong to a different song.
See data/README.md for the details.

A song goes into the bucket whose mood words appear most often in its `seeds`
(ties are dropped), provided it also passes that bucket's valence/arousal
check. Songs with none of the listed words are dropped, and so are songs whose
title or artist contains profanity, a slur or an explicit sexual word (whole
words, see PROFANITY).

Writes data/songs/songs_clean.csv with columns:
    title, artist, valence_tags, arousal_tags, seeds, bucket, lastfm_url

Usage (from the project root):
    python scripts/prepare_songs.py
"""
import argparse
import os
import re

import pandas as pd

BUCKET_WORDS = {
    "happy": ["happy", "fun", "cheerful", "positive", "joyous", "playful", "carefree", "uplifting",
              "euphoric", "exuberant", "celebratory", "optimistic", "bright", "humorous", "silly",
              "perky", "energetic", "lively", "rollicking", "effervescent", "triumphant", "sweet",
              "comic", "good-natured", "amiable", "summery", "sparkling", "boisterous", "jovial",
              "gleeful", "ebullient", "rousing"],
    "sad": ["sad", "gloomy", "lonely", "bleak", "tragic", "melancholy", "somber", "funereal",
            "regretful", "weary", "wistful", "plaintive", "elegiac", "yearning", "bittersweet",
            "poignant", "distraught", "wintry", "autumnal", "sentimental", "nostalgic"],
    # Proxy words only: every song tagged angry/aggressive/hostile is in the
    # overwritten rows. "bitter" is left out (resentful rather than angry).
    "angry": ["harsh", "defiant", "feral", "fiery", "brash", "raucous", "rowdy", "gritty", "visceral",
              "provocative", "uncompromising", "thuggish", "rebellious", "urgent", "martial",
              "desperate"],
    # "halloween" is left out: it pulls in novelty and party songs.
    "fear": ["scary", "spooky", "eerie", "paranoid", "ominous", "macabre", "suspenseful",
             "threatening", "unsettling", "nervous", "anxious", "menacing", "scary music",
             "apocalyptic", "suffocating", "tense"],
    "disgust": ["sardonic", "cynical", "sarcastic", "acerbic", "snide", "satirical", "irreverent",
                "trashy", "sleazy", "greasy", "grim"],
    # Really "quirky/unusual" rather than startled; mostly eccentric and exotic.
    "surprise": ["eccentric", "quirky", "whimsical", "exotic", "capricious", "exciting", "thrilling",
                 "flashy", "giddy", "animated", "ecstatic"],
    "neutral": ["calm", "relaxed", "soothing", "gentle", "peaceful", "mellow", "laid-back", "quiet",
                "soft", "smooth", "light", "restrained", "reserved", "understated", "meditative",
                "warm", "airy", "flowing"],
}

# Valence/arousal check per bucket (valence runs ~0.2-8.5, arousal ~0.1-7.3).
BUCKET_CHECKS = {
    "happy": lambda v, a: (v >= 6.0) & (a >= 4.0),
    "sad": lambda v, a: (v <= 5.0) & (a <= 4.5),
    "angry": lambda v, a: (v <= 5.0) & (a >= 4.0),
    "fear": lambda v, a: v <= 5.0,
    "disgust": lambda v, a: v <= 4.5,
    "surprise": lambda v, a: (v >= 5.0) & (a >= 4.5),
    "neutral": lambda v, a: (v >= 4.0) & (v <= 6.8) & (a <= 4.2),
}

# Songs whose title or artist contains one of these as a whole word are
# dropped. Whole words only, so "Cockburn", "Glass" or "Shub-Niggurath" are not
# caught. Strong profanity and slurs only; milder or ambiguous words (hell,
# damn, ass, bastard, dick, cock, piss, crap, prick) are deliberately not listed.
# Whole-word matching misses compounds, so the ones found in the data are
# listed explicitly.
PROFANITY = {
    "fuck", "fucks", "fucked", "fucking", "fuckin", "fucker", "fuckers", "motherfucker",
    "motherfuckers", "motherfucking", "starfucker", "f*ck", "f**k", "f***", "f****n", "f***ing",
    "shit", "shits", "shitty", "shite", "bullshit", "shithole", "sh*t", "s**t",
    "bitch", "bitches", "b*tch", "cunt", "cunts", "asshole", "assholes", "dickhead",
    "whore", "whores", "slut", "sluts", "pussy",
    "retard", "retards", "retarded", "faggot", "faggots", "fag", "fags", "dyke", "dykes", "tranny",
    "nigger", "niggers", "nigga", "niggas", "chink", "chinks", "kike", "kikes", "spic", "spics",
    "wetback", "wetbacks",
    "rape", "raped", "porn",
    "goatwhore", "pzychobitch", "chromebitch", "shitloads", "shitslide",
    "psychoparanoidflowerpowerelectroshitsong",
    # Explicit anatomy and sex-act words. Broader words (sex, sexy, sexual, naked,
    # lust, virgin, bondage, hooker, balls) are left off: in this data they are
    # mostly band names and ordinary titles such as "Sexual Healing".
    "penis", "penises", "vagina", "vaginas", "anal", "anus", "clit", "tits", "boobs", "dildo",
    "phallus", "scrotum", "testicles", "genitals", "semen", "sperm", "cum", "cumshot", "jizz",
    "boner", "erection", "orgasm", "orgasms", "orgy", "orgies", "masturbate", "masturbating",
    "masturbation", "blowjob", "handjob", "fellatio", "sodomy", "bukkake",
}


def has_profanity(text):
    return any(word in PROFANITY for word in re.findall(r"[a-z*]+", str(text).lower()))


def build(source):
    raw = pd.read_csv(source)
    clean = raw[raw.lastfm_url.str.contains("last.fm/music", regex=False)].copy()
    seed_lists = clean.seeds.str.strip("[]").str.replace("'", "").str.split(", ")

    matches = pd.DataFrame({bucket: seed_lists.map(lambda seeds, words=set(words): sum(s in words for s in seeds))
                            for bucket, words in BUCKET_WORDS.items()})
    most = matches.max(axis=1)
    tied = matches.eq(most, axis=0).sum(axis=1) > 1
    by_words = matches.idxmax(axis=1).where((most > 0) & ~tied)

    v, a = clean.valence_tags, clean.arousal_tags
    clean["bucket"] = None
    for bucket, check in BUCKET_CHECKS.items():
        clean.loc[(by_words == bucket) & check(v, a), "bucket"] = bucket

    stats = {"rows in source": len(raw), "intact rows (Last.fm link)": len(clean),
             "no listed mood word": int((most == 0).sum()), "tied between buckets": int(((most > 0) & tied).sum()),
             "failed valence/arousal check": int((by_words.notna() & clean.bucket.isna()).sum())}
    songs = clean[clean.bucket.notna()].rename(columns={"track": "title"})
    songs = songs[["title", "artist", "valence_tags", "arousal_tags", "seeds", "bucket", "lastfm_url"]]
    profane = songs.title.map(has_profanity) | songs.artist.map(has_profanity)
    stats["profanity in title or artist"] = int(profane.sum())
    return songs[~profane].reset_index(drop=True), stats, songs[profane].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="muse_v3.csv")
    parser.add_argument("--out", default=os.path.join("data", "songs", "songs_clean.csv"))
    parser.add_argument("--list-dropped", action="store_true",
                        help="print the songs dropped by the profanity filter")
    args = parser.parse_args()

    assert len({w for words in BUCKET_WORDS.values() for w in words}) == sum(map(len, BUCKET_WORDS.values())), \
        "a mood word is listed in two buckets"
    songs, stats, dropped = build(args.source)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    songs.to_csv(args.out, index=False, encoding="utf-8")

    for name, n in stats.items():
        print(f"{name}: {n:,}")
    print(f"songs kept: {len(songs):,} -> {args.out}")
    summary = songs.groupby("bucket").agg(songs=("bucket", "size"), valence=("valence_tags", "mean"),
                                          arousal=("arousal_tags", "mean")).round(2)
    summary.insert(1, "dropped_for_profanity", dropped.bucket.value_counts().reindex(summary.index).fillna(0).astype(int))
    print(summary.to_string())
    if args.list_dropped:
        print(dropped[["bucket", "title", "artist"]].sort_values(["bucket", "artist"]).to_string(index=False))


if __name__ == "__main__":
    main()
