"""Find a playable 30-second preview and album art for a song.

The song file only has Last.fm links, which cannot be played. This looks each
song up in the free iTunes Search API (no key needed) and keeps the first
result whose artist and title loosely match.

The API allows roughly 20 requests a minute and blocks a machine for a long
time after a burst, so results are cached on disk (CACHE_PATH), including "no
preview found", and each song is only ever looked up once.

By default the app is cache-only: it uses songs already in the cache and makes
no API calls. `scripts/fetch_previews.py` fills the cache ahead of time. Set
the environment variable LIVE_LOOKUPS=1 to let the app also look up songs it
has not seen; when the limit is hit, lookups then pause for RATE_LIMIT_PAUSE
seconds and unchecked songs are treated as not playable for now.
"""
import csv
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "data", "songs", "previews.csv")
CACHE_FIELDS = ["lastfm_url", "preview_url", "artwork_url", "itunes_title", "itunes_artist"]
SEARCH_URL = "https://itunes.apple.com/search"
MAX_WORKERS = 8
RATE_LIMIT_PAUSE = 60
LIVE_LOOKUPS = os.environ.get("LIVE_LOOKUPS") == "1"

_lock = threading.Lock()
_cache = None
_paused_until = 0.0


class RateLimited(Exception):
    pass


def _norm(text):
    """Lower-case letters and digits of any script, accents stripped."""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(ch for ch in text if ch.isalnum())


def _words(text):
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return {w for w in re.findall(r"\w+", text) if len(w) > 2}


def _loosely_same(a, b):
    """One name contains the other once punctuation, case and accents are ignored."""
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def _first_artist(artist):
    """The first-listed artist of a credit like "A, B & C feat. D"."""
    return re.split(r",|&|;|\s+(?:feat\.?|ft\.?|featuring|with|x)\s+", str(artist), maxsplit=1, flags=re.I)[0]


def _title_matches(title, found_title):
    return _loosely_same(title, found_title) or bool(_words(title) & _words(found_title))


def search_itunes(title, artist):
    """Return {preview_url, artwork_url, itunes_title, itunes_artist}, or None if no match.

    A result must loosely match the title and the artist. The artist is first
    compared as a whole credit; only if no result matches that way is the
    first-listed artist compared instead (multi-artist credits are often
    spelled differently, e.g. "G. V. Prakash, Saindhavi" vs "G.V. Prakash
    Kumar, Saindhavi"). Songs that matched before therefore keep the same
    result.

    Raises RateLimited when the API refuses the request.
    """
    response = requests.get(SEARCH_URL, timeout=15, params={
        "term": f"{title} {artist}", "media": "music", "entity": "song", "limit": 5})
    if response.status_code in (403, 429):
        raise RateLimited()
    response.raise_for_status()
    results = [r for r in response.json().get("results", [])
               if r.get("previewUrl") and _title_matches(title, r.get("trackName", ""))]
    first = _first_artist(artist)
    for same_artist in (lambda r: _loosely_same(artist, r.get("artistName", "")),
                        lambda r: len(_norm(first)) >= 3 and _loosely_same(first, _first_artist(r.get("artistName", "")))):
        for result in results:
            if same_artist(result):
                return {"preview_url": result["previewUrl"], "artwork_url": result.get("artworkUrl100", ""),
                        "itunes_title": result.get("trackName", ""), "itunes_artist": result.get("artistName", "")}
    return None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _cache[row["lastfm_url"]] = row
    return _cache


def remember(key, found):
    row = {"lastfm_url": key, "preview_url": "", "artwork_url": "", "itunes_title": "", "itunes_artist": ""}
    row.update(found or {})
    with _lock:
        cache = _load_cache()
        new_file = not os.path.exists(CACHE_PATH)
        cache[key] = row
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    return row


def cached(key):
    """The cached row for a song (preview_url is "" if it has none), or None if never looked up."""
    with _lock:
        return _load_cache().get(key)


def playable_urls():
    """The lastfm_url of every song known to have a preview."""
    with _lock:
        return {key for key, row in _load_cache().items() if row["preview_url"]}


def rate_limited():
    return time.time() < _paused_until


def _lookup(song):
    """song: (lastfm_url, title, artist). Returns the cache row, or None if it could not be checked."""
    global _paused_until
    key, title, artist = song
    row = cached(key)
    if row is not None:
        return row
    if not LIVE_LOOKUPS or rate_limited():
        return None
    try:
        return remember(key, search_itunes(title, artist))
    except RateLimited:
        _paused_until = time.time() + RATE_LIMIT_PAUSE
    except requests.RequestException:
        pass
    return None


def resolve(songs):
    """Look up many songs in parallel.

    songs: list of (lastfm_url, title, artist).
    Returns {lastfm_url: row} for the songs that have a preview.
    """
    songs = list(songs)
    if not songs:
        return {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(_lookup, songs))
    return {song[0]: row for song, row in zip(songs, rows) if row and row["preview_url"]}
