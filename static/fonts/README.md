# Fonts

Bundled so the page looks the same offline. Streamlit serves this folder at
`app/static/` (`server.enableStaticServing = true` in `.streamlit/config.toml`).

| File | Font | Used for | Licence |
|---|---|---|---|
| `Fraunces-latin-variable.woff2` | Fraunces (variable weight), by The Fraunces Project Authors | headings | SIL Open Font License 1.1, `OFL-Fraunces.txt` |
| `Inter-latin-variable.woff2` | Inter (variable weight), by The Inter Project Authors | body text | SIL Open Font License 1.1, `OFL-Inter.txt` |

Both files are the Latin subsets as packaged by Fontsource
(`@fontsource-variable/fraunces` 5.3.0 and `@fontsource-variable/inter` 5.3.0).
They hold Latin letters only: Hindi, Tamil, Telugu, Malayalam and Korean song
titles are drawn in the system's fonts. If a file can't be loaded, headings
fall back to Georgia and body text to the system sans-serif font.
