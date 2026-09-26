# LinkedIn announcement video

`ventx-announcement.mp4`: 40 s, 1080×1080 (square, fits the LinkedIn feed), 30 fps, H.264, with a silent AAC track. Every number shown comes from the main README.

The video is generated from `announcement.html`, where each frame is a pure function of time (`render(t)`).

- **Preview:** open `announcement.html` in a browser. It plays on a loop in real time.
- **Re-export:** `FFMPEG=/path/to/ffmpeg node render_video.js [out.mp4] [fps]`. This needs Playwright with Chromium and an ffmpeg build that has libx264. Rendering takes about 5 minutes.

The fonts are Inter and JetBrains Mono (both under the SIL Open Font License), bundled in `fonts/`.
