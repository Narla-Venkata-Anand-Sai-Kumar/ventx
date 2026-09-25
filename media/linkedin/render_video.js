// Export announcement.html to an MP4 by stepping render(t) frame by frame.
// Usage: FFMPEG=/path/to/ffmpeg node render_video.js [out.mp4] [fps]
// Needs Playwright (Chromium) and an ffmpeg build with libx264.
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const path = require("path");

const out = process.argv[2] || path.join(__dirname, "ventx-announcement.mp4");
const fps = Number(process.argv[3] || 30);
const ffmpeg = process.env.FFMPEG || "ffmpeg";

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1080 } });
  await page.goto("file://" + path.join(__dirname, "announcement.html") + "?export");
  await page.evaluate(() => document.fonts.ready);
  const duration = await page.evaluate(() => window.DURATION);
  const frames = Math.round(duration * fps);

  const enc = spawn(ffmpeg, [
    "-y", "-f", "image2pipe", "-framerate", String(fps), "-i", "-",
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
    "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
    "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
    "-c:a", "aac", "-b:a", "128k", out,
  ], { stdio: ["pipe", "inherit", "inherit"] });

  const stage = await page.$("#stage");
  for (let i = 0; i < frames; i++) {
    await page.evaluate((t) => window.render(t), i / fps);
    const buf = await stage.screenshot({ type: "png" });
    if (!enc.stdin.write(buf)) await new Promise((r) => enc.stdin.once("drain", r));
    if (i % fps === 0) process.stderr.write(`\rframe ${i}/${frames}`);
  }
  enc.stdin.end();
  await new Promise((r) => enc.on("close", r));
  await browser.close();
  console.error(`\nwrote ${out}`);
})();
