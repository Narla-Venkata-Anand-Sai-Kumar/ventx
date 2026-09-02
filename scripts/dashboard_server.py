"""Live local training dashboard for VenTX-100K's base pretrain -- same idea as
Chaos-135M's dashboard (stdlib-only HTTP server, polls the training log fresh off disk
every few seconds, nothing cached or faked), trimmed to what's actually applicable here:
no leaderboard/Intelligence-Index projection (that's specific to Chaos-135M's Open SLM
Leaderboard entry, which VenTX doesn't have), no SFT panel (VenTX has no SFT phase yet).
Just real progress: step/token count, loss curve, throughput, VRAM, ETA, checkpoints.

Run:  python scripts/dashboard_server.py
Then open http://localhost:8790
"""
import glob
import http.server
import json
import os
import re
import socketserver
import time

RUN_NAME = os.environ.get("VENTX_RUN", "ventx-225m-pretrain")
LOG_PATH = os.environ.get("VENTX_LOG", f"C:/ventx-data/checkpoints/{RUN_NAME}.jsonl")
CKPT_DIR = os.environ.get("VENTX_CKPT_DIR", "C:/ventx-data/checkpoints")
PORT = int(os.environ.get("VENTX_DASHBOARD_PORT", "8790"))
# train.py's startup banner is print()-ed to stdout only, never written to the .jsonl log
# (same gap Chaos-135M's own pretrain log has) -- these are this run's real launch values,
# used as a fallback whenever the banner itself isn't available to parse.
DEFAULT_TOTAL_STEPS = int(os.environ.get("VENTX_TOTAL_STEPS", "32500"))
DEFAULT_TOKENS_PER_STEP = int(os.environ.get("VENTX_TOKENS_PER_STEP", "131072"))
DEFAULT_SEQ_LEN = int(os.environ.get("VENTX_SEQ_LEN", "32768"))
DECAY_FRAC = float(os.environ.get("VENTX_DECAY_FRAC", "0.2"))
WARMUP_STEPS = int(os.environ.get("VENTX_WARMUP_STEPS", "800"))


def tail_lines(path, max_bytes=2_000_000):
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()
        data = f.read()
    return data.decode("utf-8", errors="ignore").splitlines()


def parse_training_log():
    """Same block-collection handling as Chaos-135M's parser: a banner CAN appear as
    multi-line indent=2 JSON if this run's stdout is ever redirected into this same
    file (recommended for future runs -- see USAGE), everything else is single-line."""
    lines = tail_lines(LOG_PATH)
    train_recs, val_recs, events = [], [], []
    banner = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "{":
            block = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() != "}":
                block.append(lines[j])
                j += 1
            if j < len(lines):
                block.append("}")
                try:
                    banner = json.loads("\n".join(block))
                except json.JSONDecodeError:
                    pass
                i = j + 1
                continue
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                if "val_loss" in rec:
                    val_recs.append(rec)
                elif "loss" in rec and "tok_per_s" in rec:
                    train_recs.append(rec)
            except json.JSONDecodeError:
                pass
        i += 1
    for line in lines[-200:]:
        if re.match(r"^(initialized weights|resumed from|saved |time limit reached)", line):
            events.append(line.strip())
    return train_recs, val_recs, events[-15:], banner


def checkpoint_list():
    out = []
    pat = re.compile(rf"^{re.escape(RUN_NAME)}_step(\d+)\.pt$")
    for p in glob.glob(os.path.join(CKPT_DIR, f"{RUN_NAME}_step*.pt")):
        m = pat.match(os.path.basename(p))
        if m:
            out.append(dict(step=int(m.group(1)), size_gb=round(os.path.getsize(p) / 1e9, 2),
                            mtime=os.path.getmtime(p)))
    out.sort(key=lambda r: r["step"])
    return out


def build_status():
    train_recs, val_recs, events, banner = parse_training_log()
    latest = train_recs[-1] if train_recs else None
    latest_val = val_recs[-1] if val_recs else None
    ckpts = checkpoint_list()

    total_steps = DEFAULT_TOTAL_STEPS
    tok_per_step = DEFAULT_TOKENS_PER_STEP
    seq_len = DEFAULT_SEQ_LEN
    params_m = flops_per_tok = rope_theta = None
    if banner:
        if banner.get("tokens_per_step"):
            tok_per_step = banner["tokens_per_step"]
        if banner.get("total_tokens") and tok_per_step:
            total_steps = banner["total_tokens"] // tok_per_step
        seq_len = banner.get("seq_len", seq_len)
        params_m = banner.get("params_m")
        flops_per_tok = banner.get("flops_per_token")
        rope_theta = banner.get("rope_theta")

    step = latest["step"] if latest else 0
    decay_start = total_steps - int(total_steps * DECAY_FRAC)
    phase = "decay" if step >= decay_start else ("warmup" if step < WARMUP_STEPS else "stable")

    recent_tps = [r["tok_per_s"] for r in train_recs[-50:]]
    avg_tps = sum(recent_tps) / len(recent_tps) if recent_tps else 0
    remaining_steps = max(0, total_steps - step)
    eta_seconds = remaining_steps * tok_per_step / avg_tps if avg_tps else None

    log_mtime = os.path.getmtime(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    stalled = (time.time() - log_mtime) > 300 if log_mtime else True   # generous window --
    # a single macro-step at 32768 ctx takes ~45-90s, so the usual 180s chaos threshold
    # would false-positive between ordinary log lines here.

    return dict(
        server_time=time.time(),
        run_name=RUN_NAME,
        params_m=params_m, seq_len=seq_len, rope_theta=rope_theta, flops_per_token=flops_per_tok,
        step=step,
        total_steps=total_steps,
        tokens=step * tok_per_step,
        total_tokens=total_steps * tok_per_step,
        tokens_per_step=tok_per_step,
        phase=phase,
        decay_start_step=decay_start,
        stalled=stalled,
        log_age_seconds=round(time.time() - log_mtime, 1) if log_mtime else None,
        latest=latest,
        latest_val=latest_val,
        avg_tok_per_s=round(avg_tps),
        eta_seconds=eta_seconds,
        train_history=train_recs[-400:],
        val_history=val_recs,
        events=events,
        checkpoints=ckpts,
    )


PAGE = None


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            try:
                body = json.dumps(build_status()).encode("utf-8")
                code = 200
            except Exception as e:
                body = json.dumps(dict(error=f"{type(e).__name__}: {e}")).encode("utf-8")
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global PAGE
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(html_path, encoding="utf-8") as f:
        PAGE = f.read()
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"VenTX-100K dashboard: http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
