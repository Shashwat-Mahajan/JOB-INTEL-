"""
app.py — Job Agent frontend
Place at D:/job_agent/app.py and run: streamlit run app.py
"""

import streamlit as st
import subprocess
import json
from pathlib import Path

BASE        = Path(__file__).parent
CONFIG_DIR  = BASE / "config"
RESUME_DEST = CONFIG_DIR / "resume.pdf"
CONFIG_PATH = CONFIG_DIR / "config.json"


def run_and_stream(cmd: list[str], label: str, log_box) -> bool:
    """Run a subprocess and stream its stdout+stderr live into log_box.
    Returns True on success, False on failure."""
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log_lines.append(line)
            # Show last 20 lines so the box doesn't grow forever
            log_box.code("\n".join(log_lines[-20:]), language="text")
    proc.wait()
    return proc.returncode == 0


st.set_page_config(page_title="Job Agent", page_icon="🎯", layout="centered")

st.title("🎯 Job Agent")
st.caption("Upload your resume, enter your email, and we'll send you today's matching jobs.")

st.divider()

resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
email       = st.text_input("Your email address", placeholder="you@example.com")

st.divider()

st.info("⏱️ Processing takes 10–30 minutes depending on your hardware. Don't close this tab.")

if st.button("🚀 Find & Send Jobs", type="primary", use_container_width=True):

    if not resume_file:
        st.error("Please upload your resume.")
        st.stop()
    if not email or "@" not in email:
        st.error("Please enter a valid email address.")
        st.stop()

    # ── 1. Save resume ────────────────────────────────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_DEST.write_bytes(resume_file.read())

    # ── 2. Patch email into config.json ───────────────────────────────────
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg["email_to"]      = email
    cfg["email_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ── 3. Parse resume → profile.json ───────────────────────────────────
    st.markdown("**Step 1 of 2 — Parsing resume…**")
    log1 = st.empty()
    ok = run_and_stream(
        ["python", str(BASE / "setup_profile.py")],
        "setup_profile",
        log1,
    )
    if not ok:
        st.error("❌ Resume parsing failed. See output above for the exact error.")
        st.stop()
    st.success("✅ Resume parsed")

    # ── 4. Run pipeline ───────────────────────────────────────────────────
    st.markdown("**Step 2 of 2 — Searching jobs and sending email…**")
    log2 = st.empty()
    ok = run_and_stream(
        ["python", str(BASE / "main.py")],
        "main",
        log2,
    )
    if not ok:
        st.error("❌ Pipeline failed. See output above for the exact error.")
        st.stop()

    st.success(f"✅ Job digest sent to **{email}**! Check your inbox.")