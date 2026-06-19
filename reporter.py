"""
reporter.py — Builds styled HTML email report and sends via Brevo SMTP.

Changes vs original:
  - build_html_report(internships, full_time, report_date) — new signature.
    Accepts two pre-sorted lists instead of one flat list.
  - Two separate sections: "Top Internships" and "Top Full-Time Jobs"
    each ranked by relevance_score descending (caller sorts before passing in).
  - job_type badge on every card: INTERN (purple) | FULL-TIME (blue)
  - Apply button URL is already a direct link — set by filters.fix_linkedin_url()
  - Header stats show internship count + job count separately
  - send_email() signature unchanged — caller builds the subject
"""

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


# ── Card sub-components (unchanged from original) ────────────────────────────

def _score_ring(score: int, color: str) -> str:
    return f"""
    <div style="text-align:center;min-width:52px;flex-shrink:0">
      <div style="width:50px;height:50px;border-radius:50%;
                  border:2.5px solid {color};
                  display:flex;align-items:center;justify-content:center;
                  font-size:15px;font-weight:700;color:{color};
                  background:{color}15;margin:0 auto;">
        {score}
      </div>
      <div style="font-size:10px;color:#94a3b8;margin-top:3px;
                  text-transform:uppercase;letter-spacing:0.5px;">score</div>
    </div>"""


def _breakdown_bars(bd: dict, color: str) -> str:
    axes = [("role", 30), ("skills", 30), ("level", 25), ("company", 15)]
    rows = ""
    for key, mx in axes:
        val = bd.get(key, 0)
        pct = int(val / mx * 100)
        rows += f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
          <div style="width:52px;font-size:10px;color:#94a3b8;
                      text-transform:uppercase;letter-spacing:0.4px;">{key}</div>
          <div style="flex:1;background:#f1f5f9;border-radius:4px;height:5px;">
            <div style="width:{pct}%;background:{color};
                        border-radius:4px;height:5px;"></div>
          </div>
          <div style="font-size:10px;color:#64748b;width:30px;
                      text-align:right;">{val}/{mx}</div>
        </div>"""
    return f'<div style="margin:10px 0;">{rows}</div>'


def _skill_tags(skills: list, color: str) -> str:
    return "".join(
        f'<span style="background:{color}18;color:{color};padding:2px 9px;'
        f'border-radius:20px;font-size:11px;font-weight:500;'
        f'margin:2px;display:inline-block;">{s}</span>'
        for s in skills
    )


def _red_flags_html(flags: list) -> str:
    if not flags:
        return ""
    items = " &nbsp;·&nbsp; ".join(f"⚠ {f}" for f in flags)
    return (
        f'<div style="color:#f43f5e;font-size:11px;margin-top:6px;">'
        f'{items}</div>'
    )


def _pri_badge(priority: str, color: str) -> str:
    return (
        f'<span style="background:{color}18;color:{color};padding:2px 10px;'
        f'border-radius:20px;font-size:11px;font-weight:600;">'
        f'{priority}</span>'
    )


def _type_badge(job_type: str) -> str:
    """INTERN badge (purple) or FULL-TIME badge (blue)."""
    if job_type == "internship":
        return (
            '<span style="background:#ede9fe;color:#6d28d9;padding:2px 9px;'
            'border-radius:20px;font-size:11px;font-weight:600;">INTERN</span>'
        )
    return (
        '<span style="background:#e0f2fe;color:#0369a1;padding:2px 9px;'
        'border-radius:20px;font-size:11px;font-weight:600;">FULL-TIME</span>'
    )


def _job_card(j: dict, color: str) -> str:
    bd       = j.get("score_breakdown", {})
    score    = j.get("relevance_score", 0)
    pri      = j.get("priority", "LOW")
    job_type = j.get("job_type", "full-time")
    colors   = {"HIGH": "#0ea5e9", "MEDIUM": "#8b5cf6", "LOW": "#64748b"}
    pc       = colors.get(pri, "#64748b")

    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;
                border-radius:14px;padding:20px;margin-bottom:14px;
                border-left:4px solid {color};">
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;gap:12px;">
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:8px;
                      margin-bottom:4px;flex-wrap:wrap;">
            {_pri_badge(pri, pc)}
            {_type_badge(job_type)}
            <span style="font-size:11px;color:#94a3b8;">
              {j.get("source", "")}
            </span>
          </div>
          <div style="font-size:15px;font-weight:700;color:#0f172a;
                      margin-bottom:3px;word-break:break-word;">
            {j["title"]}
          </div>
          <div style="font-size:12px;color:#64748b;">
            🏢 <strong>{j["company"]}</strong>
            &nbsp;·&nbsp; 📍 {j.get("location", "")}
            &nbsp;·&nbsp; 🗓 {str(j.get("posted", ""))[:10]}
          </div>
        </div>
        {_score_ring(score, color)}
      </div>
      {_breakdown_bars(bd, color)}
      <div style="font-size:13px;color:#475569;line-height:1.6;margin:8px 0;">
        {j.get("match_reason", "")}
      </div>
      <div style="margin-bottom:6px;">
        {_skill_tags(j.get("key_match_skills", []), color)}
      </div>
      {_red_flags_html(j.get("red_flags", []))}
      <a href="{j.get('url', '#')}" target="_blank"
         style="display:inline-block;margin-top:12px;padding:8px 20px;
                background:{color};color:#ffffff;border-radius:8px;
                font-size:13px;font-weight:600;text-decoration:none;">
        Apply →
      </a>
    </div>"""


def _section(heading: str, emoji: str, items: list, color: str) -> str:
    if not items:
        return ""
    cards = "".join(_job_card(j, color) for j in items)
    return f"""
    <div style="margin-bottom:32px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <span style="font-size:20px;">{emoji}</span>
        <h2 style="margin:0;font-size:17px;font-weight:700;color:#0f172a;">
          {heading}
        </h2>
        <span style="background:{color}18;color:{color};
                     padding:3px 12px;border-radius:20px;
                     font-size:12px;font-weight:600;">{len(items)}</span>
      </div>
      {cards}
    </div>"""


# ── Main entry point ──────────────────────────────────────────────────────────

def build_html_report(
    internships: list,
    full_time_jobs: list,
    report_date: str,
) -> str:
    """
    Build a styled HTML email report.

    Args:
        internships:    Jobs with job_type == "internship", pre-sorted by score desc.
        full_time_jobs: Jobs with job_type == "full-time",  pre-sorted by score desc.
        report_date:    ISO date string, e.g. "2026-06-17".

    The caller (crew.build_report_tool) is responsible for splitting and sorting.
    Apply → buttons use job["url"] which is already a direct link
    (set by filters.fix_linkedin_url before scoring).
    """
    total = len(internships) + len(full_time_jobs)

    # Stats for the header
    intern_high  = sum(1 for j in internships    if j.get("priority") == "HIGH")
    ft_high      = sum(1 for j in full_time_jobs if j.get("priority") == "HIGH")
    total_high   = intern_high + ft_high

    intern_med   = sum(1 for j in internships    if j.get("priority") == "MEDIUM")
    ft_med       = sum(1 for j in full_time_jobs if j.get("priority") == "MEDIUM")
    total_medium = intern_med + ft_med

    intern_low   = total - intern_high - intern_med
    ft_low       = len(full_time_jobs) - ft_high - ft_med
    total_low    = intern_low + ft_low

    internship_section = _section(
        "🎓 Top Internships", "🎓", internships, "#7c3aed"
    )
    fulltime_section = _section(
        "💼 Top Full-Time Jobs", "💼", full_time_jobs, "#0ea5e9"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Job Intel — {report_date}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
             Roboto,sans-serif;max-width:740px;margin:0 auto;
             padding:24px;background:#f8fafc;color:#0f172a;">

  <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
              border-radius:18px;padding:28px;margin-bottom:28px;color:#fff;">
    <div style="font-size:11px;letter-spacing:2px;color:#94a3b8;
                text-transform:uppercase;margin-bottom:6px;">Daily Digest</div>
    <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px;">
      🤖 Job Intel
    </div>
    <div style="font-size:13px;color:#94a3b8;margin-top:4px;">
      {report_date} &nbsp;·&nbsp; CrewAI + Groq llama-3.3-70b &nbsp;·&nbsp;
      Intent-matched for Shashwat Mahajan
    </div>

    <div style="display:flex;gap:24px;margin-top:20px;flex-wrap:wrap;">
      <div>
        <div style="font-size:28px;font-weight:800;">{total}</div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Total matches</div>
      </div>
      <div style="width:1px;background:#334155;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#a78bfa;">
          {len(internships)}
        </div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Internships</div>
      </div>
      <div style="width:1px;background:#334155;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#38bdf8;">
          {len(full_time_jobs)}
        </div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Full-Time Jobs</div>
      </div>
      <div style="width:1px;background:#334155;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#34d399;">
          {total_high}
        </div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">High Priority</div>
      </div>
      <div style="width:1px;background:#334155;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#fbbf24;">
          {total_medium}
        </div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Medium Priority</div>
      </div>
    </div>
  </div>

  {internship_section}
  {fulltime_section}

  <div style="text-align:center;font-size:11px;color:#94a3b8;
              margin-top:28px;padding-top:16px;
              border-top:1px solid #e2e8f0;">
    Job Intel Agent v2.1 &nbsp;·&nbsp; CrewAI + Groq + Ollama &nbsp;·&nbsp;
    Filters: 0–2 yr exp · no senior/lead roles · direct apply links
  </div>
</body>
</html>"""


def send_email(html: str, cfg: dict, subject: str) -> None:
    """Send HTML digest via Brevo SMTP. Signature unchanged."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg["email_from"]
        msg["To"]      = cfg["email_to"]
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587)) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.sendmail(cfg["email_from"], cfg["email_to"], msg.as_string())

        log.info("Email sent successfully.")
    except Exception as e:
        log.error(f"Email failed: {e}")