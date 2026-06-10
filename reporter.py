"""
reporter.py
Builds the HTML email digest and sends it via Brevo SMTP.
"""

import smtplib
import logging
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def build_html_report(jobs: list[dict], report_date: str) -> str:
    """Build a styled HTML email report from scored jobs."""
    ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    jobs  = sorted(
        jobs,
        key=lambda j: (ORDER.get(j.get("priority", "LOW"), 3), -j.get("relevance_score", 0)),
    )

    high   = [j for j in jobs if j.get("priority") == "HIGH"]
    medium = [j for j in jobs if j.get("priority") == "MEDIUM"]
    low    = [j for j in jobs if j.get("priority") == "LOW"]

    def score_ring(score: int, color: str) -> str:
        return f"""
        <div style="text-align:center;min-width:52px;">
          <div style="width:50px;height:50px;border-radius:50%;border:2.5px solid {color};
                      display:flex;align-items:center;justify-content:center;
                      font-size:15px;font-weight:600;color:{color};
                      background:{color}12;margin:0 auto;">
            {score}
          </div>
          <div style="font-size:10px;color:#94a3b8;margin-top:3px;text-transform:uppercase;
                      letter-spacing:0.5px;">score</div>
        </div>"""

    def breakdown(bd: dict, color: str) -> str:
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
                <div style="width:{pct}%;background:{color};border-radius:4px;height:5px;"></div>
              </div>
              <div style="font-size:10px;color:#64748b;width:28px;text-align:right;">
                {val}/{mx}
              </div>
            </div>"""
        return f'<div style="margin:10px 0;">{rows}</div>'

    def skills_html(skills: list, color: str) -> str:
        return "".join(
            f'<span style="background:{color}18;color:{color};padding:2px 9px;'
            f'border-radius:20px;font-size:11px;font-weight:500;'
            f'margin:2px;display:inline-block;">{s}</span>'
            for s in skills
        )

    def red_flags_html(flags: list) -> str:
        if not flags:
            return ""
        flagstr = " &nbsp;·&nbsp; ".join(f"⚠ {f}" for f in flags)
        return f'<div style="color:#f43f5e;font-size:11px;margin-top:6px;">{flagstr}</div>'

    def job_card(j: dict, color: str) -> str:
        bd    = j.get("score_breakdown", {})
        score = j.get("relevance_score", 0)
        pri   = j.get("priority", "LOW")
        pri_colors = {"HIGH": "#0ea5e9", "MEDIUM": "#8b5cf6", "LOW": "#64748b"}
        pri_c = pri_colors.get(pri, "#64748b")

        return f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:20px;margin-bottom:14px;border-left:4px solid {color};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="background:{pri_c}18;color:{pri_c};padding:2px 10px;
                             border-radius:20px;font-size:11px;font-weight:600;">
                  {pri}
                </span>
                <span style="font-size:11px;color:#94a3b8;">{j.get("source","")}</span>
              </div>
              <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:3px;">
                {j["title"]}
              </div>
              <div style="font-size:12px;color:#64748b;">
                🏢 <strong>{j["company"]}</strong>
                &nbsp;·&nbsp; 📍 {j.get("location","")}
                &nbsp;·&nbsp; 🗓 {str(j.get("posted",""))[:10]}
              </div>
            </div>
            {score_ring(score, color)}
          </div>
          {breakdown(bd, color)}
          <div style="font-size:13px;color:#475569;line-height:1.6;margin:8px 0;">
            {j.get("match_reason","")}
          </div>
          <div style="margin-bottom:6px;">
            {skills_html(j.get("key_match_skills",[]), color)}
          </div>
          {red_flags_html(j.get("red_flags",[]))}
          <a href="{j.get("url","#")}" target="_blank"
             style="display:inline-block;margin-top:12px;padding:8px 20px;
                    background:{color};color:#ffffff;border-radius:8px;
                    font-size:13px;font-weight:600;text-decoration:none;">
            Apply →
          </a>
        </div>"""

    def section(title: str, emoji: str, items: list, color: str) -> str:
        if not items:
            return ""
        cards = "".join(job_card(j, color) for j in items)
        return f"""
        <div style="margin-bottom:32px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <span style="font-size:20px;">{emoji}</span>
            <h2 style="margin:0;font-size:17px;font-weight:700;color:#0f172a;">{title}</h2>
            <span style="background:{color}18;color:{color};padding:3px 12px;
                         border-radius:20px;font-size:12px;font-weight:600;">
              {len(items)}
            </span>
          </div>
          {cards}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Job Intel — {report_date}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             max-width:740px;margin:0 auto;padding:28px;background:#f8fafc;color:#0f172a;">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
              border-radius:18px;padding:30px;margin-bottom:30px;color:#fff;">
    <div style="font-size:11px;letter-spacing:2px;color:#94a3b8;
                text-transform:uppercase;margin-bottom:6px;">Daily Digest</div>
    <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px;">
      🤖 Job Intel
    </div>
    <div style="font-size:13px;color:#94a3b8;margin-top:4px;">
      {report_date} &nbsp;·&nbsp; CrewAI + Groq LLaMA 3.1 70B &nbsp;·&nbsp;
      Intent-matched for Shashwat Mahajan
    </div>
    <div style="display:flex;gap:28px;margin-top:20px;flex-wrap:wrap;">
      <div>
        <div style="font-size:28px;font-weight:800;">{len(jobs)}</div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Total matches</div>
      </div>
      <div style="width:1px;background:#1e3a5f;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#38bdf8;">{len(high)}</div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">High priority</div>
      </div>
      <div style="width:1px;background:#1e3a5f;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#a78bfa;">{len(medium)}</div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Medium priority</div>
      </div>
      <div style="width:1px;background:#1e3a5f;"></div>
      <div>
        <div style="font-size:28px;font-weight:800;color:#64748b;">{len(low)}</div>
        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:1px;">Low priority</div>
      </div>
    </div>
  </div>

  {section("🔥 High Priority — Apply today", "🔥", high,   "#0ea5e9")}
  {section("⚡ Medium Priority — Apply this week", "⚡", medium, "#8b5cf6")}
  {section("📋 Low Priority — Apply if time permits", "📋", low, "#64748b")}

  <div style="text-align:center;font-size:11px;color:#94a3b8;
              margin-top:28px;padding-top:16px;
              border-top:1px solid #e2e8f0;">
    Job Intel Agent v2.0 &nbsp;·&nbsp; CrewAI + Groq LLaMA 3.1 70B &nbsp;·&nbsp; All free tools
  </div>
</body>
</html>"""


def send_email(html: str, cfg: dict, subject: str) -> None:
    """Send HTML email via Brevo SMTP (smtp-relay.brevo.com:587)."""
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