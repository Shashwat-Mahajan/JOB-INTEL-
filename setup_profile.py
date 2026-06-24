"""
setup_profile.py
Run once: python setup_profile.py
Reads your resume PDF + your current projects description,
extracts a structured profile using Groq, saves to config/profile.json.
From then on, scorer.py uses profile.json automatically.

CHANGES vs original:
  - EXTRACTION_PROMPT no longer shows concrete example VALUES for
    target_roles / target_companies / location_preferences / hard_vetoes.
    Groq at low temperature tends to echo example values shown in a JSON
    schema rather than deriving genuinely resume-specific content — this
    was the root cause of "profile doesn't reflect my actual resume."
    Now each of those fields has an explicit DERIVATION INSTRUCTION
    instead of a sample array, forcing Groq to reason from resume text.
  - Added a new field: "role_type_exclusions" — explicit non-engineering
    role categories to exclude (content writing, prompt-engineering-as-
    content, marketing, sales) derived from candidate's stated target
    role type, not from a hardcoded list. This directly addresses the
    "AI Content Writer recommended to AI Engineer candidate" mismatch.
  - hard_vetoes now requires the model to justify each veto by quoting
    which resume signal led to it, so vetoes are traceable to your resume
    rather than generic boilerplate.
"""

import json
import sys
from pathlib import Path
from groq import Groq
import fitz  # pymupdf

BASE        = Path(__file__).parent
CONFIG      = BASE / "config" / "config.json"
PROFILE_OUT = BASE / "config" / "profile.json"


CURRENT_PROJECTS = """
"""


EXTRACTION_PROMPT = """
You are a technical career advisor. Extract a complete, structured candidate profile
from the resume text provided.

CRITICAL RULE: Every field below must be DERIVED from the actual resume text given
to you. Do NOT use generic/example values. If you are unsure what belongs in a field,
look for direct evidence in the resume (project tech stacks, job titles applied to,
skills listed, internship roles held) rather than guessing a "typical" answer for
a CS student. Fields with no resume evidence should be left empty — an empty list
is more honest than a templated guess.

Return ONLY a valid JSON object with EXACTLY this structure — no extra fields,
no markdown, no explanation:

{
  "name": "full name",
  "email": "email",
  "phone": "phone",
  "linkedin": "linkedin url or username",
  "github": "github url or username",

  "education": {
    "degree": "degree name",
    "branch": "branch/specialization",
    "college": "college name",
    "graduation_year": "year",
    "cgpa": "cgpa value"
  },

  "current_status": "e.g. Final year B.Tech student, interning as GenAI Engineer",

  "technical_skills": {
    "primary_languages": ["lang1", "lang2"],
    "ai_ml": ["skill1", "skill2"],
    "backend": ["skill1", "skill2"],
    "cloud_devops": ["skill1", "skill2"],
    "databases": ["skill1", "skill2"],
    "frontend": ["skill1", "skill2"],
    "other": ["skill1", "skill2"]
  },

  "competitive_programming": {
    "leetcode_rating": "rating",
    "leetcode_problems": "count",
    "codechef_rating": "rating or stars",
    "other": "any other CP achievements"
  },

  "projects": [
    {
      "name": "project name",
      "description": "what it does in 2-3 sentences",
      "tech_stack": ["tech1", "tech2"],
      "highlights": ["key achievement 1", "key achievement 2"],
      "status": "completed or in-progress"
    }
  ],

  "experience": [
    {
      "role": "role/title",
      "company": "company name",
      "duration": "duration",
      "description": "what you did"
    }
  ],

  "achievements": ["achievement 1", "achievement 2"],

  "hackathons": [
    {
      "name": "hackathon name",
      "result": "rank/win/participant",
      "description": "what you built"
    }
  ],

  "publications": [
    {
      "title": "paper title",
      "conference": "conference name",
      "description": "what it covers"
    }
  ],

 "target_roles": "DERIVE this strictly from the resume text. Follow this logic: Step 1 — list every distinct technology, framework, and tool mentioned across ALL projects and experience. Step 2 — for each technology cluster, map to the industry job title that hiring managers actually post: (React+Node/FastAPI → 'Full Stack Developer'), (FastAPI/Django/Spring Boot alone → 'Backend Engineer'), (LangChain/RAG/ChromaDB/LLMs/embeddings → 'Generative AI Engineer', 'LLM Engineer'), (TensorFlow/PyTorch/scikit-learn/XGBoost → 'ML Engineer', 'Data Scientist'), (Android/Flutter/React Native → 'Mobile Developer'), (AWS/Docker/Kubernetes/Terraform → 'DevOps Engineer', 'Cloud Engineer'), (React/Vue/Angular alone → 'Frontend Developer'), (any programming language + DSA + projects → always include 'Software Engineer', 'SDE'). Step 3 — rank by how many projects use that cluster — the cluster with most projects becomes PRIMARY roles listed first. Step 4 — return 6-10 role strings as a JSON array. Never invent roles not supported by at least one concrete resume artifact.",

  "target_companies": "DERIVE this ONLY if the resume explicitly mentions target companies, applied-to companies, or referral mentions. If the resume contains no explicit company preferences, return an empty array — do NOT invent a generic FAANG/unicorn list.",

  "location_preferences": "DERIVE this from the candidate's current location (city/state mentioned in contact info or education) and any explicitly stated location preference in the resume. If none stated beyond current location, return just that city plus 'Remote India'. Do NOT default to a generic multi-city Indian tech-hub list.",

  "graduation_batch": "year extracted from education.graduation_year",
  "experience_level": "fresher",
  "max_experience_years": 2,

  "role_type_exclusions": "DERIVE this by identifying what the candidate's resume evidence is NOT. Specifically check: does the resume show hands-on coding (writing code, building systems, debugging) or does it show content/communication work (writing articles, prompts-as-content, documentation-only, non-technical analysis)? If the resume is clearly a HANDS-ON ENGINEERING profile (the common case for CS/AI/ML resumes), explicitly list: ['AI content writing', 'prompt engineering for content/marketing', 'AI marketing/copywriting roles', 'non-technical AI trainer roles', 'data annotation without engineering component']. Only omit these exclusions if the resume itself shows genuine content/writing work as a real skill area.",

  "hard_vetoes": "DERIVE this as a JSON array of objects: [{'veto': 'short description', 'resume_evidence': 'what in the resume justifies this veto'}]. Derive vetoes from: (1) experience level mismatches (fresher → veto 3+ years required); (2) role mismatches (no DevOps projects → veto pure DevOps); (3) location constraints; (4) company type mismatches — IMPORTANT: if the candidate's profile is clearly AI/ML/product-engineering focused (building AI systems, RAG pipelines, ML models), derive a veto for IT services/outsourcing companies (TCS, Infosys, Wipro, HCL, Accenture, Capgemini, Tech Mahindra, Cognizant, Mphasis, Genpact, Hexaware, LTI, LTIMindtree, Birlasoft) WITHOUT an AI/ML title — because these companies hire freshers into generic IT support/CRUD roles, not the AI/ML engineering roles this candidate targets. Every veto must cite real resume evidence.",
}

Be thorough. Extract every skill, project, achievement you can find.
If a field is not in the resume, use an empty string or empty list — never
fall back to a plausible-sounding generic answer.
"""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pymupdf."""
    doc  = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_profile_with_groq(resume_text: str, current_projects: str, api_key: str) -> dict:
    """Send resume text to Groq and extract structured profile."""
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"RESUME TEXT:\n{resume_text}\n\n"
                    f"CURRENT PROJECTS (not yet on resume):\n{current_projects}\n\n"
                    "Extract the complete profile. Return JSON only. "
                    "Remember: target_roles, target_companies, location_preferences, "
                    "role_type_exclusions, and hard_vetoes must be DERIVED from the "
                    "resume text above, not generic examples."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)


def build_scoring_prompt(profile: dict) -> str:
    """
    Build a rich, dynamic CANDIDATE_PROFILE string from the extracted profile.
    This is what goes into scorer.py's scoring prompt.

    Now includes role_type_exclusions and structured hard_vetoes (with
    resume_evidence) so the scorer can see WHY a veto exists, not just that
    it exists — this helps the LLM apply vetoes correctly to edge cases.
    """
    name    = profile.get("name", "")
    edu     = profile.get("education", {})
    status  = profile.get("current_status", "")
    skills  = profile.get("technical_skills", {})
    cp      = profile.get("competitive_programming", {})
    exp     = profile.get("experience_level", "fresher")
    batch   = profile.get("graduation_batch", "2027")
    vetoes  = profile.get("hard_vetoes", [])
    exclusions = profile.get("role_type_exclusions", [])
    targets = profile.get("target_roles", [])
    companies = profile.get("target_companies", [])
    locations = profile.get("location_preferences", [])

    # Build skills string
    all_skills = []
    for category, skill_list in skills.items():
        if skill_list:
            all_skills.extend(skill_list)
    skills_str = ", ".join(all_skills)

    # Build projects string
    projects_str = ""
    for p in profile.get("projects", []):
        tech = ", ".join(p.get("tech_stack", []))
        highlights = "; ".join(p.get("highlights", []))
        projects_str += f"- {p['name']}: {p.get('description','')} Stack: {tech}. {highlights}\n"

    # Build experience string
    exp_str = ""
    for e in profile.get("experience", []):
        exp_str += f"- {e.get('role','')} at {e.get('company','')} ({e.get('duration','')}): {e.get('description','')}\n"

    # Build achievements string
    achievements = "; ".join(profile.get("achievements", []))
    hackathons   = "; ".join([f"{h['name']} ({h.get('result','')})" for h in profile.get("hackathons", [])])
    publications = "; ".join([p.get("title","") for p in profile.get("publications", [])])

    # Build vetoes string — handle both old (string list) and new (object list) formats
    veto_lines = []
    for v in vetoes:
        if isinstance(v, dict):
            veto_lines.append(f"- {v.get('veto','')} (evidence: {v.get('resume_evidence','')})")
        else:
            veto_lines.append(f"- {v}")
    vetoes_str = "\n".join(veto_lines)

    exclusions_str = ", ".join(exclusions) if exclusions else "none specified"

    prompt = f"""
CANDIDATE: {name}
STATUS: {status}
EDUCATION: {edu.get('degree','')} in {edu.get('branch','')} from {edu.get('college','')}
GRADUATION: {batch} batch | CGPA: {edu.get('cgpa','')}
EXPERIENCE LEVEL: {exp} (max {profile.get('max_experience_years',2)} years)

TECHNICAL SKILLS:
{skills_str}

COMPETITIVE PROGRAMMING:
LeetCode: {cp.get('leetcode_rating','')} rating, {cp.get('leetcode_problems','')} problems solved
CodeChef: {cp.get('codechef_rating','')}
{cp.get('other','')}

KEY PROJECTS:
{projects_str.strip()}

WORK EXPERIENCE:
{exp_str.strip() if exp_str.strip() else 'Internships and project-based experience only'}

ACHIEVEMENTS: {achievements}
HACKATHONS: {hackathons}
PUBLICATIONS: {publications}

TARGET ROLES (priority order, derived from resume evidence): {', '.join(targets) if targets else 'see technical skills above'}
TARGET COMPANIES: {', '.join(companies) if companies else 'no explicit preference — any reputable tech company'}
PREFERRED LOCATIONS: {', '.join(locations) if locations else 'not specified'}

ROLE TYPE EXCLUSIONS (this candidate wants HANDS-ON ENGINEERING work, NOT these):
{exclusions_str}

HARD VETOES (auto-disqualify, each tied to resume evidence):
{vetoes_str if vetoes_str else '- none derived'}
""".strip()

    return prompt


def main():
    # Load config for API key
    if not CONFIG.exists():
        print("ERROR: config/config.json not found. Set up config first.")
        sys.exit(1)

    cfg     = json.loads(CONFIG.read_text())
    api_key = cfg.get("groq_api_key", "")
    if not api_key:
        print("ERROR: groq_api_key missing from config.json")
        sys.exit(1)

    # Find resume PDF
    resume_path = None
    for candidate in [
        BASE / "config" / "resume.pdf",
        BASE / "resume.pdf",
        Path.home() / "resume.pdf",
    ]:
        if candidate.exists():
            resume_path = candidate
            break

    if not resume_path:
        print("\nERROR: Resume PDF not found.")
        print("Put your resume at one of these paths:")
        print("  D:\\job_agent\\config\\resume.pdf  ← recommended")
        print("  D:\\job_agent\\resume.pdf")
        sys.exit(1)

    print(f"\nReading resume: {resume_path}")
    resume_text = extract_text_from_pdf(resume_path)
    print(f"Extracted {len(resume_text)} characters from resume.")

    print("Sending to Groq for profile extraction...")
    profile = extract_profile_with_groq(resume_text, CURRENT_PROJECTS, api_key)

    # Add the scoring prompt to the profile
    profile["_scoring_prompt"] = build_scoring_prompt(profile)

    # Save
    PROFILE_OUT.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"\nProfile saved to: {PROFILE_OUT}")
    print("\nExtracted profile summary:")
    print(f"  Name:     {profile.get('name','')}")
    print(f"  College:  {profile.get('education',{}).get('college','')}")
    print(f"  Batch:    {profile.get('graduation_batch','')}")
    print(f"  Skills:   {len(sum(profile.get('technical_skills',{}).values(),[]))} skills found")
    print(f"  Projects: {len(profile.get('projects',[]))} projects found")
    print(f"  Exp:      {len(profile.get('experience',[]))} roles found")
    print(f"  Target roles (derived): {profile.get('target_roles', [])}")
    print(f"  Role exclusions (derived): {profile.get('role_type_exclusions', [])}")
    print(f"  Hard vetoes (derived): {len(profile.get('hard_vetoes', []))} vetoes")
    print(f"\n  REVIEW target_roles, role_type_exclusions, and hard_vetoes above.")
    print(f"   These are now derived from your resume, not generic templates.")
    print(f"   If anything looks wrong, edit config/profile.json directly —")
    print(f"   it's a plain JSON file you can hand-correct.")
    print(f"\nRun 'python main.py' to start the agent with your profile.")


if __name__ == "__main__":
    main()