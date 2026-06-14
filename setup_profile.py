"""
setup_profile.py
Run once: python setup_profile.py
Reads your resume PDF + your current projects description,
extracts a structured profile using Groq, saves to config/profile.json.
From then on, scorer.py uses profile.json automatically.
"""

import json
import sys
from pathlib import Path
from groq import Groq
import fitz  # pymupdf

BASE       = Path(__file__).parent
CONFIG     = BASE / "config" / "config.json"
PROFILE_OUT = BASE / "config" / "profile.json"


CURRENT_PROJECTS = """
PROJECT I AM CURRENTLY WORKING ON:

Job Intel Agent (this system):
- Multi-agent job discovery system using CrewAI + Groq + Ollama
- Scout agent fetches jobs from 15 sources (LinkedIn, Naukri, Amazon, Google, etc.)
- Analyst agent scores jobs using Groq llama-3.3-70b-versatile with intent matching
- Verifier agent does second-pass accuracy check on HIGH priority jobs
- Reporter agent builds HTML digest and sends via Brevo SMTP
- Fully automated via GitHub Actions daily cron at 8 AM IST
- Stack: Python, CrewAI 1.x, Groq API, Ollama, BeautifulSoup, GitHub Actions

(Add any other current projects here before running this script)
"""


EXTRACTION_PROMPT = """
You are a technical career advisor. Extract a complete, structured candidate profile
from the resume text provided.

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

  "target_roles": [
    "GenAI Engineer",
    "ML Engineer",
    "Software Engineer - AI",
    "Backend Engineer",
    "Data Engineer"
  ],

  "target_companies": [
    "Amazon", "Microsoft", "Google", "Flipkart", "Swiggy",
    "Meesho", "Razorpay", "PhonePe", "Atlassian", "Adobe",
    "Walmart Global Tech", "Oracle", "JPMorgan", "Uber",
    "Samsung R&D", "Nutanix", "Qualcomm", "NVIDIA"
  ],

  "location_preferences": ["Bengaluru", "Pune", "Hyderabad", "Mumbai", "NCR", "Remote India"],

  "graduation_batch": "2027",
  "experience_level": "fresher",
  "max_experience_years": 2,

  "hard_vetoes": [
    "IT outsourcing firms (TCS/Infosys/Wipro/Accenture/Cognizant/HCL) unless title explicitly says AI/ML/GenAI",
    "Roles requiring 3+ years experience without fresher mention",
    "Pure DevOps, Network Engineer, Hardware, Sales, non-tech roles",
    "Unpaid internships",
    "Outside India unless fully remote"
  ]
}

Be thorough. Extract every skill, project, achievement you can find.
If a field is not in the resume, use an empty string or empty list.
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
                    "Extract the complete profile. Return JSON only."
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
    """
    name    = profile.get("name", "")
    edu     = profile.get("education", {})
    status  = profile.get("current_status", "")
    skills  = profile.get("technical_skills", {})
    cp      = profile.get("competitive_programming", {})
    exp     = profile.get("experience_level", "fresher")
    batch   = profile.get("graduation_batch", "2027")
    vetoes  = profile.get("hard_vetoes", [])
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

TARGET ROLES (priority order): {', '.join(targets)}
TARGET COMPANIES: {', '.join(companies)}
PREFERRED LOCATIONS: {', '.join(locations)}

HARD VETOES (auto-disqualify):
{chr(10).join('- ' + v for v in vetoes)}
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
    print(f"\nRun 'python main.py' to start the agent with your profile.")


if __name__ == "__main__":
    main()