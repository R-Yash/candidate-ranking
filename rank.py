import csv
import json
import sys
from datetime import date, datetime

from docx import Document

from jd_parsing import parse_jd
from retrival import retrieve, client

CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree", "tech mahindra"}
TOP_LOCATIONS = ["pune", "noida"]
PREFERRED_LOCATIONS = ["hyderabad", "mumbai", "delhi"]
TECHNICAL_TITLE_KEYWORDS = {
    "engineer", "scientist", "developer", "architect", "researcher",
    "ml", "ai", "data", "nlp", "backend", "software", "technical",
    "platform", "infrastructure", "lead"
}

PROFICIENCY_WEIGHTS = {"expert": 1.0, "advanced": 0.9, "intermediate": 0.65, "beginner": 0.35}

WEIGHTS = {
    "rrf": 0.40,
    "required_skills": 0.20,
    "preferred_skills": 0.08,
    "experience": 0.10,
    "behavioral": 0.15,
    "location": 0.07,
}
HONEYPOT_MULTIPLIER = 0.1
RETRIEVE_TOP_K = 500

def is_consulting_only(career_history):
    if not career_history:
        return False
    
    return all(any(firm in co.get("company", "").lower() for firm in CONSULTING_FIRMS) for co in career_history)

def get_years_experience(candidate):
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if yoe:
        return float(yoe)
    
    return sum(c.get("duration_months", 0) for c in candidate.get("career_history", [])) / 12.0

def experience_score(years, min_exp, max_exp):
    if min_exp <= years <= max_exp:
        return 1.0
    if years < min_exp:
        return max(0.0, 1.0 - (min_exp - years) * 0.15)
    
    return max(0.0, 1.0 - (years - max_exp) * 0.10)

def location_score(profile, willing_to_relocate):
    country = (profile.get("country") or "").lower()
    loc = (profile.get("location") or "").lower()

    if country != "india":
        return 0.3 if willing_to_relocate else 0.1

    if any(city in loc for city in TOP_LOCATIONS):
        return 1.0
    if any(city in loc for city in PREFERRED_LOCATIONS):
        return 0.7

    return 0.5 if willing_to_relocate else 0.25

def behavioral_score(signals, preferred_notice_days):
    last_active = signals.get("last_active_date")
    try:
        days_ago = (date.today() - datetime.strptime(last_active, "%Y-%m-%d").date()).days
        if days_ago < 7:    
            recency = 1.0
        elif days_ago < 30: 
            recency = 0.9
        elif days_ago < 90: 
            recency = 0.7
        elif days_ago < 180: 
            recency = 0.5
        else:               
            recency = 0.2
    except (TypeError, ValueError):
        print("ERROR")
        recency = 0.3

    response = float(signals.get("recruiter_response_rate", 0.5))
    interview = float(signals.get("interview_completion_rate", 0.5))

    notice_days = int(signals.get("notice_period_days", 90))
    if notice_days <= preferred_notice_days: 
        notice = 1.0
    elif notice_days <= 60:                  
        notice = 0.6
    else:
         notice = 0.2

    raw_github = float(signals.get("github_activity_score", -1))
    github = 0.0 if raw_github < 0 else min(raw_github / 50.0, 1.0)

    return 0.30 * recency + 0.25 * response + 0.20 * notice + 0.15 * interview + 0.10 * github


def skill_coverage(candidate_skills_raw, target_skills, assessment_scores):
    if not target_skills:
        return 0.0

    skills_dict = {
        s["name"].lower(): PROFICIENCY_WEIGHTS.get(s["proficiency"], 0.5)
        for s in candidate_skills_raw
    }
    assessed = {k.lower(): v / 100.0 for k, v in assessment_scores.items()}

    total = 0.0
    for skill in target_skills:
        if skill in assessed:
            total += assessed[skill]          
        elif skill in skills_dict:
            total += skills_dict[skill]       

    return total / len(target_skills)

def is_honeypot(candidate, req_skill_coverage):
    if req_skill_coverage < 0.70:
        return False
    title = (candidate.get("profile", {}).get("current_title") or "").lower()
    return not any(kw in title for kw in TECHNICAL_TITLE_KEYWORDS)


def build_reasoning(candidate, parsed_jd, years, req_matches, pref_matches):
    profile = candidate.get("profile", {})
    title = profile.get("current_title") or "Unknown title" 
    location = profile.get("location") or ""
    country = profile.get("country") or ""
    signals = candidate.get("redrob_signals", {})
    notice_days = int(signals.get("notice_period_days", 90))
    response_rate = float(signals.get("recruiter_response_rate", 0.5))
    min_exp, max_exp = parsed_jd["min_experience"], parsed_jd["max_experience"]

    parts = [f"{title}, {years:.1f} yrs exp"]
    if req_matches:
        parts.append(f"{len(req_matches)} required skills matched ({', '.join(req_matches[:3])})")
    if pref_matches:
        parts.append(f"{len(pref_matches)} preferred ({pref_matches[0]})")
    loc_lower = location.lower()
    if any(city in loc_lower for city in TOP_LOCATIONS + PREFERRED_LOCATIONS):
        parts.append(f"based in {location}")
    elif country and country.lower() != "india":
        parts.append(f"based in {location}, {country}")
    sentence1 = "; ".join(parts) + "."

    concerns = []
    if years < min_exp:
        concerns.append(f"under target experience range ({min_exp}–{max_exp} yrs)")
    elif years > max_exp:
        concerns.append(f"above target range ({min_exp}–{max_exp} yrs)")
    if notice_days > parsed_jd["preferred_notice_days"]:
        concerns.append(f"{notice_days}-day notice")
    if response_rate < 0.4:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if country.lower() != "india" and not signals.get("willing_to_relocate"):
        concerns.append("not willing to relocate, based outside India")

    if concerns:
        sentence2 = "Concern: " + ", ".join(concerns) + "."
    elif response_rate >= 0.8 and notice_days <= parsed_jd["preferred_notice_days"]:
        sentence2 = f"Strong signals: {response_rate:.0%} response rate, {notice_days}-day notice."
    else:
        sentence2 = ""

    return (sentence1 + (" " + sentence2 if sentence2 else "")).strip()

def load_candidates_for_ids(path, candidate_ids):
    target = set(candidate_ids)
    candidates = {}
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in target:
                candidates[c["candidate_id"]] = c
    return candidates

def score_all(retrieved, candidates_by_id, parsed_jd, top_k=100):
    required = parsed_jd["required_skills"]
    preferred = parsed_jd["preferred_skills"]
    rrf_max = max((r["rrf_score"] for r in retrieved), default=1.0)

    results = []
    for r in retrieved:
        cid = r["candidate_id"]
        c = candidates_by_id.get(cid)
        if not c:
            continue

        signals = c.get("redrob_signals", {})

        if not signals.get("open_to_work_flag", True):
            continue
        if is_consulting_only(c.get("career_history", [])):
            continue

        raw_skills = c.get("skills", [])
        assessment_scores = signals.get("skill_assessment_scores", {})

        req_cov  = skill_coverage(raw_skills, required, assessment_scores)
        pref_cov = skill_coverage(raw_skills, preferred, assessment_scores)

        skills_set = {s["name"].lower() for s in raw_skills}
        req_matches = [s for s in required  if s in skills_set]
        pref_matches = [s for s in preferred if s in skills_set]

        years = get_years_experience(c)
        norm_rrf = r["rrf_score"] / rrf_max
        willing = signals.get("willing_to_relocate", False)

        final = (
            WEIGHTS["rrf"] * norm_rrf +
            WEIGHTS["required_skills"] * req_cov +
            WEIGHTS["preferred_skills"] * pref_cov +
            WEIGHTS["experience"] * experience_score(years, parsed_jd["min_experience"], parsed_jd["max_experience"]) +
            WEIGHTS["location"] * location_score(c.get("profile", {}), willing) +
            WEIGHTS["behavioral"] * behavioral_score(signals, parsed_jd["preferred_notice_days"])
        )

        if is_honeypot(c, req_cov):
            final *= HONEYPOT_MULTIPLIER

        results.append({
            "candidate_id": cid,
            "score": round(final, 6),
            "reasoning": build_reasoning(c, parsed_jd, years, req_matches, pref_matches),
        })

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    results = results[:top_k]

    for i, row in enumerate(results, start=1):
        row["rank"] = i

    return results

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

    jd_text = "\n".join(p.text for p in Document("data/job_description.docx").paragraphs)

    parsed_jd = parse_jd(jd_text)
    print("Required skills:", parsed_jd["required_skills"])
    print("Preferred skills:", parsed_jd["preferred_skills"])
    print(f"Experience: {parsed_jd['min_experience']}–{parsed_jd['max_experience']} years")
    print(f"Notice: ≤{parsed_jd['preferred_notice_days']} days preferred\n")

    retrieved = retrieve(
        jd_text,
        top_k=RETRIEVE_TOP_K,
        weights={"profile_summary": 0.5, "skills": 0.65, "career_history": 0.75},
    )
    
    print(f"Retrieved {len(retrieved)} candidates\n")

    candidates = load_candidates_for_ids(
        "data/candidates.jsonl",
        [r["candidate_id"] for r in retrieved],
    )

    top_100 = score_all(retrieved, candidates, parsed_jd)

    COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top_100)

    client.close()
    print(f"Wrote {len(top_100)} candidates → {out_file}")