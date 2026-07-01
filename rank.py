import csv
import json
import pickle
import sys
from datetime import date, datetime

from retrival import retrieve, client

CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree", "tech mahindra"}

TOP_LOCATIONS = ["pune", "noida"]
PREFERRED_LOCATIONS = ["hyderabad", "mumbai", "delhi"]

TECHNICAL_TITLE_KEYWORDS = {
    "engineer", "scientist", "developer", "architect", "researcher",
    "ml", "ai", "data", "nlp", "backend", "software", "technical",
    "platform", "infrastructure", "lead"
}

NON_TECHNICAL_ENGINEERING = {
    "mechanical", "civil", "chemical", "structural", "aerospace", "electrical", "industrial", "manufacturing", "process", "materials"
}

PROFICIENCY_WEIGHTS = {"expert": 0.9, "advanced": 0.9, "intermediate": 0.8, "beginner": 0.75}

WEIGHTS = {
    "rrf": 0.30,
    "required_skills": 0.20,
    "preferred_skills": 0.10,
    "experience": 0.10,
    "behavioral": 0.25,
    "location": 0.05,
}

def is_consulting_only(career_history):
    if not career_history:
        return False
    
    return all(any(firm in job.get("company", "").lower() for firm in CONSULTING_FIRMS) for job in career_history)

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
        recency = 0.3

    response = float(signals.get("recruiter_response_rate", 0.5))
    interview = float(signals.get("interview_completion_rate", 0.5))

    notice_days = int(signals.get("notice_period_days", 30))
    if notice_days <= preferred_notice_days:
        notice = 1.0
    elif notice_days <= 60:
        notice = 0.25
    else:
        notice = -0.4

    raw_github = float(signals.get("github_activity_score", -1))
    github = 0.0 if raw_github < 0 else min(raw_github / 50.0, 1.0)

    open_flag = 1.0 if signals.get("open_to_work_flag", True) else 0.20

    return 0.25 * recency + 0.20 * response + 0.20 * notice + 0.15 * interview + 0.10 * github + 0.10 * open_flag

    # return 0.30 * recency + 0.20 * response + 0.30 * notice + 0.10 * interview + 0.10 * github

def skill_coverage(candidate_skills_raw, target_skills, assessment_scores):
    if not target_skills:
        return 0.0

    skills_dict = {s["name"].lower(): PROFICIENCY_WEIGHTS.get(s["proficiency"], 0.5) for s in candidate_skills_raw}
    assessed = {k.lower(): v / 100.0 for k, v in assessment_scores.items()}

    total = 0.0
    for skill in target_skills:
        sl = skill.lower() 
        if sl in assessed:
            total += assessed[sl]
        elif sl in skills_dict:
            total += skills_dict[sl]

    return total / len(target_skills)

def is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education",{})
    skills = candidate.get("skills", [])
    yoe_months = float(profile.get("years_of_experience", 0)) * 12
    title = (profile.get("current_title") or "").lower()

    if any(prefix in title for prefix in NON_TECHNICAL_ENGINEERING):
        return True

    if not any(kw in title for kw in TECHNICAL_TITLE_KEYWORDS):
        return True

    if yoe_months > 0:
        for job in career:
            if job.get("duration_months", 0) > yoe_months + 6:
                return True

        total_career_months = sum(j.get("duration_months", 0) for j in career)
        if total_career_months > yoe_months * 1.5:
            return True
        
        for skill in skills:
            if skill.get("duration_months", 0) > yoe_months + 3:
                return True

    # today = date.today()
    # for skill in skills:
    #     launch = TOOL_LAUNCH_DATES.get(skill["name"].lower())
    #     if launch:
    #         max_months = (today - launch).days / 30.5
    #         if skill.get("duration_months", 0) > max_months + 3:
    #             return True

    return False

def is_keyword_stuffer(candidate_skills_raw, required, assessment_scores):
    if assessment_scores:
        return 1.0

    skills_by_name = {s["name"].lower(): s for s in candidate_skills_raw}
    matched = [skills_by_name[r.lower()] for r in required if r.lower() in skills_by_name]

    if len(matched) < 3:
        return 1.0

    avg_duration = sum(s.get("duration_months", 0) for s in matched) / len(matched)
    if avg_duration < 6:
        return 0.65

    return 1.0

def build_reasoning(candidate, parsed_jd, years, req_matches, pref_matches, req_cov, rank=None):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    raw_skills = candidate.get("skills", [])
 
    title = profile.get("current_title") or "professional"
    company = profile.get("current_company") or ""
    country = (profile.get("country") or "").lower()
    location = profile.get("location") or ""
    notice_days = int(signals.get("notice_period_days", 90))
    response_rate = float(signals.get("recruiter_response_rate", 0.5))
    github = float(signals.get("github_activity_score", -1))
    assessment_scores = {k.lower(): v for k, v in signals.get("skill_assessment_scores", {}).items()}
    min_exp, max_exp = parsed_jd["min_experience"], parsed_jd["max_experience"]
    skills_by_name = {s["name"].lower(): s for s in raw_skills}
    n_req = len(parsed_jd["required_skills"])
 
    def describe_skill(name):
        sl = name.lower()
        if sl in assessment_scores:
            return f"{name} ({int(assessment_scores[sl])}% assessed)"
        sk = skills_by_name.get(sl)
        if sk:
            dur = sk.get("duration_months", 0)
            prof = sk.get("proficiency", "")
            return f"{name} ({prof}, {dur}mo)" if dur >= 12 else f"{name} ({prof})"
        return name
 
    current_job = next((j for j in career if j.get("is_current")), career[0] if career else None)
    current_industry = (current_job.get("industry") or "") if current_job else ""
    consulting_jobs = [j for j in career if any(f in (j.get("company") or "").lower() for f in CONSULTING_FIRMS)]
    product_jobs = [j for j in career if j not in consulting_jobs]
    is_mostly_consulting = len(consulting_jobs) >= len(career) * 0.7 and len(career) > 1
 
    req_matched_lower = {s.lower() for s in req_matches}
    missing_req = [s for s in parsed_jd["required_skills"] if s.lower() not in req_matched_lower]
    identity = f"{title} at {company}" if company else title
 
    if rank is not None and rank <= 25:
        tone = "positive"
    elif rank is not None and rank <= 60:
        tone = "balanced"
    else:
        tone = "critical"
 
    if tone == "positive":
        top_skills = ", ".join(describe_skill(s) for s in req_matches[:3])
        if req_matches:
            lead = f"{identity}, {years:.1f} yrs. required skills — {top_skills}."
        else:
            lead = f"{identity}, {years:.1f} yrs."
        if pref_matches:
            lead = lead.rstrip(".") + f"; also brings {', '.join(pref_matches[:2])} from preferred list."
 
    elif tone == "balanced":
        have = ", ".join(describe_skill(s) for s in req_matches[:2])
        if missing_req:
            miss = ", ".join(missing_req[:2])
            lead = f"{identity}, {years:.1f} yrs. Has {have} but no coverage of {miss}. {len(req_matches)}/{n_req} required skills."
        else:
            lead = f"{identity}, {years:.1f} yrs. {len(req_matches)}/{n_req} required skills — {have}."
 
    else:
        miss = ", ".join(missing_req[:3])
        lead = (
            f"{identity}, {years:.1f} yrs. Only {len(req_matches)}/{n_req} required skills covered; "
            f"missing {miss}."
        )
 
    if years < min_exp:
        exp_note = f"{years:.1f} yrs is below the {min_exp}-{max_exp} yr target."
    elif years > max_exp:
        exp_note = f"{years:.1f} yrs exceeds the target range of {max_exp} yrs."
    else:
        exp_note = ""
 
    if is_mostly_consulting:
        career_note = (
            f"Mostly consulting background; some product experience at {product_jobs[0]['company']}."
            if product_jobs else "Consulting-only background across all roles."
        )
    elif current_industry and current_industry.lower() not in {"it services", "information technology"}:
        career_note = f"{current_industry} domain background."
    else:
        career_note = ""
 
    concerns = []
    if notice_days > parsed_jd["preferred_notice_days"]:
        concerns.append(f"{notice_days}-day notice period")
    if response_rate < 0.4:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if country != "india":
        if not signals.get("willing_to_relocate"):
            concerns.append(f"based in {location or country}, not open to relocation")
        else:
            concerns.append("based outside India, willing to relocate")
    if 0 <= github < 10 and req_cov >= 0.5:
        concerns.append(f"low GitHub activity ({github:.0f}/100)")
    concern_str = "Concern: " + ", ".join(concerns) + "." if concerns else ""
 
    positives = []
    if github >= 40:
        positives.append(f"GitHub score {github:.0f}/100")
    if response_rate >= 0.8 and notice_days <= parsed_jd["preferred_notice_days"]:
        positives.append(f"{response_rate:.0%} response rate, {notice_days}-day notice")
    elif notice_days <= 15 and req_cov >= 0.5:
        positives.append("immediately available")
    positive_str = "Strong signals: " + ", ".join(positives) + "." if positives else ""
 
    return " ".join(p for p in [lead, exp_note, career_note, concern_str, positive_str] if p).strip()

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
        
        # if not signals.get("open_to_work_flag", True):
        #     continue
        if is_consulting_only(c.get("career_history", [])):
            continue

        raw_skills = c.get("skills", [])
        assessment_scores = signals.get("skill_assessment_scores", {})

        req_cov = skill_coverage(raw_skills, required, assessment_scores)

        if req_cov == 0.0:
            continue

        if is_honeypot(c):
            continue
        
        pref_cov = skill_coverage(raw_skills, preferred, assessment_scores)

        skills_set = {s["name"].lower() for s in raw_skills}
        req_matches = [s for s in required if s.lower() in skills_set]
        pref_matches = [s for s in preferred if s.lower() in skills_set]

        years = get_years_experience(c)
        norm_rrf = r["rrf_score"] / rrf_max
        willing = signals.get("willing_to_relocate", False)

        final = (
            WEIGHTS["rrf"] * norm_rrf
            + WEIGHTS["required_skills"] * req_cov
            + WEIGHTS["preferred_skills"] * pref_cov
            + WEIGHTS["experience"] * experience_score(years, parsed_jd["min_experience"], parsed_jd["max_experience"])
            + WEIGHTS["location"] * location_score(c.get("profile", {}), willing)
            + WEIGHTS["behavioral"] * behavioral_score(signals, parsed_jd["preferred_notice_days"])
        )
            
        final *= is_keyword_stuffer(raw_skills, required, assessment_scores)

        results.append({
            "candidate_id": cid,
            "score": final,
            "_c": c,
            "_req_matches": req_matches,
            "_pref_matches": pref_matches,
            "_req_cov": req_cov,
            "_years": years,
        })

    if results:
        max_s = max(r["score"] for r in results)
        min_s = min(r["score"] for r in results)
        rng = max_s - min_s
        for r in results:
            r["score"] = round((r["score"] - min_s) / rng if rng > 0 else 1.0, 6)

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    results = results[:top_k]

    for i, row in enumerate(results, start=1):
        row["rank"] = i
        row["reasoning"] = build_reasoning(
            row.pop("_c"),
            parsed_jd,
            row.pop("_years"),
            row.pop("_req_matches"),
            row.pop("_pref_matches"),
            row.pop("_req_cov"),
            rank=i,
        )

    return results

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

    with open("artifacts/jd_parsed.pkl", "rb") as f:
        parsed_jd = pickle.load(f)

    print("Required skills:", parsed_jd["required_skills"])
    print("Preferred skills:", parsed_jd["preferred_skills"])
    print(f"Experience: {parsed_jd['min_experience']}-{parsed_jd['max_experience']} years")
    print(f"Notice: ≤{parsed_jd['preferred_notice_days']} days preferred\n")

    retrieved = retrieve(top_k=1500,raw_top_k=5000,weights={"profile_summary": 0.65, "skills": 0.65, "career_history": 0.75, "graph": 0.9})
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