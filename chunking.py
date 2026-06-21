import json
from llama_index.core import Document

def chunk(data):
    docs = []

    for candidate in data:
        id = candidate['candidate_id']
        profile = candidate['profile']
        signals = candidate.get('redrob_signals', {})

        candidate_meta = {
            "candidate_id": id,
            "total_experience_years": profile.get("years_of_experience"),
            "current_title": profile.get("current_title"),
            "location": profile.get("location"),
            "current_industry": profile.get("current_industry"),
            "open_to_work_flag": str(signals.get("open_to_work_flag")).lower(),
            "willing_to_relocate": str(signals.get("willing_to_relocate")).lower(),
        }
        embed_exclude = list(candidate_meta) + ["chunk_type"]

        summary = f"Headline: {profile.get('headline')}\nSummary: {profile.get('summary')}"

        doc_profile = Document(
            text=summary,
            metadata={**candidate_meta, "chunk_type": "profile_summary"},
            excluded_embed_metadata_keys=embed_exclude
        )
        docs.append(doc_profile)

        skill_list = candidate.get("skills", [])
        skills = ", ".join([f"{s['name']} (Proficiency: {s['proficiency']}, Used for: {s['duration_months']} months)" for s in skill_list])
        skills_text = f"Technical and Professional Skills: {skills}"

        doc_skills = Document(
                text=skills_text,
                metadata={
                    **candidate_meta, "chunk_type": "skills", "skill_names": [s["name"].lower() for s in skill_list]},
                excluded_embed_metadata_keys=embed_exclude + ["skill_names"]
            )
        docs.append(doc_skills)

        for job in candidate.get("career_history", []):
            job_text = (
                f"Role: {job.get('title')} at {job.get('company')}\n"
                f"Industry: {job.get('industry')}\n"
                f"Description: {job.get('description')}"
            )

            doc_job = Document(
                    text=job_text,
                    metadata={
                        **candidate_meta,
                        "chunk_type": "career_history",
                        "company": job.get("company"),
                        "duration_months": job.get("duration_months"),
                        "is_current": str(job.get("is_current")).lower()
                    },
                    excluded_embed_metadata_keys=embed_exclude + ["company", "duration_months", "is_current"]
                )
            docs.append(doc_job)

    return docs