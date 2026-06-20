import json
from llama_index.core import Document

def chunk(data):
    docs = []

    for candidate in data:
        id = candidate['candidate_id']
        profile = candidate['profile']

        summary = f"Headline: {profile.get('headline')}\nSummary: {profile.get('summary')}"

        doc_profile = Document(
            text=summary,
            metadata={
                "candidate_id": id,
                "chunk_type": "profile_summary",
                "total_experience_years": profile.get("years_of_experience"),
                "current_title": profile.get("current_title"),
                "location": profile.get("location")
            },

            excluded_embed_metadata_keys=["candidate_id", "chunk_type"]
        )

        docs.append(doc_profile)

        skills = ", ".join([f"{s['name']} (Proficiency: {s['proficiency']}, Used for: {s['duration_months']} months)" for s in candidate.get("skills", [])])
        skills_text = f"Technical and Professional Skills: {skills}"

        doc_skills = Document(
                text=skills_text,
                metadata={
                    "candidate_id": id,
                    "chunk_type": "skills"
                },
                excluded_embed_metadata_keys=["candidate_id", "chunk_type"]
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
                        "candidate_id": id,
                        "chunk_type": "career_history",
                        "company": job.get("company"),
                        "duration_months": job.get("duration_months"),
                        "is_current": job.get("is_current")
                    },
                    excluded_embed_metadata_keys=["candidate_id", "chunk_type"]
                )
            docs.append(doc_job)
    
    return docs
