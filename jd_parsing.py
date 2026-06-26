import json
import os
import pickle

from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def get_vocabulary(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_jd(jd_text: str) -> dict:
    client = genai.Client()
    vocab = get_vocabulary("artifacts/skill_vocab.pkl")

    prompt = f"""
    Extract the following information from the job description below.
    Return ONLY a valid JSON object with the exact keys:
    
    - "job_title": string (the exact job title)
    - "locations": list of strings (acceptable locations for the job)
    - "domain": string (the industry or domain of the company, e.g. HR-tech, Recruiting)
    - "required_skills": List of must-have skills (must be exact matches from the vocabulary list provided).
    - "preferred_skills": List of nice-to-have/preferred skills (must be exact matches from the vocabulary list provided).
    - "min_experience": Minimum years of experience required (integer, default 5 if not found).
    - "max_experience": Maximum years of experience expected (integer, default 9 if not found).
    - "preferred_notice_days": Preferred notice period in days (integer, default 30 if not found).
    
    Make sure the skills exist in the Vocabulary list provided.
    Vocabulary list:
    {vocab}
    
    Job Description:
    {jd_text}
    """

    llm_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    jd_parsed: dict = json.loads(llm_response.text)

    embed_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=jd_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
    )
    jd_vector: list[float] = embed_response.embeddings[0].values

    with open("artifacts/jd_parsed.pkl", "wb") as f:
        pickle.dump(jd_parsed, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("artifacts/jd_embedding.pkl", "wb") as f:
        pickle.dump(jd_vector, f, protocol=pickle.HIGHEST_PROTOCOL)

    return jd_parsed

if __name__ == "__main__":
    jd = Document("data/job_description.docx")
    jd_text = "\n".join(p.text for p in jd.paragraphs)

    result = parse_jd(jd_text)
    print(json.dumps(result, indent=2))