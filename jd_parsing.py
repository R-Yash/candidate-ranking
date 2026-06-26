import re
import json
import os
import pickle
from google import genai
from dotenv import load_dotenv
from docx import Document
load_dotenv()

def get_vocabulary(path):
    with open(path, "rb") as f:
        vocab = pickle.load(f) 
    return vocab

def parse_jd(jd_text):
    client = genai.Client()
    vocab = get_vocabulary("skill_vocab.pkl")
    
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
    
    response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
    config={
        'response_mime_type': 'application/json'
        },
    )
    
    return response.text
    

if __name__ == "__main__":
    jd = Document('data/job_description.docx')
    jd_text = '\n'.join([paragraph.text for paragraph in jd.paragraphs])

    print(parse_jd(jd_text))