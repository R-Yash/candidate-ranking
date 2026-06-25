import re
import json
import os
import pickle 

def get_vocabulary(path):
    with open(path, "rb") as f:
        vocab = pickle.load(f) 
    return vocab

def parse_jd(jd_text):
    chunks = [c.strip().lower() for c in jd_text.split('\n\n') if c.strip()]
    skills_chunk = next((c for c in chunks if 'skills' in c.split('\n')[0]), "")
 
    required_lines, preferred_lines = [], []
    current_sub = None

    for line in skills_chunk.split('\n'):
        if 'absolutely need' in line:
            current_sub = 'required'
        elif "we'd like" in line or 'like you to have' in line:
            current_sub = 'preferred'
        elif 'do not want' in line:
            current_sub = None  
        elif current_sub == 'required':
            required_lines.append(line)
        elif current_sub == 'preferred':
            preferred_lines.append(line)
 
    required_section = ' '.join(required_lines)
    preferred_section = ' '.join(preferred_lines)
 
    vocab = get_vocabulary("skill_vocab.pkl")
 
    required_skills = [s for s in vocab if s in required_section]
    preferred_skills = [s for s in vocab if s in preferred_section]
 
    exp_match = re.search(r'(\d+)\s*[–\-]\s*(\d+)\s*years', jd_text)
    min_exp = int(exp_match.group(1)) if exp_match else 5
    max_exp = int(exp_match.group(2)) if exp_match else 9
 
    notice_match = re.search(r'sub-?(\d+)-?day', jd_text, re.IGNORECASE)
    preferred_notice_days = int(notice_match.group(1)) if notice_match else 30
 
    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "min_experience": min_exp,
        "max_experience": max_exp,
        "preferred_notice_days": preferred_notice_days,
    }