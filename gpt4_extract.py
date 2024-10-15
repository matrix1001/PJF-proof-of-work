import os
from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key='what?')

import json
dataset = json.load(open('dataset.json'))
resumes =  dataset['resumes']
jobs = dataset['jobs']

job_skills = []
job_orgs = []
for job in jobs:
    prompt = 'Extract skills from the follwing job description to a python list, DO NOT RETURN ANYTHING ELSE: \n{}'.format(job)
    #print(prompt)
    query_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        )
    result = query_completion.choices[0].message.content
    #print(result)
    job_skills.append(result)

    prompt = 'Extract organizations from the follwing job description to a python list, DO NOT RETURN ANYTHING ELSE: \n{}'.format(job)
    #print(prompt)
    query_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        )
    result = query_completion.choices[0].message.content
    #print(result)
    job_orgs.append(result)
print(job_skills)
print(job_orgs)


job_detailed = {'names': jobs, 'skills': job_skills, 'orgs': job_orgs}
df = pd.DataFrame(job_detailed)
df.to_excel('gpt4_job_skills_orgs.xlsx')

for test in job_skills + job_orgs:
    eval(test)

resume_skills = []

for i, resume in enumerate(resumes):
    print(i)
    #resume = "Randolph was born in Riverside Township, New Jersey, and grew up in two communities in New Jersey, Palmyra and the Glendora section of Gloucester Township. He graduated from Triton Regional High School in 1961, as part of the school's first graduating class. Randolph earned a Bachelor of Science degree from Drexel University in 1966, majoring in economics and basic engineering. At Drexel, he was president of the debate society, vice president of the Student Senate, and a member of the varsity wrestling squad. He then attended the University of Pennsylvania Law School. He served as managing editor of the University of Pennsylvania Law Review and graduated summa cum laude in 1969 with a Juris Doctor degree, having been first in his class all three years of law school "
    prompt = 'Extract skills from the follwing resume to a python list, DO NOT RETURN ANYTHING ELSE: \n{}'.format(resume)
    #print(prompt)
    query_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        )
    result = query_completion.choices[0].message.content
    #print(result)
    resume_skills.append(result)

    chk_data = {'names': dataset['names'][:i+1],'resumes': dataset['resumes'][:i+1],'skills': resume_skills}
    df = pd.DataFrame(chk_data)
    df.to_excel('gpt4_extracted_skills.xlsx', index=False)

resume_orgs = []
for i, resume in enumerate(resumes):
    print(i)
    #resume = "Randolph was born in Riverside Township, New Jersey, and grew up in two communities in New Jersey, Palmyra and the Glendora section of Gloucester Township. He graduated from Triton Regional High School in 1961, as part of the school's first graduating class. Randolph earned a Bachelor of Science degree from Drexel University in 1966, majoring in economics and basic engineering. At Drexel, he was president of the debate society, vice president of the Student Senate, and a member of the varsity wrestling squad. He then attended the University of Pennsylvania Law School. He served as managing editor of the University of Pennsylvania Law Review and graduated summa cum laude in 1969 with a Juris Doctor degree, having been first in his class all three years of law school "
    prompt = 'Extract organizations from the follwing resume to a python list, DO NOT RETURN ANYTHING ELSE: \n{}'.format(resume)
    #print(prompt)
    query_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        )
    result = query_completion.choices[0].message.content
    #print(result)
    resume_orgs.append(result)

    chk_data = {'names': dataset['names'][:i+1],'resumes': dataset['resumes'][:i+1],'orgs': resume_orgs}
    df = pd.DataFrame(chk_data)
    df.to_excel('gpt4_extracted_orgs.xlsx', index=False)
