import os
from openai import OpenAI
import pandas as pd
import os

client = OpenAI(api_key='what?')

skill_lst = pd.read_excel('./gpt4_extracted_skills.xlsx')['skills'].tolist()
org_lst = pd.read_excel('./gpt4_extracted_orgs.xlsx')['orgs'].tolist()

job_detail_df = pd.read_excel('./gpt4_job_skills_orgs.xlsx')
skill_lst.extend(job_detail_df['skills'].tolist())
org_lst.extend(job_detail_df['orgs'].tolist())

desc_df = pd.read_excel('./gpt4_desc.xlsx')
item_desc = dict(zip(desc_df['item'], desc_df['desc']))

skills = []
for item in skill_lst:
    try:
        skills.extend(eval(item))
    except:
        print(item)
orgs = []
for item in org_lst:
    try:
        orgs.extend(eval(item))
    except:
        print(item)


skill_set = set(skills)
org_set = set(orgs)
print(len(skill_set))
print(len(org_set))
print(len(skill_set & org_set))

total_set = skill_set | org_set
print(len(total_set))

item_lst = list(total_set)

for i, item in enumerate(item_lst):
    if item in item_desc:
        continue
    print(i)
    prompt = 'Give me a brief description of the following content. If you don\'t know, return `None`. DO NOT RETURN ANYTHING ELSE: \n{}'.format(item)
    print(prompt)
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
    item_desc[item] = result
    print(result)

df = pd.DataFrame(list(item_desc.items()), columns=['item', 'desc'])

df.to_excel('gpt4_desc.xlsx')