import json
data = json.load(open('dataset.json'))
print(f'total number of resumes: {len(data["resumes"])}')
print(f'total number of jobs: {len(data["jobs"])}')
job_applications = sum(sum(label) for label in data['labels'])
print(f'total number of job applications: {job_applications}')
exe_cnt = sum(1 for label in data['labels'] if label[0] == 1)
leg_cnt = sum(1 for label in data['labels'] if label[1] == 1)
jud_cnt = sum(1 for label in data['labels'] if label[2] == 1)
print(f'number of job applications for executive positions: {exe_cnt}')
print(f'number of job applications for legal positions: {leg_cnt}')
print(f'number of job applications for judicial positions: {jud_cnt}')
count_one_1 = sum(1 for label in data['labels'] if label.count(1) == 1)
count_one_2 = sum(1 for label in data['labels'] if label.count(1) == 2)
count_one_3 = sum(1 for label in data['labels'] if label.count(1) == 3)
print(f'number of job applications with one label: {count_one_1}')
print(f'number of job applications with two labels: {count_one_2}')
print(f'number of job applications with three labels: {count_one_3}')

# result:
'''
total number of resumes: 3452
total number of jobs: 3
total number of job applications: 4389
number of job applications for executive positions: 1664
number of job applications for legal positions: 1533
number of job applications for judicial positions: 1192
number of job applications with one label: 2530
number of job applications with two labels: 907
number of job applications with three labels: 15
'''