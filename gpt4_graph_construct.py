import json
import pickle
import pandas as pd

text2embedding = pickle.load(open('bert_embedding.pkl', 'rb'))
dataset = json.load(open('dataset.json', 'r', encoding='utf-8'))

person_org_df = pd.read_excel('./gpt4_extracted_orgs.xlsx')
person_skill_df = pd.read_excel('./gpt4_extracted_skills.xlsx')

desc_df = pd.read_excel('./gpt4_desc.xlsx')
item_desc = dict(zip(desc_df['item'], desc_df['desc']))

def eval_list(lst):
    return [eval(i) for i in lst]
person_orgs = dict(zip(person_org_df['names'], eval_list(person_org_df['orgs'])))
person_skills = dict(zip(person_skill_df['names'], eval_list(person_skill_df['skills'])))

job_skill_org_df = pd.read_excel('./gpt4_job_skills_orgs.xlsx')
job_skills = dict(zip(job_skill_org_df['names'], eval_list(job_skill_org_df['skills'])))
job_orgs = dict(zip(job_skill_org_df['names'], eval_list(job_skill_org_df['orgs'])))

skills = []
orgs = []
for person in person_skills:
    skills.extend(person_skills[person])
    orgs.extend(person_orgs[person])
for job in job_skills:
    skills.extend(job_skills[job])
    orgs.extend(job_orgs[job])
skills_set = set(skills)
orgs_set = set(orgs)


import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from torch_geometric.loader import NeighborLoader

# 创建异质信息网络数据
data = HeteroData()

# 添加节点
data['person'].num_nodes = len(person_skills)
data['job'].num_nodes = len(job_skills)
data['skill'].num_nodes = len(skills_set)
data['org'].num_nodes = len(orgs_set)

person_to_idx = {person: i for i, person in enumerate(dataset['names'])}
job_to_idx = {job: i for i, job in enumerate(dataset['jobs'])}
skill_to_idx = {skill: i for i, skill in enumerate(skills_set)}
org_to_idx = {org: i for i, org in enumerate(orgs_set)}


# person-skill 边
edge_index = [[], []]
for person, skills in person_skills.items():
    for skill in skills:
        edge_index[0].append(person_to_idx[person])
        edge_index[1].append(skill_to_idx[skill])
data['person', 'person-skill', 'skill'].edge_index = torch.tensor(edge_index, dtype=torch.long)

# person-org 边
edge_index = [[], []]
for person, orgs in person_orgs.items():
    for org in orgs:
        edge_index[0].append(person_to_idx[person])
        edge_index[1].append(org_to_idx[org])
data['person', 'person-org', 'org'].edge_index = torch.tensor(edge_index, dtype=torch.long)

# job-skill 边
edge_index = [[], []]
for job, skills in job_skills.items():
    for skill in skills:
        edge_index[0].append(job_to_idx[job])
        edge_index[1].append(skill_to_idx[skill])
data['job', 'job-skill', 'skill'].edge_index = torch.tensor(edge_index, dtype=torch.long)

# job-org 边
edge_index = [[], []]
for job, orgs in job_orgs.items():
    for org in orgs:
        edge_index[0].append(job_to_idx[job])
        edge_index[1].append(org_to_idx[org])
data['job', 'job-org', 'org'].edge_index = torch.tensor(edge_index, dtype=torch.long)

person_features = {name: text2embedding[dataset['resumes'][i]] for i, name in enumerate(dataset['names'])}
job_features = {job: text2embedding[job] for i, job in enumerate(dataset['jobs'])}
skill_features = {skill: text2embedding[item_desc[skill]] for skill in skills_set}
org_features = {org: text2embedding[item_desc[org]] for org in orgs_set}
data['person'].x = torch.tensor([person_features[person] for person in dataset['names']], dtype=torch.float)
data['job'].x = torch.tensor([job_features[job] for job in dataset['jobs']], dtype=torch.float)
data['skill'].x = torch.tensor([skill_features[skill] for skill in skills_set], dtype=torch.float)
data['org'].x = torch.tensor([org_features[org] for org in orgs_set], dtype=torch.float)

def check_person_edges(data, person_name):
    person_idx = person_to_idx[person_name]
    edges = {}
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        if src == 'person':
            edge_index = data[edge_type].edge_index
            mask = edge_index[0] == person_idx
            connected_nodes = edge_index[1][mask]
            if dst == 'skill':
                node_names = [list(skill_to_idx.keys())[list(skill_to_idx.values()).index(idx)] for idx in connected_nodes.tolist()]
                node_features = data['skill'].x[connected_nodes][:, :4]
            elif dst == 'org':
                node_names = [list(org_to_idx.keys())[list(org_to_idx.values()).index(idx)] for idx in connected_nodes.tolist()]
                node_features = data['org'].x[connected_nodes][:, :4]
            edges[edge_type] = list(zip(node_names, node_features.tolist()))
    return edges
person_name = 'Nicole_Malliotakis'
edges = check_person_edges(data, person_name)
print(edges)


import torch
from torch_geometric.data import HeteroData

# 假设你已经创建并加载了HeteroData对象
data = torch.load('gpt4_hetero_data.pt')

# 获取统计数据
def get_data_statistics(data):
    stats = {}
    stats['num_nodes'] = {node_type: data[node_type].num_nodes for node_type in data.node_types}
    stats['num_edges'] = {edge_type: data[edge_type].edge_index.size(1) for edge_type in data.edge_types}
    return stats

# 打印统计数据
stats = get_data_statistics(data)
print("节点数:")
for node_type, num_nodes in stats['num_nodes'].items():
    print(f"{node_type}: {num_nodes}")

print("\n边数:")
for edge_type, num_edges in stats['num_edges'].items():
    print(f"{edge_type}: {num_edges}")

