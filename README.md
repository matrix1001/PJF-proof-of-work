# PJF-proof-of-work
Proof of work for my paper addressing Person-Job Fit. For review only. (promise to reformat after acceptance)

file list:
- bert_embed.py script to encode texts to bert vectors
- bert_embdding.pkl bert vectors, failed to upload due to size
- cascaded_att.py our multi-view feature fusion and PJF model
- dataset.json raw text dataset
- gpt4_desc.py using gpt4 to get descriptions of skills and organizations
- gpt4_extract.py extracting skills and organizations from resumes
- gpt4_extracted_orgs.xlsx as its name
- gpt4_extracted_skills.xlsx
- gpt4_graph_construct.py using upper two excel to build the graph data
- gpt4_graph_pretrain.py RGCN training
- gpt4_hetero_data_pretrained.pt pretrained graph data, failed to upload due to size
- gpt4_hetero_data.pt untrained graph data, failed to upload due to size
- gpt4_job_sklils_orgs.xlsx extracted skills and organizations from job descriptions
- statistics.py


the pipeline:

1. gpt4_extract.py
2. gpt4_desc.py
3. bert_embed.py
4. gpt4_graph_construct.py
5. gpt4_graph_pretrain.py
6. cascaded_att.py


reading suggestion:
read dataset.json, then read those excels if you want to know how we build the network. read cascaded_att.py to know our model.
