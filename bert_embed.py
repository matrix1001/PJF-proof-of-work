from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import json
import pickle
text2embedding = pickle.load(open('bert_embedding.pkl', 'rb'))

tokenizer = BertTokenizer.from_pretrained('D:/sync/wiki-us-officer-dataset2/models/bert-base-cased')
model = BertModel.from_pretrained('D:/sync/wiki-us-officer-dataset2/models/bert-base-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_bert_embedding(text, title=''):
    if text == 'None':
        text = title
    if text in text2embedding:
        return text2embedding[text]
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    # 获取[CLS] token的向量
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
    text2embedding[text] = cls_embedding
    return cls_embedding
def get_bert_embeddings(texts, titles=[]):
    embeddings = []
    if len(titles) == len(texts):
        for text, title in zip(texts, titles):
            embedding = get_bert_embedding(text, title)
            embeddings.append(embedding)
    else:
        for text in texts:
            embedding = get_bert_embedding(text)
            embeddings.append(embedding)
    return embeddings

dataset = json.load(open('./dataset.json'))
desc_texts = pd.read_excel('./gpt4_desc.xlsx')['desc'].tolist()
desc_titles = pd.read_excel('./gpt4_desc.xlsx')['item'].tolist()

bert_embeddings = get_bert_embeddings(desc_texts, desc_titles)

job_texts = dataset['jobs']
resume_texts = dataset['resumes']

bert_embeddings.extend(get_bert_embeddings(job_texts + resume_texts))

pickle.dump(text2embedding, open('bert_embedding.pkl', 'wb'))
