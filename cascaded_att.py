import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
dataset = json.load(open('dataset.json', 'r', encoding='utf-8'))
data_untrained = torch.load('gpt4_hetero_data.pt')
data = torch.load('gpt4_hetero_data_pretrained.pt')

raw_label = np.array(dataset['labels'])
train_indices, val_test_indices = train_test_split(list(range(raw_label.shape[0])), stratify=raw_label, test_size=0.4, random_state=996)
val_indices, test_indices = train_test_split(val_test_indices, stratify=raw_label[val_test_indices], test_size=0.5, random_state=996)

padding_skill_feature_avg = np.mean(data['skill'].x.detach().numpy(), axis=0)
padding_org_feature_avg = np.mean(data['org'].x.detach().numpy(), axis=0)

class CustomDataset(Dataset):
    def __init__(self, indices, data, data_untrained, raw_label):
        self.data = data
        self.data_untrained = data_untrained
        self.raw_label = raw_label
        self.indices = indices
        self.data_list = self.construct_dataset()

    def get_feature(self, typ, idx, pretrained=True):
        if pretrained:
            return self.data[typ].x[idx].detach().numpy()
        else:
            return self.data_untrained[typ].x[idx].detach().numpy()
    
    def get_neighbor_feature_avg(self, node_typ, neig_typ, idx, pretrained=True):
        if pretrained:
            data = self.data
        else:
            data = self.data_untrained
        triplet = (node_typ, node_typ+'-'+neig_typ, neig_typ)
        neighbor_nodes = data[triplet].edge_index[1][data[triplet].edge_index[0] == idx]
        if neighbor_nodes.shape[0] != 0:
            return np.mean(data[neig_typ].x[neighbor_nodes].detach().numpy(), axis=0)
        else:
            return padding_skill_feature_avg if neig_typ =='skill' else padding_org_feature_avg
        
    def construct_dataset(self):
        data_list = []
        for idx in self.indices:
            bert_feature = self.get_feature('person', idx, pretrained=False)
            node_feature = self.get_feature('person', idx)
            neighbor_skill_features_avg = self.get_neighbor_feature_avg('person', 'skill', idx)
            neighbor_org_features_avg = self.get_neighbor_feature_avg('person', 'org', idx)
            
            labels = self.raw_label[idx]
            for job_idx, y in enumerate(labels):
                job_bert_feature = self.get_feature('job', job_idx, pretrained=False)
                job_node_feature = self.get_feature('job', job_idx)
                job_neighbor_skill_features_avg = self.get_neighbor_feature_avg('job','skill', job_idx)
                job_neighbor_org_features_avg = self.get_neighbor_feature_avg('job', 'org', job_idx)
                
                data_list.append((bert_feature.copy(), node_feature.copy(), neighbor_skill_features_avg.copy(), neighbor_org_features_avg.copy(),\
                    job_bert_feature, job_node_feature, job_neighbor_skill_features_avg, job_neighbor_org_features_avg, y))

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

train_dataset = CustomDataset(train_indices, data, data_untrained, raw_label)
val_dataset = CustomDataset(val_indices, data, data_untrained, raw_label)
test_dataset = CustomDataset(test_indices, data, data_untrained, raw_label)

def get_metrics(inputs, outputs, labels):
    # 获取 exe, leg, jud 的嵌入向量
    exe_embed = data_untrained['job'].x[0].detach().numpy()
    leg_embed = data_untrained['job'].x[1].detach().numpy()
    jud_embed = data_untrained['job'].x[2].detach().numpy()

    # 初始化结果字典
    results = {
        'exe': {'outputs': [], 'labels': []},
        'leg': {'outputs': [], 'labels': []},
        'jud': {'outputs': [], 'labels': []},
    }

    # 将数据分配到对应类别
    for input, output, label in zip(inputs, outputs, labels):
        job_embed = input[768*4:768*5].detach().numpy()
        if np.array_equal(job_embed, exe_embed):
            results['exe']['outputs'].append(output.detach().item())
            results['exe']['labels'].append(label.detach().item())
        elif np.array_equal(job_embed, leg_embed):
            results['leg']['outputs'].append(output.detach().item())
            results['leg']['labels'].append(label.detach().item())
        elif np.array_equal(job_embed, jud_embed):
            results['jud']['outputs'].append(output.detach().item())
            results['jud']['labels'].append(label.detach().item())
        else:
            print('unknown job')

    # 计算每个类别的指标以及整体平均值
    metrics = {'precision': {}, 'recall': {}, 'f1': {}}
    precisions, recalls, f1s = [], [], []

    for key in ['exe', 'leg', 'jud']:
        if len(results[key]['labels']) > 0:
            preds = np.array(results[key]['outputs']) >= 0.5  # 将输出转换为二分类标签（0 或 1）
            true_labels = np.array(results[key]['labels'])

            precision = precision_score(true_labels, preds)
            recall = recall_score(true_labels, preds)
            f1 = f1_score(true_labels, preds)

            metrics['precision'][key] = precision
            metrics['recall'][key] = recall
            metrics['f1'][key] = f1

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        else:
            metrics['precision'][key] = None
            metrics['recall'][key] = None
            metrics['f1'][key] = None

    # 计算整体的平均值
    metrics['precision']['average'] = np.mean([p for p in precisions if p is not None])
    metrics['recall']['average'] = np.mean([r for r in recalls if r is not None])
    metrics['f1']['average'] = np.mean([f for f in f1s if f is not None])
    return metrics

def print_metrics(metrics):
    print("\tPrec.\tRecall\tF1")
    for key in ['exe', 'leg', 'jud']:
        if metrics['precision'][key] is not None:
            print(f"{key}:\t{metrics['precision'][key]:.4f}\t{metrics['recall'][key]:.4f}\t{metrics['f1'][key]:.4f}")
        else:
            print(f"  No data available for {key}")
    print(f"avg:\t{metrics['precision']['average']:.4f}\t{metrics['recall']['average']:.4f}\t{metrics['f1']['average']:.4f}")
        
# 在测试模型时调用 show_metrics 函数
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_inputs, all_outputs, all_labels = [], [], []
    with torch.no_grad():
        for x1, x2, x3, x4, x5, x6, x7, x8, labels in test_loader:
            inputs = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
            labels = labels.float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 收集所有输入、输出和标签
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    # 将所有批次的数据拼接在一起
    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 打印测试损失
    #print(f'Test Loss: {test_loss / len(test_loader):.4f}')

    # 调用 show_metrics 计算并显示指标
    metrics = get_metrics(all_inputs, all_outputs, all_labels)
    return test_loss / len(test_loader), metrics

import warnings
warnings.filterwarnings("ignore")
class MLP(nn.Module):
    def __init__(self, num_heads=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8*768, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.person_atts = nn.ModuleList([nn.MultiheadAttention(768, num_heads=num_heads) for _ in range(4)])
        self.person_layer_norms = nn.ModuleList([nn.BatchNorm1d(768) for _ in range(4)])
        self.person_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(4)])
        self.job_atts = nn.ModuleList([nn.MultiheadAttention(768, num_heads=num_heads) for _ in range(4)])
        self.job_layer_norms = nn.ModuleList([nn.BatchNorm1d(768) for _ in range(4)])
        self.job_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(4)])


    def cascaded_att(self, x1, x2, x3, x4, typ):
        if typ == 'job':
            atts = self.job_atts
            dropouts = self.job_dropouts
            layer_norms = self.job_layer_norms
        else:
            atts = self.person_atts
            dropouts = self.person_dropouts
            layer_norms = self.person_layer_norms
        # 增加维度以符合 MultiheadAttention 输入的要求
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x4 = x4.unsqueeze(0)

        # 第一层注意力
        c1, _ = atts[0](x1, x1, x1)
        
        # 第二层注意力
        c2, _ = atts[1](c1.unsqueeze(0), x2, x2)
        c2 = dropouts[1](layer_norms[1](c2.squeeze(0) + c1.squeeze(0)))

        # 第三层注意力
        c3, _ = atts[2](c2.unsqueeze(0), x3, x3)
        c3 = dropouts[2](layer_norms[2](c3.squeeze(0) + c2.squeeze(0)))

        # 第四层注意力
        c4, _ = atts[3](c3.unsqueeze(0), x4, x4)
        c4 = dropouts[3](layer_norms[3](c4.squeeze(0) + c3.squeeze(0)))

        return torch.cat((c1.squeeze(0), c2.squeeze(0), c3.squeeze(0), c4.squeeze(0)), dim=1)

    def forward(self, x):
        person_x1, person_x2, person_x3, person_x4 = x[:,:768], x[:,768:768*2], x[:,768*2:768*3], x[:,768*3:768*4]
        job_x1, job_x2, job_x3, job_x4 = x[:,768*4:768*5], x[:,768*5:768*6], x[:,768*6:768*7], x[:,768*7:768*8]
        person_x = self.cascaded_att(person_x1, person_x2, person_x3, person_x4, 'person')
        job_x = self.cascaded_att(job_x1, job_x2, job_x3, job_x4, 'job')
        x = torch.cat((person_x, job_x), dim=1)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = MLP(num_heads=4)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)

# 训练模型
num_epochs = 40

best_val_f1 = 0
best_test_metric = None
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x1,x2,x3,x4,x5,x6,x7,x8,labels in train_loader:
        optimizer.zero_grad()
        inputs = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1)
        labels = labels.float().unsqueeze(1)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, val_f1: {val_metrics['f1']['average']:.4f}, test_f1: {test_metrics['f1']['average']:.4f}")
    if val_metrics['f1']['average'] > best_val_f1:
        best_val_f1 = val_metrics['f1']['average']
        best_test_metric = test_metrics
if best_test_metric is not None:
    print_metrics(best_test_metric)