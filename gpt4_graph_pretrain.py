import torch
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear, RGCNConv
from torch_geometric.utils import negative_sampling

# 加载数据
data = torch.load('gpt4_hetero_data.pt')
data = T.ToUndirected()(data)

edge_index = []
edge_type = []
x = []
num_nodes = 0
node_offset = {}

# 计算每个节点类型的偏移量
for node_type in data.node_types:
    node_offset[node_type] = num_nodes
    x.append(data[node_type].x)
    num_nodes += data[node_type].x.size(0)

# 转换edge_index并加上偏移
for edge_type_tuple in data.edge_types:
    src, rel, dst = edge_type_tuple
    offset_src = node_offset[src]
    offset_dst = node_offset[dst]
    edge_index_src, edge_index_dst = data[edge_type_tuple].edge_index
    edge_index_src = edge_index_src + offset_src
    edge_index_dst = edge_index_dst + offset_dst
    edge_index.append(torch.stack([edge_index_src, edge_index_dst], dim=0))
    edge_type.append(torch.full((data[edge_type_tuple].edge_index.size(1),), len(edge_type)))

edge_index = torch.cat(edge_index, dim=1)
edge_type = torch.cat(edge_type, dim=0)
x = torch.cat(x, dim=0)

data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, edge_type=edge_type)
print(data)

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)
        self.relation_embedding = torch.nn.Embedding(num_relations, out_channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * out_channels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def predict(self, s, r, o):
        r_emb = self.relation_embedding(r)
        x = torch.cat([s, r_emb, o], dim=-1)
        return self.mlp(x)
# 模型参数
num_nodes = data.num_nodes
in_channels = data.x.size(1)
out_channels = 768  # 可以根据需要调整
num_relations = torch.max(data.edge_type).item() + 1

# 初始化模型
model = RGCN(num_nodes, in_channels, out_channels, num_relations)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义邻居加载器
train_loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=1024, shuffle=True)

# 训练模型
def train(data, model, optimizer, train_loader):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 正样本
        pos_edge_index = batch.edge_index
        pos_edge_type = batch.edge_type
        
        # 负样本
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=batch.num_nodes)
        neg_edge_type = torch.randint(0, num_relations, (neg_edge_index.size(1),)).to(data.x.device)
        
        # 前向传播
        out = model(batch.x, pos_edge_index, pos_edge_type)
        
        # 获取节点嵌入
        s_pos = out[pos_edge_index[0]]
        o_pos = out[pos_edge_index[1]]
        s_neg = out[neg_edge_index[0]]
        o_neg = out[neg_edge_index[1]]
        
        # 计算损失
        pos_scores = model.predict(s_pos, pos_edge_type, o_pos)
        neg_scores = model.predict(s_neg, neg_edge_type, o_neg)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 训练循环
for epoch in range(100):  # 可以根据需要调整
    loss = train(data, model, optimizer, train_loader)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')


model.eval()
node_embeddings = model(data.x, data.edge_index, data.edge_type)

pretrained_data = torch.load('gpt4_hetero_data.pt')
pretrained_data = T.ToUndirected()(pretrained_data)

start = 0
for node_type in pretrained_data.node_types:
    num_nodes = pretrained_data[node_type].x.size(0)
    pretrained_data[node_type].x = node_embeddings[start:start + num_nodes]
    start += num_nodes
torch.save(pretrained_data, 'gpt4_hetero_data_pretrained.pt')