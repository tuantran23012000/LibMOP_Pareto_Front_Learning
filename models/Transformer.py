from torch import nn
import torch.nn.functional as F
import torch

class Trans(nn.Module):
    def __init__(self, ray_hidden_dim, out_dim, n_tasks,n_heads):
        super(Trans, self).__init__()
        self.n_tasks = n_tasks
        self.ray_hidden_dim = ray_hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        for i in range(self.n_tasks):
            setattr(self, f"task_encode_{i}", nn.Sequential(
                                        nn.Linear(1, self.ray_hidden_dim),
                                        nn.ReLU(inplace=True)))
        self.attention = nn.MultiheadAttention(embed_dim=self.ray_hidden_dim, num_heads=self.n_heads)
        self.ffn1 = nn.Linear(self.ray_hidden_dim,self.ray_hidden_dim)
        self.ffn2 = nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim)
        self.output_layer =  nn.Linear(self.ray_hidden_dim, out_dim)
    def transformer(self,x):
        x_ = x         
        x,_ = self.attention(x,x,x)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
        return x
    def forward(self, ray):
        task_encode = []
        for i in range(self.n_tasks):
            task_encode.append(getattr(self, f"task_encode_{i}")(
            ray[i].unsqueeze(0)
        ))
        x = torch.stack(task_encode)
        x = self.transformer(x)
        x = torch.mean(x,dim=0)
        x = x.unsqueeze(0)
        return x
