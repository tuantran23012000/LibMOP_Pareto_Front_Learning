
from torch import nn
class MLP(nn.Module):
    def __init__(self, ray_hidden_dim, out_dim, n_tasks,n_hidden_layers):
        super(MLP, self).__init__()
        self.n_tasks = n_tasks
        self.ray_hidden_dim = ray_hidden_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            setattr(self, f"hidden_layer_{i}", nn.Sequential(
                            nn.Linear(self.ray_hidden_dim, self.ray_hidden_dim),
                            nn.ReLU(inplace=True)))
        self.ray_encoder = nn.Sequential(
                nn.Linear(self.n_tasks, self.ray_hidden_dim),
                nn.ReLU(inplace=True),
            )
        self.output = nn.Sequential(
                nn.Linear(self.ray_hidden_dim, self.out_dim),
            )
    def forward(self, ray):
        x = self.ray_encoder(ray) 
        for i in range(self.n_hidden_layers):
            x = getattr(self, f"hidden_layer_{i}")(
            x
        )
        x = self.output(x)
        x = x.unsqueeze(0)
        return x