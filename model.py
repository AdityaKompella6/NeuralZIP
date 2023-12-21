import torch.nn as nn
import torch

class MLPBlock(nn.Module):
    def __init__(self,input_size,output_size,dropout = 0, norm = nn.LayerNorm, activation= nn.GELU):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size,output_size,bias=False))
        self.layers.append(norm(output_size))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())
        self.model = nn.Sequential(*self.layers)
        self.use_residual = input_size == output_size
         
    def forward(self,x):    
        if self.use_residual:
            return x + self.model(x)
        else:
            return self.model(x)
            
class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim, num_layers,dropout = 0, norm = nn.LayerNorm, activation= nn.GELU):
        super().__init__()
        assert num_layers >= 2
        self.layers = []
        self.layers.append(MLPBlock(input_size,hidden_dim,dropout=dropout,norm=norm,activation = activation))

        for i in range(num_layers - 2):
            self.layers.append(MLPBlock(hidden_dim,hidden_dim,dropout=dropout,norm=norm,activation = activation))
        self.layers.append(nn.Linear(hidden_dim,output_size))
        self.model = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.model(x)
    

class PatchModel(nn.Module):
    def __init__(self,num_patches = 16,embed_dim = 10, patch_dim = 49, hidden_dim = 100, num_layers = 3, width_patches = 4):
        super().__init__()
        self.layers = []
        self.idx_embed = nn.Embedding(num_patches,embed_dim)
        self.width_patches = width_patches
        self.mlp = MLP(embed_dim,patch_dim,hidden_dim,num_layers)
         
    def forward(self,patch_x,patch_y):    
        idx = patch_x*self.width_patches + patch_y
        x_embed = self.idx_embed(idx)
        out = self.mlp(x_embed)
        return out