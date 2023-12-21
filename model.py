import torch.nn as nn
import torch
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, embed_dim):
        super().__init__()
        
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        
        encodings = self.get_encodings(height, width, embed_dim)
        self.register_buffer('encodings', encodings)
        
    def get_encodings(self, height, width, embed_dim):
        encodings = torch.zeros(height, width, embed_dim)
        
        for h in range(height):
            for w in range(width):
                for i in range(0, embed_dim, 2):
                    encodings[h,w,i] = math.sin(h / (10000 ** (i/embed_dim)))
                    encodings[h,w,i+1] = math.cos(h / (10000 ** (i/embed_dim)))
                    
                for i in range(0, embed_dim, 2):
                    encodings[h,w,i] = math.sin(w / (10000 ** (i/embed_dim)))  
                    encodings[h,w,i+1] = math.cos(w / (10000 ** (i/embed_dim)))
                    
        return encodings
        
    def forward(self, patch_x, patch_y):
        encodings = self.encodings[patch_x, patch_y]
        return encodings

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
    def __init__(self,num_patches_w = 4,num_patches_h = 4,embed_dim = 10, patch_dim = 49, hidden_dim = 100, num_layers = 3):
        super().__init__()
        self.layers = []
        num_patches = num_patches_h*num_patches_w
        self.idx_embed = nn.Embedding(num_patches,embed_dim)
        self.width_patches = num_patches_w
        self.mlp = MLP(embed_dim,patch_dim,hidden_dim,num_layers)
         
    def forward(self,patch_x,patch_y):    
        idx = patch_x*self.width_patches + patch_y
        x_embed = self.idx_embed(idx)
        out = self.mlp(x_embed)
        return out