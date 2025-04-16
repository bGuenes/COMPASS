import torch
import torch.nn as nn
import math

#################################################################################################
# ///////////////////////////////////////// Transformer /////////////////////////////////////////
#################################################################################################

"""
    Transformer model for diffusion models.
    
    - modulate: Modulates the input with shift and scale.
    - InputEmbedder: Embeds joint data into vector representations.
    - TimestepEmbedder: Embeds scalar timesteps into vector representations.
    - Mlp: MLP for Output of Self-Attention
    - TransformerBlock: A ConditionTransformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    - FinalLayer: The final layer of ConditionTransformer.
    - ConditionTransformer: Transformer model for diffusion models.
"""

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

#############################################
# ----- Timestep & Input Embedding -----
#############################################

class InputEmbedder(nn.Module):
    """
    Embeds joint data into vector representations.
    """
    def __init__(self, nodes_size, hidden_size):
        super().__init__()
        self.embedding_params = nn.Parameter(torch.ones(1, nodes_size, hidden_size))

    def forward(self, x):
        x = x.unsqueeze(-1) * self.embedding_params
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, nodes_size, hidden_size, mlp_ratio=4.0, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mlp_ratio*hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_ratio*hidden_size, frequency_embedding_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        args = t * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
#############################################
# ----- MLP -----
#############################################

class Mlp(nn.Module):
    """
    MLP for Output of Self-Attention
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

#############################################
# ----- Core ConditionTransformer Model -----
#############################################

class TransformerBlock(nn.Module):
    """
    A ConditionTransformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, nodes_size, mlp_ratio=4.0, time_embedding_size=256, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.nodes_size = nodes_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, batch_first=True, **block_kwargs )  

        self.norm2 = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_size, 6 * nodes_size * hidden_size, bias=True)
        )
    
    def forward(self, x, c, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).reshape(-1, self.nodes_size, 6*self.hidden_size).chunk(6, dim=-1)
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        q, k, v = x_norm.repeat(1,1,3).chunk(3, dim=-1)

        # Attention mask to prevent latent nodes from attending to other latent nodes
        attn_mask = (1-c).type(torch.bool).unsqueeze(1).repeat(self.num_heads, self.nodes_size, 1)

        x = x + gate_msa * self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)[0]
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

#############################################
# ----- Final Layer -----
#############################################

class FinalLayer(nn.Module):
    """
    The final layer of ConditionTransformer.
    """
    def __init__(self, hidden_size, nodes_size, time_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.nodes_size = nodes_size

        self.norm_final = nn.LayerNorm((nodes_size, hidden_size), elementwise_affine=False, eps=1e-6)
        self.embedding_params = nn.Parameter(torch.zeros(nodes_size*hidden_size, nodes_size))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_size, 2 * nodes_size * hidden_size, bias=True)
        )

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).reshape(-1, self.nodes_size, 2*self.hidden_size).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = nn.SiLU()(x)
        x = x.flatten(1) @ self.embedding_params
        return x.squeeze(-1)

#############################################
# ----- ConditionTransformer Model -----
#############################################

class ConditionTransformer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        nodes_size,
        hidden_size=64,
        time_embedding_size=256,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.nodes_size = nodes_size
        self.num_heads = num_heads
        self.time_embedding_size = time_embedding_size

        self.x_embedder = InputEmbedder(nodes_size=nodes_size, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(nodes_size, hidden_size, mlp_ratio=mlp_ratio, frequency_embedding_size=time_embedding_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, nodes_size, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, nodes_size)                   
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ConditionTransformer blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, c):
        """
        Forward pass of ConditionTransformer.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N,) tensor of data conditions (latent or conditioned)
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        #c_embed = self.c_embedder(c.type(torch.int))
        #t += c_embed
       
        for block in self.blocks:
            x = block(x, c, t)
        
        x = self.final_layer(x, t)

        return x
