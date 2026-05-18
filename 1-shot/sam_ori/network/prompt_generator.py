import torch.nn as nn
import torch
from lib.utils import Attn, get_point_and_bbox_from_mask
from sam_ori.modeling import Sam
from sam_ori.modeling.common import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self,
                 channel: int = 1024,
                 fsem_embed_dim: int = 512,
                 prompt_embed_dim: int = 256,
                 num_queries: int = 30,
                 ):
        super(PromptGenerator, self).__init__()
        self.num_queries = num_queries
        self.embed_dim = fsem_embed_dim

        self.queries = nn.Parameter(torch.zeros(size=(num_queries, fsem_embed_dim)))  # [N, c]
        self.down_linear = nn.Sequential(nn.Linear(fsem_embed_dim, prompt_embed_dim), nn.GELU())

        self.cross_attn = Attn(dim=fsem_embed_dim)
        self.self_attn = Attn(dim=fsem_embed_dim)

    def forward(self, f_sem: torch.Tensor):
        m_coarse = f_sem
        queries = self.queries.unsqueeze(0).repeat(f_sem.shape[0], 1, 1)  # [B, N, c]
        q_sem = f_sem.reshape(f_sem.shape[0], -1, f_sem.shape[-1])
        saliency_info, _ = self.cross_attn(queries, q_sem, q_sem)
        saliency_info, _ = self.self_attn(saliency_info, saliency_info, saliency_info)
        saliency_info = self.down_linear(saliency_info)

        point, bbox = get_point_and_bbox_from_mask(m_coarse)

        return m_coarse, saliency_info, point, bbox
    
class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam):
        super(BasePromptEncodeAdapter, self).__init__()

        self.sam_prompt_encoder = ori_sam.prompt_encoder

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings