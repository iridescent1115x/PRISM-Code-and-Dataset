import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.layers.common import MyMLP
from prism.layers.mgdcf import MGDCF


class SoftHopPropagator(nn.Module):
    """Learn per-node hop weights over 1..K propagation steps."""

    def __init__(self, max_k, alpha, beta, edge_drop_rate, z_drop_rate, in_dim):
        super().__init__()
        self.max_k = max_k
        self.mgdcf = MGDCF(max_k, alpha, beta, 0.0, edge_drop_rate, z_drop_rate)
        self.hop_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, max_k),
        )

    def forward(self, g, x):
        hop_list = self.mgdcf(g, x, return_all=True)  # list[K] of [N, D]
        hop_stack = torch.stack(hop_list, dim=1)  # [N, K, D]

        hop_logits = self.hop_gate(x)
        hop_weights = F.softmax(hop_logits, dim=-1)  # [N, K]
        out = (hop_stack * hop_weights.unsqueeze(-1)).sum(dim=1)
        return out, hop_weights


class AdaptivePRISM(nn.Module):
    """
    Innovations:
    1) Adaptive hop weighting (soft K selection) per node and modality.
    2) Information-driven global sampling for memory construction.
    3) Conflict-aware multimodal fusion with gated weighting.
    """

    def __init__(
        self,
        k_e,
        k_t,
        k_v,
        alpha,
        beta,
        input_feat_drop_rate,
        feat_drop_rate,
        user_x_drop_rate,
        item_x_drop_rate,
        edge_drop_rate,
        z_drop_rate,
        user_in_channels=None,
        item_v_in_channels=None,
        item_v_hidden_channels_list=None,
        item_t_in_channels=None,
        item_t_hidden_channels_list=None,
        bn=True,
        num_clusters=5,
        num_samples=10,
        *args,
        **kwargs,
    ):
        super().__init__()

        embed_dim = item_t_hidden_channels_list[-1]

        self.num_clusters = max(1, int(num_clusters))
        self.num_samples = max(1, int(num_samples))
        self.embed_dim = embed_dim

        self.user_gnn_input_dropout = nn.Dropout(user_x_drop_rate)
        self.item_gnn_input_dropout = nn.Dropout(item_x_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)

        self.t_mlp = nn.Sequential(
            nn.Dropout(input_feat_drop_rate),
            MyMLP(
                item_t_in_channels,
                item_t_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=bn,
            ),
        )
        self.v_mlp = nn.Sequential(
            nn.Dropout(input_feat_drop_rate),
            MyMLP(
                item_v_in_channels,
                item_v_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=bn,
            ),
        )

        self.emb_prop = (
            SoftHopPropagator(k_e, alpha, beta, edge_drop_rate, 0.0, embed_dim)
            if k_e >= 1
            else None
        )
        self.t_prop = (
            SoftHopPropagator(k_t, alpha, beta, edge_drop_rate, 0.0, embed_dim)
            if k_t >= 1
            else None
        )
        self.v_prop = (
            SoftHopPropagator(k_v, alpha, beta, edge_drop_rate, 0.0, embed_dim)
            if k_v >= 1
            else None
        )

        # conflict-aware fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3 + 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
        )
        self.conflict_scale = nn.Parameter(torch.tensor(1.0))

        # information-driven sampling score heads
        self.importance_head = nn.Linear(embed_dim, 1)
        self.score_coef = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

    @staticmethod
    def _safe_entropy(prob):
        prob = prob.clamp(min=1e-9)
        ent = -(prob * prob.log()).sum(dim=-1)
        norm = torch.log(torch.tensor(prob.size(-1), device=prob.device, dtype=prob.dtype))
        return ent / (norm + 1e-9)

    @staticmethod
    def _cosine_conflict(a, b):
        sim = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
        return (1.0 - sim).unsqueeze(-1)

    def _build_info_driven_memory(self, g, combined_h, modal_weights, hop_weights_dict, num_users):
        item_h = combined_h[num_users:]  # [I, D]
        num_items = item_h.size(0)
        device = item_h.device

        # representativeness (close to dataset centroid)
        center = item_h.mean(dim=0, keepdim=True)
        represent = F.cosine_similarity(item_h, center, dim=-1, eps=1e-8)

        # uncertainty from fusion entropy + hop entropy
        fusion_unc = self._safe_entropy(modal_weights[num_users:])

        hop_unc_list = []
        for hop_weights in hop_weights_dict.values():
            if hop_weights is not None:
                hop_unc_list.append(self._safe_entropy(hop_weights[num_users:]))
        if len(hop_unc_list) > 0:
            hop_unc = torch.stack(hop_unc_list, dim=0).mean(dim=0)
            uncertainty = 0.5 * fusion_unc + 0.5 * hop_unc
        else:
            uncertainty = fusion_unc

        # centrality proxy from graph in-degree
        item_deg = g.in_degrees()[num_users:].float().to(device)
        item_deg = (item_deg - item_deg.min()) / (item_deg.max() - item_deg.min() + 1e-9)

        # learnable importance proxy
        importance = self.importance_head(item_h).squeeze(-1)

        coef = F.softplus(self.score_coef)
        score = (
            coef[0] * represent
            + coef[1] * uncertainty
            + coef[2] * item_deg
            + coef[3] * importance
        )

        num_global = max(1, min(num_items, self.num_samples, self.num_clusters - 1))
        top_idx = torch.topk(score, k=num_global, dim=0, largest=True).indices
        global_tokens = item_h[top_idx]  # [S, D]

        if self.num_clusters > 1:
            if global_tokens.size(0) < self.num_clusters - 1:
                pad_n = self.num_clusters - 1 - global_tokens.size(0)
                global_tokens = torch.cat([global_tokens, global_tokens[:1].repeat(pad_n, 1)], dim=0)
            global_tokens = global_tokens[: self.num_clusters - 1]
            item_memory = torch.cat(
                [item_h.unsqueeze(1), global_tokens.unsqueeze(0).expand(num_items, -1, -1)], dim=1
            )
        else:
            item_memory = item_h.unsqueeze(1)

        user_memory = torch.zeros(num_users, self.num_clusters, self.embed_dim, device=device)
        z_memory_h = torch.cat([user_memory, item_memory], dim=0)
        return z_memory_h, score

    def forward(self, g, user_embeddings, item_v_feat, item_t_feat, item_embeddings=None, return_all=False):
        encoded_t = self.t_mlp(item_t_feat)
        encoded_v = self.v_mlp(item_v_feat)

        num_users = user_embeddings.size(0)
        num_items = encoded_t.size(0)
        device = user_embeddings.device

        # modality inputs
        emb_item = torch.zeros_like(encoded_t) if item_embeddings is None else item_embeddings
        emb_input = torch.cat(
            [self.user_gnn_input_dropout(user_embeddings), self.item_gnn_input_dropout(emb_item)], dim=0
        )
        t_input = torch.cat([torch.zeros_like(user_embeddings), self.item_gnn_input_dropout(encoded_t)], dim=0)
        v_input = torch.cat([torch.zeros_like(user_embeddings), self.item_gnn_input_dropout(encoded_v)], dim=0)

        emb_h, emb_hop_w = (self.emb_prop(g, emb_input) if self.emb_prop is not None else (None, None))
        t_h, t_hop_w = (self.t_prop(g, t_input) if self.t_prop is not None else (None, None))
        v_h, v_hop_w = (self.v_prop(g, v_input) if self.v_prop is not None else (None, None))

        if emb_h is None:
            emb_h = torch.zeros(num_users + num_items, self.embed_dim, device=device)
        if t_h is None:
            t_h = torch.zeros(num_users + num_items, self.embed_dim, device=device)
        if v_h is None:
            v_h = torch.zeros(num_users + num_items, self.embed_dim, device=device)

        c_et = self._cosine_conflict(emb_h, t_h)
        c_ev = self._cosine_conflict(emb_h, v_h)
        c_tv = self._cosine_conflict(t_h, v_h)
        conflict_feat = torch.cat([c_et, c_ev, c_tv], dim=-1)

        fusion_in = torch.cat([emb_h, t_h, v_h, conflict_feat], dim=-1)
        fusion_logits = self.fusion_gate(fusion_in)

        # conflict-aware damping for conflicting text/vision channels
        tv_conf = c_tv.squeeze(-1)
        fusion_logits[:, 1] = fusion_logits[:, 1] - self.conflict_scale * tv_conf
        fusion_logits[:, 2] = fusion_logits[:, 2] - self.conflict_scale * tv_conf

        modal_weights = F.softmax(fusion_logits, dim=-1)  # [N, 3]
        combined_h = (
            modal_weights[:, 0:1] * emb_h
            + modal_weights[:, 1:2] * t_h
            + modal_weights[:, 2:3] * v_h
        )
        combined_h = self.z_dropout(combined_h)

        hop_weights_dict = {"emb": emb_hop_w, "t": t_hop_w, "v": v_hop_w}
        z_memory_h, sample_score = self._build_info_driven_memory(
            g, combined_h, modal_weights, hop_weights_dict, num_users
        )

        if return_all:
            aux = {
                "modal_weights": modal_weights,
                "hop_weights": hop_weights_dict,
                "sample_score": sample_score,
            }
            return combined_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h, aux
        return combined_h

