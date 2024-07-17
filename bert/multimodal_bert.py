
from .modeling_bert import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiModalBert(BertModel):
    def __init__(self, config, embed_dim, pwam_idx=[3,6,9,12], num_heads_fusion=[1,1,1,1], fusion_drop=0.0):
        super().__init__(config)
        self.pwam_idx = pwam_idx
        self.num_heads_fusion = num_heads_fusion
        self.fusion_drop = fusion_drop

        pwam_dims=[embed_dim * 2** i for i in range(len(pwam_idx))] 
        #print(pwam_dims)
        self.pwams = nn.ModuleList()
        self.res_gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(0, len(pwam_idx)):
            dim = pwam_dims[i]
            fusion = PWAM(768,  # both the visual input and for combining, num of channels
                          dim,  # v_in
                          768,  # l_in
                          768,  # key
                          768,  # value
                          num_heads=num_heads_fusion[i],
                          dropout=fusion_drop)
            self.pwams.append(fusion)

            res_gate = nn.Sequential(
                nn.Linear(768, 768, bias=False),
                nn.ReLU(),
                nn.Linear(768, 768, bias=False),
                nn.Tanh()
            )
            nn.init.zeros_(res_gate[0].weight)
            nn.init.zeros_(res_gate[2].weight)
            self.res_gates.append(res_gate)

            self.norms.append(nn.LayerNorm(768))

    def forward_stem(self, input_ids, attention_mask):
        input_shape = input_ids.size()
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        #print(embedding_output.shape, extended_attention_mask.shape, "?>>>")
        return embedding_output, extended_attention_mask

    def forward_stage1(self, hidden_states, attention_mask):
        for i in range(0, self.pwam_idx[0]):
            layer_module = self.encoder.layer[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return layer_outputs[0]

    def forward_stage2(self, hidden_states, attention_mask):
        for i in range(self.pwam_idx[0], self.pwam_idx[1]):
            layer_module = self.encoder.layer[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return layer_outputs[0]

    def forward_stage3(self, hidden_states, attention_mask):
        for i in range(self.pwam_idx[1], self.pwam_idx[2]):
            layer_module = self.encoder.layer[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return layer_outputs[0]

    def forward_stage4(self, hidden_states, attention_mask):
        for i in range(self.pwam_idx[2], self.pwam_idx[3]):
            layer_module = self.encoder.layer[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return layer_outputs[0]

    def forward_pwam1(self, x, l, l_mask):
        l_residual = self.pwams[0](x, l, l_mask)
        l = l + (self.res_gates[0](l_residual) * l_residual)
        return self.norms[0](l_residual), l

    def forward_pwam2(self, x, l, l_mask):
        l_residual = self.pwams[1](x, l, l_mask)
        l = l + (self.res_gates[1](l_residual) * l_residual)
        return self.norms[1](l_residual), l

    def forward_pwam3(self, x, l, l_mask):
        l_residual = self.pwams[2](x, l, l_mask)
        l = l + (self.res_gates[2](l_residual) * l_residual)
        return self.norms[2](l_residual), l

    def forward_pwam4(self, x, l, l_mask):
        l_residual = self.pwams[3](x, l, l_mask)
        l = l + (self.res_gates[3](l_residual) * l_residual)
        return self.norms[3](l_residual), l

class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        #self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
        #                                 nn.GELU(),
        #                                 nn.Dropout(dropout)
        #                                )
        #self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
        self.vis_project = nn.Sequential(nn.Linear(dim, dim),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        #print("???", x.shape, l.shape, l_mask.shape)
        #print(self.vis_project)
        #vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        vis = self.vis_project(l)  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        #print("vis", vis.shape, "lang", lang.shape)
        mm = torch.mul(vis.permute(0,2,1), lang)
        #print(mm.shape)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm

        #self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
        #                   dim,  # v_in
        #                   768,  # l_in
        #                   dim,  # key
        #                   dim,  # value
        #                   num_heads=num_heads_fusion,
        #                   dropout=fusion_drop)

class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_query = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_key = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        #self.f_value = nn.Sequential(
        #    nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        #)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        #print('input shape', x.shape, l.shape, l_mask.shape)
        l_mask = l_mask.squeeze(1)
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l = l.permute(0,2,1)
        #l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        l_mask = l_mask  # (B, N_l, 1) -> (B, 1, N_l)

        #query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        #query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        #key = self.f_key(l)  # (B, key_channels, N_l)
        #value = self.f_value(l)  # (B, self.value_channels, N_l)
        #key = key * l_mask  # (B, key_channels, N_l)
        #value = value * l_mask  # (B, self.value_channels, N_l)

        #print(l.shape, self.f_query)
        query = self.f_query(l)  # (B, key_channels, H*W) if Conv1D
        query = query * l_mask  # (B, key_channels, N_l)
        query = query.permute(0, 2, 1)  # (B, N_l, key_channels)

        key = self.f_key(x)  # (B, key_channels, H*W) if Conv1D
        value = self.f_value(x)  # (B, key_channels, H*W) if Conv1D

        n_l = query.size(1)
        #print(query.shape, key.shape, value.shape)

        #query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        #key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        #value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, HW)
        value = value.reshape(B, self.num_heads, self.key_channels//self.num_heads, HW)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        #query = query.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        query = query.reshape(B, n_l, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        #value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        #print('after reshape', query.shape, key.shape, value.shape)

        l_mask = l_mask.unsqueeze(-1)  # (b, 1, 1, n_l)

        #sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, N_l, H*W)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        #print('out', out.shape)
        #out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, n_l, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out
