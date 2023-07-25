# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,

                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 # 下面两个参数是Encoder使用的
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type,
                                                              key_aware_type=key_aware_type,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          decoder_query_perturber=decoder_query_perturber,
                                          dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                          dec_layer_share=dec_layer_share,
                                          use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                          )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == 'standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:

                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        # 这两个属性是在dino类中init的时期直接被赋值的
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                           0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[
                           0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]

            prepare_for_cdn生成的
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi] 空间位置编码
            - tgt: [bs, num_dn, d_model]. None in infer  prepare_for_cdn生成的

        """
        # prepare input for encoder
        # 先处理一下输入特征 ---------------------------------------------------------------------------------------------
        src_flatten = []
        mask_flatten = []
        # 标识是哪一层特征层
        lvl_pos_embed_flatten = []
        # 各个特征层的高宽
        spatial_shapes = []
        # 这地方的处理与Deformable DETR是相同的
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # src mask pos_embed 尺寸是相同的
            bs, c, h, w = src.shape
            # 特征图的高宽
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c

            # 这个是Deformable DETR论文中提到的Level embed
            if self.num_feature_levels > 1 and self.level_embed is not None:
                # pos_embed和level_embed相加
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:

                lvl_pos_embed = pos_embed

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # [bs,all hw,256]
        # 所有特征层的拼在一起
        # 他们就是在维度1上长度不同，尺寸越大的特征层，维度1上的数量越多
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        # [bs,all hw]
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        # [bs, all hw,256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        # [特征层的数量，2] 存储的是高宽
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 各个src层 起始的位置, 第一个spatial_shapes.new_zeros((1,))是在起始位置填的0
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 有效高宽占总的batch高宽的比率 [bs,4,2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # 以上的处理与Deformable DETR基本一致
        # -------------------------------------------------------------------------------------------------------------


        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        # 先经过encoder处理
        # 后两个输出值是encoder的中间层的输出
        #########################################################
        # Begin Encoder
        #########################################################
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            # 空间位置编码
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten, # 以上这些参数是上面处理的
            # 这两个参数是Deformable DETR中没有的，其他的是相同的, 不过这两个参数在上面设定为了None
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        if self.two_stage_type == 'standard':

            if self.two_stage_learn_wh:

                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None

            # output_memory 是memory 经过填充inf，经过一层全连接后的结果 [bs,all hw,256]
            # output_proposals 是制作的proposals，非法的位置填充了inf [bs,all hw,4]
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes,
                                                                           input_hw)
            # 这一行在Deformable DETR中是在gen_encoder_output_proposals方法中的，这里是放到了这里
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            # 0
            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape

                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256

                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)

                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)

                output_memory = output_memory + _pats

                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            # 0
            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None

                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            # 经过分类头 [bs,sum(hw),91]
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            # 经过box头 [bs,sum(hw),4] +output_proposals 是因为经过网络头的结果是修正
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            # enc_outputs_class_unselected.max(-1)[0] 是各个91中的最大值
            # topk两个返回值，第一个值是相应的value，第二个值是对应的index，因此这里取第二个返回值
            # [bs,900] 在各个image中选出最大的topk个
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq
            # gather boxes
            # 取出对应index的四个坐标值出来
            # [bs,900,4]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            # 参考点位脱离
            refpoint_embed_ = refpoint_embed_undetach.detach()
            # 同样的取出对应的初始点位的值
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

            # gather tgt
            # 同样的取出对应的memory
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

            # 对应于论文中图5 c的 static content queries
            if self.embed_init_tgt:
                # [900,256] -> [900,bs,256] -> [bs,900,256], 这里依然使用的是网络的参数 一个mebedding
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:

                tgt_ = tgt_undetach.detach()

            # refpoint_embed 是prepare_for_cdn生成的, 带有噪声的gt, tgt也是prepare_for_cdn生成的
            # 带 _ 后缀的这两个是上面处理的，利用encoder的output topk筛选的
            if refpoint_embed is not None:
                # cat prepare_for_cdn生成的去噪的，加上match部分使用的
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                # cat prepare_for_cdn生成的去噪的，加上match部分使用的
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                # 这种是推理模式，没有去噪的内容, 传入的参数refpoint_embed和tgt也就没有使用
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':

            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model

            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:

                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)

                tgt = torch.cat([tgt, tgt_], dim=1)
            else:

                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)

                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)

                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries,
                                                                             1)  # 1, n_q*n_pat, d_model

                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 
        ######################################################### 


        #########################################################
        # Begin Decoder
        #########################################################
        # references比hs多一个，hs是6，references是7   hs [bs,all, 256] references [bs,all, 4]
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            # 参考点位
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            # prepare_for_cdn时生成的
            tgt_mask=attn_mask)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model  nq是num query 如900+200=1100
        # references: n_dec+1, bs, nq, query_dim  query_dim=4 就是xywh
        #########################################################


        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':

            if self.two_stage_keep_all_tokens:
                # 没有经过topk筛选的
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)


                init_box_proposal = output_proposals

            else:
                # [1,bs,900,256] tgt_undetach是encoder的输出memory经过topk选取后的
                hs_enc = tgt_undetach.unsqueeze(0)
                # [1,bs,900,4] refpoint_embed_undetach是encoder的输出经过topk选取后的
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        # init_box_proposal 最最初始的参考点位
        # hs_enc, ref_enc 这两个是encoder的输出经过topk筛选的
        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers,
                 norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # spatial_shapes [特征层数,2] valid_ratios [bs,特征层数,2]
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成网格点,从0.5开始 到 减掉一个0.5
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            # 坐标进行缩放 valid_ratios[:, None, lvl, 1] * H_是在H_基础上进一步缩减范围
            # reshape(-1) 拉平会变成一维的 shape=hw，[None]，会在最前面加上一个1维度 -> [1,hw] -> [2,hw]
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [bs,hw]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # [bs,hw,2]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 所有特征层的参考点拼在一起 [bs,all hw,2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points[:,:,None] -> [2,all hw,1,2]
        # valid_ratios[:,None] -> [bs,1,特征层数量,2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # [2,all hw,4,2]
        return reference_points

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None
                ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        # preparation and reshape

        if self.num_layers > 0:

            if self.deformable_encoder:
                #  获取参考点，encoder的参考点是生成的grid，Deformable DETR中的方法
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []

        intermediate_ref = []
        # 这个传进来的值就是None
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        # encoder layer的循环
        for layer_id, layer in enumerate(self.layers):
            # main process

            dropflag = False
            # 默认是None
            if self.enc_layer_dropout_prob is not None:

                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:

                if self.deformable_encoder:
                    # encoder layer 是与Deformable DETR相同的
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:
                    # 正常的attention
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)

            # two_stage_type 默认是standard
            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                # 在每一层encoder都进行topk proposal的选择

                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)

                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                # gather boxes
                topk = self.num_queries

                enc_outputs_class = self.class_embed[layer_id](output_memory)

                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # bs, nq

                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory
            # ref_token_index 默认是None
            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))

                intermediate_output.append(out_i)

                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:

            intermediate_output = torch.stack(intermediate_output)  # n_enc/n_enc-1, bs, \sum{hw}, d_model

            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,

                ):
        """
        Input:
            - tgt: nq, bs, d_model [900+2*100,bs,256]
            - memory: hw, bs, d_model [sum(hw),bs,256]
            - pos: hw, bs, d_model [sum(hw),bs,256]
            - refpoints_unsigmoid: nq, bs, 2/4 [900+2*100,bs,4]
            - valid_ratios/spatial_shapes: bs, nlevel, 2  [bs,4,2],[4,2] 4=feature level
        """
        output = tgt
        # 每一层decoder计算后的结果
        intermediate = []
        # [900+2*100,bs,4] 限制在0-1
        reference_points = refpoints_unsigmoid.sigmoid()
        # 有一个初始的点位，以及每一层decoder计算后，进行修正后的结果
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            # 对box添加随机的扰动噪声
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:

                if reference_points.shape[-1] == 4:
                    # [N,bs,1,4]*[1,bs,4,4] -> [N,bs,4,4]
                    reference_points_input = reference_points[:, :, None] \
                                             * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2

                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                # get sine embedding for the query vector -> [N,bs,512]
                # xywh的高频位置编码
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2
            else:

                query_sine_embed = gen_sineembed_for_position(reference_points)  # nq, bs, 256*2

                reference_points_input = None

            # conditional query [N,bs,256]
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            # conditional detr的做法
            query_pos = pos_scale * raw_query_pos
            # 如果不使用变形attention
            if not self.deformable_decoder:
                # 这里就是conditional detr的那个乘法
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # 如果使用了变形attention 就不执行hw的调制了，todo 论文中好像没有提到过
            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                # DAB-DETR的部分
                # 结构图中第二行的MLP的右边那个MLP
                # 公式7的Wref,Href [300,bs,2]
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                # 公式6的Xref
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(
                    -1)
                # 公式6的Yref
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(
                    -1)

            # 随机的跨过某些decoder layer，并不处理，直接进入下一层
            # random drop some layers if needed
            dropflag = False

            if self.dec_layer_dropout_prob is not None:

                prob = random.random()

                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                # [N,bs,256]
                output = layer(
                    tgt=output,  # [N,bs,256]
                    tgt_query_pos=query_pos,  # [N,bs,256]
                    tgt_query_sine_embed=query_sine_embed,  # [N,bs,512]
                    tgt_key_padding_mask=tgt_key_padding_mask,  # None
                    tgt_reference_points=reference_points_input,  # [N,bs,4,4]

                    memory=memory,  # [sum(hw),bs,256]
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,  # [level,2]
                    memory_pos=pos,  # [sum(hw),bs,256]
                    # prepare_for_cdn时生成的
                    self_attn_mask=tgt_mask,  # [N,N]
                    # 默认为None，调用时并未传入
                    cross_attn_mask=memory_mask  # None
                )

            # iter update
            if self.bbox_embed is not None:
                # 得到特征图上的值
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                # 得到网络输出的修正值
                delta_unsig = self.bbox_embed[layer_id](output)
                # 进行修正
                outputs_unsig = delta_unsig + reference_before_sigmoid
                # 限制在0-1
                new_reference_points = outputs_unsig.sigmoid()

                # select # ref points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:

                    nq_now = new_reference_points.shape[0]

                    select_number = self.dec_layer_number[layer_id + 1]

                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)  # nq, bs, 91

                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]  # new_nq, bs

                        new_reference_points = torch.gather(new_reference_points, 0,
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid

                if self.rm_detach and 'dec' in self.rm_detach:

                    reference_points = new_reference_points
                else:
                    # 脱离
                    reference_points = new_reference_points.detach()

                if self.use_detached_boxes_dec_out:
                    # 这里是脱离的
                    ref_points.append(reference_points)
                else:
                    # 这个是没有脱离的，这个地方的处理就是论文钟的look forward twice
                    ref_points.append(new_reference_points)

            # 中间层的值
            intermediate.append(self.norm(output))

            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:

                if nq_now != select_number:
                    output = torch.gather(output, 0,
                                          topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # unsigmoid

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],  # 将bs维度放在前面
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]  # 将bs维度放在前面
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 # 下面这三个参数是Deformable DETR中encoder layer没有的
                 # 这两个参数如果都是False，那么EncoderLayer的逻辑是与Deformable-DETR是相同的
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        # self attention
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                      used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        # 比Deformable DETR的encoder多的部分
        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ):
        super().__init__()
        self.module_seq = module_seq
        # 固定是这个顺序，不过这个是进行的sort之后的排序
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        # cross attention
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                       used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # 正常的ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                   # for tgt
                   tgt: Optional[Tensor],  # nq, bs, d_model
                   tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   # 未使用
                   tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                   # 未使用
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                   # for memory
                   memory: Optional[Tensor] = None,  # hw, bs, d_model
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,  # num_levels
                   memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                   # 未使用
                   memory_pos: Optional[Tensor] = None,  # pos for memory

                   # sa
                   self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                   # 未使用
                   cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                   ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                # 正常的torch中的attention,没有在使用Conditional DETR中的那种attention
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)

            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                      tgt_reference_points.transpose(0, 1).contiguous(),
                                      memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                                      memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                   # for tgt
                   tgt: Optional[Tensor],  # nq, bs, d_model
                   tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   # 未使用
                   tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                   # 未使用
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                   # for memory
                   memory: Optional[Tensor] = None,  # hw, bs, d_model
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,  # num_levels
                   memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2

                   # 未使用
                   memory_pos: Optional[Tensor] = None,  # pos for memory

                   # sa
                   # 未使用
                   self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                   # 未使用
                   cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                   ):

        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':

                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':

                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))

        # Deformable DETR的cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                # for tgt
                # [N,bs,256]
                tgt: Optional[Tensor],  # nq, bs, d_model
                # [N,bs,256]
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                # [N,bs,512]
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                # None
                tgt_key_padding_mask: Optional[Tensor] = None,
                # [N,bs,4,4]
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                # [sum(hw),bs,256]
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                # [bs,sum(hw)]
                memory_key_padding_mask: Optional[Tensor] = None,
                # (level,)
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                # [level,2]
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                # [sum(hw),bs,256]
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                # [N,N] prepare_for_cdn制作的mask，仅是给self_attention使用的
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                # None
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):

        # module_seq是固定的['sa', 'ca', 'ffn']
        # 就是先进行self_attention, 然后cross_attention, 然后ffn
        for funcname in self.module_seq:
            if funcname == 'ffn':

                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                # ca和sa的参数是相同的，也是为了通用
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':

                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out
    )
