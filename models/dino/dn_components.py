# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        在传给transformer之前的预处理
        方法参数个数与DN-DETR不同
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.

        dn_args与dn-detr的不同了，少了一个pattern参数
        第二个参数在dn-detr中的意义是group的数量，在这里是总的噪声对的数量
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        这个Embedding在DN-DETR中的维度是255，在这里是256
        在DN-DETR中 255+1=256, 加上的最后一位是0或者1，1标识这是个噪声
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        # 与DN-DETR的 dn args有一点区别，少了group的参数（第二个参数在DN-DETR中是group，这里是dn_number)  和 num_patterns的参数
        #
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        # 正负样本数量相同
        dn_number = dn_number * 2
        # bs大小的list，item是全是1的tensor，size是各个image gt的数量
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        # 各个image gt的数量
        known_num = [sum(k) for k in known]

        # 这个地方与dn不同
        # 没有gt
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                # 有点类似于dn-detr的group的处理
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        # 以下的这些处理与DN-DETR基本一致
        # 所有的1 cat到一起
        unmask_bbox = unmask_label = torch.cat(known)
        # 取出所有gt的label值
        labels = torch.cat([t['labels'] for t in targets])
        # 取出所有gt的box
        boxes = torch.cat([t['boxes'] for t in targets])
        # 标识属于哪个图片 like tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2], device='cuda:0')
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        # 返回一个二维张量，其中每一行都是非零值的索引
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        # 拉平
        known_indice = known_indice.view(-1)
        # 并没有使用
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        # 这三项是gt相关的值
        # (all gt*2*dn_number) 这里的N是bs中所有的gt的数量总和
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        # (all gt*2*dn_number)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        # [all gt*2*dn_number, 4]
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        # 这两个是克隆的
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        # --------------------------------------------------------------------------------------------------------------

        if label_noise_ratio > 0:
            # 随机值，0-1内
            p = torch.rand_like(known_labels_expaned.float())
            # 被选择的id 这里比dn-detr 多了一个 1/2
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            # 给被选择的gt 一个随机的label id
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            # 把上面的值塞进去
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        # bs中最多的target的数量
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        # [dn_number,gt count]  [组数, gt count]
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        # 加上了group间的偏移量
        # += like tensor([[  0],
        #         [ 40],
        #         [ 80],
        #         [120],
        #         [160],
        #         [200],
        #         [240],
        #         [280],
        #         [320],
        #         [360]], device='cuda:0')

        #
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        # [gt count*dn number]，推平
        positive_idx = positive_idx.flatten()
        # [gt count*dn number] 正好剩下的位置是留给negative的
        negative_idx = positive_idx + len(boxes)

        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # 中心坐标减掉宽高的一半，左上的边界点
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            # 中心坐标加上宽高的一半，右下的边界点
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            # 宽高的一半放到中心坐标的位置(占位,为了计算使用)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            # 宽高是宽高的一半
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            # torch.randint_like(known_bboxs, low=0, high=2) 选出的值都是0 1这两种值
            # torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 选出的值是-1 或者 1
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            # rand_part 0-1内的值 [2*gt count*dn number]
            rand_part = torch.rand_like(known_bboxs)

            # 负样本位置的值+1, 负样本的偏离比正样本更多
            rand_part[negative_idx] += 1.0
            # [2*gt count*dn number,4] 坐标位置随机的乘上1 -1
            rand_part *= rand_sign
            # 加上随机的偏移，左上，右下的点随机的进行了偏移
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            # 裁剪，防止溢出
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # 左上和右下点的和除2就是中心点坐标
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            # 右下减去左上的差值，就是高宽
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # 这里的known_labels_expaned 已经被添加过随机的噪声了
        # (2*gt count*dn_number) 新的label信息
        m = known_labels_expaned.long().to('cuda')
        # [2*gt count*dn_number, 256]  将label tensor传入label_enc的embedding得到编码后的值
        input_label_embed = label_enc(m)
        # 对坐标取反函数  对应于特征图上的坐标
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        # [pad_size, 256]
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        # [pad_size, 4]
        padding_bbox = torch.zeros(pad_size, 4).cuda()
        # 重复bs [pad_size,256] -> [bs,pad_size,256]
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        # 重复bs [pad_size,4] -> [bs, pad_size,4]
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        # 如果有gt的话
        if len(known_num):
            # 各个image的合并在一起了 like tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            # 加上了偏移，这个偏移是这些batch中最大的gt的数量
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()

        # known_bid 标识属于哪个图片的
        if len(known_bid):
            # 替换对应的embed input_query_label第一个维度是bs，known_bid是标识的属于哪一个image
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed

            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # 这里pad_size就是cdn总共的数量，包括了正负样本，num_queries是正常的query的数量
        tgt_size = pad_size + num_queries

        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0

        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True

        # 各个组的掩码
        # reconstruct cannot see each other
        for i in range(dn_number):
            # 第一组
            if i == 0:
                # 看不到他后面的所有
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            # 最后一组
            if i == dn_number - 1:
                # 看不到他前面的所有
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                # 中间组
                # 看不到他后面的
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                # 也看不到他前面的
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        # 返回值比DN-DETR少了很多
        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,  # 多少组
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
    # 这里并不包含正常match部分的tgt，这点与DN-DETR的实现不同了
    return input_query_label, input_query_bbox, attn_mask, dn_meta


# 这里与DN-DETR的处理基本一致，只多了一个aux_loss的处理
def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        transformer处理之后的后处理
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        # 前面的这些是去噪的部分
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        # 后面这些是正常的匹配预测部分
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        # output_known_lbs_bboxes 内容是去噪部分的
        dn_meta['output_known_lbs_bboxes'] = out
    # 返回这俩还是网络自己预测的，不包括去噪部分的
    return outputs_class, outputs_coord
