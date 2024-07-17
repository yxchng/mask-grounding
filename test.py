
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.multimodal_bert import MultiModalBert
import torchvision

from lib import multimodal_segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

from modeling.MaskFormerModel import MaskFormerHead
from addict import Dict
from bert.modeling_bert import BertLMPredictionHead, BertEncoder




def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                output = model(image, sentences[:, :, j], attentions[:, :, j])
                mask_cls_results = output["pred_logits"]
                mask_pred_results = output["pred_masks"]

                target_shape = target.shape[-2:]
                mask_pred_results = F.interpolate(mask_pred_results, size=target_shape, mode='bilinear', align_corners=True)

                pred_masks = model.semantic_inference(mask_cls_results, mask_pred_results)                
                output = pred_masks[0]

                output = output.cpu()
                output_mask = (output > 0.5).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

class WrapperModel(nn.Module):
    def __init__(self, image_model, language_model, classifier, args) :
        super(WrapperModel, self).__init__()
        self.image_model = image_model
        self.language_model = language_model
        self.classifier = classifier
        self.lang_proj = nn.Linear(768,256)

    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=1)[...,1:]
        mask_pred = mask_pred.sigmoid()      
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
        return semseg

    def forward(self, image, sentences, attentions): 
        input_shape = image.shape[-2:]
        l_mask = attentions.unsqueeze(dim=-1)

        i0, Wh, Ww = self.image_model.forward_stem(image)
        l0, extended_attention_mask = self.language_model.forward_stem(sentences, attentions)

        i1 = self.image_model.forward_stage1(i0, Wh, Ww)
        l1 = self.language_model.forward_stage1(l0, extended_attention_mask)
        i1_residual, H, W, i1_temp, Wh, Ww  = self.image_model.forward_pwam1(i1, Wh, Ww, l1, l_mask)
        l1_residual, l1 = self.language_model.forward_pwam1(i1, l1, extended_attention_mask) 
        i1 = i1_temp

        i2 = self.image_model.forward_stage2(i1, Wh, Ww)
        l2 = self.language_model.forward_stage2(l1, extended_attention_mask)
        i2_residual, H, W, i2_temp, Wh, Ww  = self.image_model.forward_pwam2(i2, Wh, Ww, l2, l_mask)
        l2_residual, l2 = self.language_model.forward_pwam2(i2, l2, extended_attention_mask) 
        i2 = i2_temp

        i3 = self.image_model.forward_stage3(i2, Wh, Ww)
        l3 = self.language_model.forward_stage3(l2, extended_attention_mask)
        i3_residual, H, W, i3_temp, Wh, Ww  = self.image_model.forward_pwam3(i3, Wh, Ww, l3, l_mask)
        l3_residual, l3 = self.language_model.forward_pwam3(i3, l3, extended_attention_mask) 
        i3 = i3_temp

        i4 = self.image_model.forward_stage4(i3, Wh, Ww)
        l4 = self.language_model.forward_stage4(l3, extended_attention_mask)
        i4_residual, H, W, i4_temp, Wh, Ww  = self.image_model.forward_pwam4(i4, Wh, Ww, l4, l_mask)
        l4_residual, l4 = self.language_model.forward_pwam4(i4, l4, extended_attention_mask) 
        i4 = i4_temp

        outputs = {}
        outputs['s1'] = i1_residual
        outputs['s2'] = i2_residual
        outputs['s3'] = i3_residual
        outputs['s4'] = i4_residual

        predictions, _ = self.classifier(outputs)
        return predictions

def main(args):
    device = 'cuda'
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                  sampler=test_sampler, num_workers=args.workers)
    image_model = multimodal_segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')

    if args.model != 'lavt_one':
        language_model = MultiModalBert.from_pretrained(args.ck_bert, embed_dim=image_model.backbone.embed_dim)
        if args.ddp_trained_weights:
            language_model.pooler = None
    else:
        bert_model = None

    input_shape = dict()
    input_shape['s1'] = Dict({'channel': 128,  'stride': 4})
    input_shape['s2'] = Dict({'channel': 256,  'stride': 8})
    input_shape['s3'] = Dict({'channel': 512,  'stride': 16})
    input_shape['s4'] = Dict({'channel': 1024, 'stride': 32})



    cfg = Dict()
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.0 
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 4
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["s1", "s2", "s3", "s4"]

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 10
    cfg.MODEL.MASK_FORMER.PRE_NORM = False


    maskformer_head = MaskFormerHead(cfg, input_shape)

    model = WrapperModel(image_model.backbone, language_model, maskformer_head, args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
