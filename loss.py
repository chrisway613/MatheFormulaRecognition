import torch
import torch.nn as nn

import math


def get_class_balanced_cross_entropy(gt_score, pred_score):
    # 请写出类平衡交叉熵的loss
    # shape of score: (N, 1, 1/4h, 1/4w)

    # for classes balance
    # weight of positive sample
    beta = 1 - torch.sum(gt_score) / gt_score.numel()
    # print("sum gt score:", torch.sum(gt_score))
    # print("gt score numel:", gt_score.numel())
    # print("beta shape:", beta.shape)

    '''(N, 1, 1/4h, 1/4w) -> (N, 1/4h*1/4w)'''
    batch_size = pred_score.size(0)
    pred_score = pred_score.squeeze(1).view(batch_size, -1)
    gt_score = gt_score.squeeze(1).view(batch_size, -1)
    # print("pred_score shape:", pred_score.shape)
    # print("gt_score shape:", gt_score.shape)

    # -beta*y*log(y^hat) - (1-beta)*(1-y)log(1-y^hat)
    score_loss_batch = -beta * gt_score * torch.log(pred_score) \
                       - (1 - beta) * (1 - gt_score) * torch.log(1 - pred_score)
    # mean first on each feature map in a batch,
    # then overage on this batch
    score_loss = torch.mean(score_loss_batch, dim=(1, 0))

    return score_loss


def get_geo_loss(gt_geo, pred_geo):
    # 写出d1, d2, d3, d4, 4个feature map的 iou_loss 和 angle_map 的 loss
    # shape of geo: (N, 5, 1/4h, 1/4w)

    batch_size = pred_geo.size(0)

    '''(N, 5, 1/4h, 1/4w) -> (N, 1/4h*1/4w, 5)'''
    pred_geo = pred_geo.view(batch_size, 5, -1)
    pred_geo = pred_geo.transpose(1, 2)
    gt_geo = gt_geo.view(batch_size, 5, -1)
    gt_geo = gt_geo.transpose(1, 2)

    '''shape: (N*1/4h*1/4w, 4)'''
    pred_d = pred_geo[:, :, :4].view(-1, 4)
    gt_d = gt_geo[:, :, :4].view(-1, 4)
    min_d = torch.min(pred_d, gt_d)

    '''shape: (N*1/4h*1/4w,)'''
    inter_area = torch.sum(min_d[:, :2], dim=1) * torch.sum(min_d[:, 2:], dim=1)
    pred_area = torch.sum(pred_d[:, :2], dim=1) * torch.sum(pred_d[:, 2:], dim=1)
    gt_area = torch.sum(gt_d[:, :2], dim=1) * torch.sum(gt_d[:, 2:], dim=1)
    union_area = pred_area + gt_area - inter_area

    # summary iou loss of a batch
    iou_loss_batch = -torch.log(inter_area / union_area)

    '''shape: (N*1/4h*1/4w,)'''
    pred_angle = pred_geo[:, :, -1].view(-1)
    gt_angle = gt_geo[:, :, -1].view(-1)

    # summary angle loss of a batch
    angle_loss_batch = 1. - torch.cos(pred_angle - gt_angle)

    # mean loss overage a batch
    iou_loss = torch.mean(iou_loss_batch)
    angle_loss = torch.mean(angle_loss_batch)

    return iou_loss, angle_loss


class Loss(nn.Module):
    def __init__(self, weight_angle=10):
        super(Loss, self).__init__()

        # weight of angle loss
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        # gt_score has been filtered corresponding the ignored map when it generated
        classify_loss = get_class_balanced_cross_entropy(gt_score, pred_score * (1 - ignored_map))
        iou_loss, angle_loss = get_geo_loss(gt_geo, pred_geo)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.
              format(classify_loss, angle_loss, iou_loss))

        return geo_loss + classify_loss


if __name__ == '__main__':
    gt_score = torch.randint(0, 2, (1, 1, 128, 128))
    print("Ground Truth Score:\n", gt_score, '\n')

    pred_score = torch.randn(1, 1, 128, 128)
    pred_score = torch.sigmoid(pred_score)
    print("Predicted Score:\n", pred_score, '\n')

    gt_geo = torch.randn(1, 5, 128, 128)
    gt_geo[:, :4, :, :] = torch.sigmoid(gt_geo[:, :4, :, :]) * 128
    gt_geo[:, 4, :, :] = (torch.sigmoid(gt_geo[:, 4, :, :]) - .5) * math.pi
    print("Ground Truth GEO:\n", gt_geo, '\n')

    pred_geo = torch.randn(1, 5, 128, 128)
    pred_geo[:, :-1, :, :] = torch.sigmoid(pred_geo[:, :-1, :, :]) * 128
    pred_geo[:, -1, :, :] = (torch.sigmoid(pred_geo[:, -1, :, :]) - .5) * math.pi
    print("Predicted GEO:\n", pred_geo, '\n')

    criterion = Loss()
    ignore_map = torch.zeros(pred_score.shape)
    loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignore_map)
    print("Loss:", loss)
