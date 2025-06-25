import argparse
import os
import sys
import re
import numpy as np
import json
import torch
from PIL import Image
import torchvision

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))





def calculate_iou(mask1, mask2):
    """
    计算两个二值掩码的 IoU（Intersection over Union）

    参数:
    mask1 (numpy.ndarray): 第一个二值掩码，形状为 (height, width)
    mask2 (numpy.ndarray): 第二个二值掩码，形状为 (height, width)

    返回:
    float: 计算得到的 IoU 值
    """
    # 计算交集 (Intersection)
    intersection = torch.sum(torch.logical_and(mask1, mask2))
    
    # 计算并集 (Union)
    union = torch.sum(torch.logical_or(mask1, mask2))
    
    # 计算 IoU
    iou = intersection / union if union != 0 else 0.0  # 防止除零错误
    return iou

def calculate_iouA(mask1, mask2):
    """
    计算两个二值掩码的 IoU（Intersection over Union）

    参数:
    mask1 (numpy.ndarray): 第一个二值掩码，形状为 (height, width)
    mask2 (numpy.ndarray): 第二个二值掩码，形状为 (height, width)

    返回:
    float: 计算得到的 IoU 值
    """
    # 计算交集 (Intersection)
    intersection = torch.sum(torch.logical_and(mask1, mask2))
    
    # 计算并集 (Union)
    union = torch.sum(mask1)
    
    # 计算 IoU
    iou = intersection / union if union != 0 else 0.0  # 防止除零错误
    return iou

def calculate_iouB(mask1, mask2):
    """
    计算两个二值掩码的 IoU（Intersection over Union）

    参数:
    mask1 (numpy.ndarray): 第一个二值掩码，形状为 (height, width)
    mask2 (numpy.ndarray): 第二个二值掩码，形状为 (height, width)

    返回:
    float: 计算得到的 IoU 值
    """
    # 计算交集 (Intersection)
    intersection = torch.sum(torch.logical_and(mask1, mask2))
    
    # 计算并集 (Union)
    union = torch.sum(mask2)
    
    # 计算 IoU
    iou = intersection / union if union != 0 else 0.0  # 防止除零错误
    return iou

def compute_weighted_iou_loss(masks, scores):
    """
    计算每个掩码与其他掩码的加权 IoU 损失并进行求和

    参数:
    masks (list of numpy.ndarray): 一个包含多个二值掩码的列表，每个掩码的形状应相同
    scores (list of float): 一个包含每个掩码对应的置信度分数的列表

    返回:
    list: 每个掩码与其他掩码的加权 IoU 损失之和的列表
    """
    num_masks = len(masks)
    loss_list = []

    # 对每个掩码计算与其他掩码的加权 IoU 损失并求和
    for i in range(num_masks):
        mask_i = masks[i]
        score_i = scores[i]
        total_loss = 0.0
        for j in range(num_masks):
            if i != j:
                mask_j = masks[j]
                # iou = calculate_iou(mask_i, mask_j)+0.5*calculate_iouA(mask_i, mask_j)+0.5*calculate_iouB(mask_i, mask_j)
                iou = calculate_iou(mask_i, mask_j)
                # 计算 IoU 损失
                weighted_loss = iou * score_i*scores[j]  # 加权损失
                total_loss += weighted_loss
                #total_loss += iou
        loss_list.append(total_loss)

    return loss_list

def get_best_mask(masks, scores):
    """
    根据加权 IoU 损失对掩码进行排序，返回得分最高的掩码

    参数:
    masks (list of numpy.ndarray): 一个包含多个二值掩码的列表，每个掩码的形状应相同
    scores (list of float): 一个包含每个掩码对应的置信度分数的列表

    返回:
    numpy.ndarray: 得分最高的掩码
    """
    # 计算每个掩码的加权 IoU 损失
    #print('masks',len(masks))
    # 计算每个掩码的面积（以像素为单位）
    ans = masks[0].shape[1]*masks[0].shape[2]
    #print('ans',ans)
    # 去掉mask中面积大于0.3*ans的
    masks = [mask for mask, score in zip(masks, scores) if mask.sum()<= 0.15*ans]
    # print('masks',len(masks))
    # print('mask',masks)
    scores = [score for mask, score in zip(masks, scores) if mask.sum()<= 0.15*ans]
    if len(masks)>4:
       masks=masks[1:-1]
       scores=scores[1:-1]
    weighted_iou_losses = compute_weighted_iou_loss(masks, scores)

    # 根据损失值进行排序（损失值越小，掩码越好）
    sorted_indices = torch.argsort(torch.tensor(weighted_iou_losses))  
    best_mask_index = sorted_indices[-1]  # 得分最高的掩码的索引

    # 返回得分最高的掩码
    return masks[best_mask_index]
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    加载并预处理图像
    Args:
        image_path: 图像文件的路径
    Returns:
        image_pil: 原始PIL图像对象
        image: 预处理后的张量
    """
    # 使用PIL加载图像并转换为RGB格式
    image_pil = Image.open(image_path).convert("RGB")  # load image

    # 定义图像预处理流程
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),    # 将图像调整到高度为800，最大宽度为1333
            T.ToTensor(),                            # 将PIL图像转换为张量
            T.Normalize([0.485, 0.456, 0.406],       # 使用ImageNet数据集的均值进行标准化
                       [0.229, 0.224, 0.225]),       # 使用ImageNet数据集的标准差进行标准化
        ]
    )
    # 应用预处理转换
    # 返回预处理后的图像张量和None（因为这里不需要目标框）
    image, _ = transform(image_pil, None)  # 3, h, w  # 输出张量形状为[3通道, 高度, 宽度]
    
    return image_pil, image  # 返回原始图像和预处理后的图像张量


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    """
    加载和初始化模型
    Args:
        model_config_path: 模型配置文件路径
        model_checkpoint_path: 模型检查点文件路径
        bert_base_uncased_path: BERT模型路径
        device: 运行设备（CPU/GPU）
    Returns:
        model: 加载好的模型
    """
    # 从配置文件加载模型参数
    args = SLConfig.fromfile(model_config_path)
    # 设置运行设备
    args.device = device
    # 设置BERT模型路径
    args.bert_base_uncased_path = bert_base_uncased_path
    
    # 根据配置构建模型
    model = build_model(args)
    
    # 加载模型检查点
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    # 清理并加载模型状态字典，strict=False允许部分权重不匹配
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # 打印加载结果，显示哪些权重成功加载
    print(load_res)
    
    # 将模型设置为评估模式
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """
    获取图像中物体的定位输出
    Args:
        model: 加载的模型
        image: 预处理后的图像张量
        caption: 描述文本
        box_threshold: 边界框置信度阈值
        text_threshold: 文本匹配置信度阈值
        with_logits: 是否在输出中包含置信度分数
        device: 运行设备
    Returns:
        boxes_filt: 过滤后的边界框坐标
        pred_phrases: 预测的短语列表
    """
    # 预处理文本描述
    caption = caption.lower()                # 转换为小写
    caption = caption.strip()                # 去除首尾空格
    if not caption.endswith("."):            # 确保文本以句号结尾
        caption = caption + "."
    #print('caption', caption)
    # 将模型和图像移动到指定设备
    model = model.to(device)
    image = image.to(device)
    
    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256) 预测的置信度分数
    boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4) 预测的边界框坐标
    logits.shape[0]

    # 根据置信度阈值过滤输出
    logits_filt = logits.clone()                       # 复制置信度张量
    boxes_filt = boxes.clone()                         # 复制边界框张量
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold  # 创建过滤掩码
    logits_filt = logits_filt[filt_mask]              # 过滤置信度 num_filt, 256
    boxes_filt = boxes_filt[filt_mask]                # 过滤边界框 num_filt, 4
    logits_filt.shape[0]

    # 获取文本短语
    tokenlizer = model.tokenizer                      # 获取分词器
    tokenized = tokenlizer(caption)                   # 对描述文本进行分词
    
    # 构建预测结果
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        # 根据文本阈值获取预测短语
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            # 如果需要，添加置信度分数
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    """
    在给定的轴上显示分割掩码
    Args:
        mask: 分割掩码张量
        ax: matplotlib轴对象
        random_color: 是否使用随机颜色
    """
    if random_color:
        # 生成随机RGB颜色和0.6的透明度
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # 使用默认的蓝色，RGB值为(30, 144, 255)，透明度0.6
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]                                   # 获取掩码的高度和宽度
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # 将掩码转换为彩色图像
    ax.imshow(mask_image)                                   # 显示掩码图像


def show_box(box, ax, label):
    """
    在给定的轴上显示边界框和标签
    Args:
        box: 边界框坐标 [x0, y0, x1, y1]
        ax: matplotlib轴对象
        label: 边界框的标签文本
    """
    x0, y0 = box[0], box[1]                                # 获取左上角坐标
    w, h = box[2] - box[0], box[3] - box[1]               # 计算宽度和高度
    # 添加矩形框，绿色边框，无填充
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)                                # 添加标签文本


def save_mask_data(output_dir, mask_list, box_list, label_list):
    print('label_list', label_list)
    """
    保存分割掩码数据到图像和JSON文件
    Args:
        output_dir: 输出目录
        mask_list: 分割掩码列表
        box_list: 边界框列表
        label_list: 标签列表
    """
    value = 0  # 背景值为0

    # 创建掩码图像，每个对象使用不同的值
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        break
    print('mask_img', mask_img.shape)
    # 保存掩码图像
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # # 准备JSON数据
    # json_data = [{
    #     'value': value,
    #     'label': 'background'
    # }]
    # # 为每个对象添加数据
    # for label, box in zip(label_list, box_list):
    #     value += 1
    #     name, logit = label.split('(')                    # 分离标签名称和置信度
    #     logit = logit[:-1]                                # 移除最后的括号
    #     json_data.append({
    #         'value': value,                               # 掩码中的像素值
    #         'label': name,                                # 对象标签
    #         'logit': float(logit),                        # 置信度分数
    #         'box': box.numpy().tolist(),                  # 边界框坐标
    #     })
    # # 保存JSON数据
    # with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
    #     json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=False, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=False, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=False, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # 配置参数
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'               # 模型配置文件路径
    grounded_checkpoint = 'groundingdino_swint_ogc.pth'  # Grounded-DINO模型检查点路径
    sam_version = args.sam_version          # SAM模型版本
    sam_checkpoint = 'sam_vit_h_4b8939.pth'    # SAM模型检查点路径
    sam_hq_checkpoint = args.sam_hq_checkpoint  # SAM-HQ模型检查点路径
    use_sam_hq = args.use_sam_hq           # 是否使用SAM-HQ
    
    box_threshold = 0.3     # 边界框置信度阈值
    text_threshold = 0.25   # 文本匹配置信度阈值
    device = 'cuda'                   # 运行设备
    bert_base_uncased_path = args.bert_base_uncased_path  # BERT模型路径

    
    # 加载Grounded-DINO模型
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    # 初始化SAM模型
    if use_sam_hq:
        # 使用高质量版本的SAM
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        # 使用标准版本的SAM
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # 遍历读取 'data/waldo_kitchen/json' 目录下的所有 JSON 文件
    data_dir = 'data/teatime'

    # 创建输出目录
    output_dir = 'data/teatime/mask1'
    black_mask=torch.zeros((1296, 968), dtype=torch.uint8)
              # 输出目录 
    test=[6,24,60,65,81,119,128]
    # test=[41,105,152,195]
    # test=[2,25,43,107,129,140]
    # test=[53,66,89,140,154]
    test=[140]
    # test=[83,97,146,179]
    os.makedirs(output_dir, exist_ok=True)
    for k in range(0,164):  # 遍历 frame_00001 到 frame_00190
        if k not in test:
            continue
        json_file_path = os.path.join(data_dir, f'json/frame_{k:05d}.json')
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        output_dir_i=os.path.join(output_dir, f'frame_{k:05d}')
        os.makedirs(output_dir_i, exist_ok=True)
        objects=data.get('object',[])
        for obj in objects:
            if 1:
                if obj['category']=='tea glass':
                    image_path=os.path.join(data_dir, f'images/frame_{k:05d}.jpg')
                    if not os.path.exists(image_path):
                        print(f"图片不存在: {image_path}")
                        break
                    all_boxes=torch.tensor([[0,0,0,0]])
                    all_phrases=[0]
                    l=0
                    for sentence in obj['sentence']:
                        text_prompt=sentence.replace(',', '')
                        # 加载并预处理图像
                        image_pil, image = load_image(image_path)
                        # 运行Grounded-DINO模型进行目标检测
                        boxes_filt, pred_phrases = get_grounding_output(
                            model, image, text_prompt, box_threshold, text_threshold, device=device
                        )
                        scores = [float(re.search(r'\((\d+\.\d+)\)', label).group(1)) for label in pred_phrases]

                        all_boxes=torch.cat((all_boxes, boxes_filt), dim=0)
                        all_phrases.extend(scores)
                    #print('all_boxes', all_boxes)
                    boxes_filt=all_boxes
                    phrases=all_phrases
                    #print('boxes_filt', boxes_filt)
                    
                    # 读取并转换图像格式
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)

                    # 调整边界框坐标到原始图像尺寸
                    size = image_pil.size
                    H, W = size[1], size[0]
                    for i in range(boxes_filt.size(0)):
                        # 将归一化坐标转换为像素坐标
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        # 将中心点坐标转换为左上角坐标
                        boxes_filt[i][:2] = boxes_filt[i][:2] - boxes_filt[i][2:] / 2
                        # 计算右下角坐标
                        boxes_filt[i][2:] = boxes_filt[i][2:] + boxes_filt[i][:2]
                        # 确保结果为整数类型
                        boxes_filt[i] = boxes_filt[i].long()

                    # 转换边界框格式以适应SAM模型
                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                    # 使用SAM模型生成分割掩码
                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    # # draw output image
                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(image)
                    # for mask in masks:
                    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    # for box, label in zip(boxes_filt, phrases):
                    #     show_box(box.numpy(), plt.gca(), label)

                    # plt.axis('off')
                    # plt.savefig(
                    #     os.path.join(output_dir_i, f'{obj["category"]}_mask.png'),
                    #     bbox_inches="tight", dpi=300, pad_inches=0.0
                    # )
                    #print('masks',len(masks))
                    masks = masks.bool()
                    s=torch.argsort(torch.tensor(phrases))
                    select_mask=masks[s[-1]]
                    #select_mask=get_best_mask(masks, phrases)
                    #将select_mask保存为二值掩码单通道图像
                    #print('select_mask', select_mask.sum())
                    save_path=os.path.join(output_dir_i, f'{obj["category"]}.png')
                    select_mask = select_mask.float().cpu().numpy().squeeze(0)
                    select_mask = (select_mask * 255).astype(np.uint8)
                    img = Image.fromarray(select_mask, mode='L')
                    img.save(save_path)
                
                #print('select_mask', select_mask.shape)
                #torchvision.utils.save_image(select_mask.float().cpu(), save_path)
                #print('save_path:', select_mask.shape)
                #import pdb; pdb.set_trace()
                #plt.imsave(save_path, masks[0].squeeze(0).cpu().numpy(), cmap='gray', format='png', bits=8)
            else:
                save_path=os.path.join(output_dir_i, f'{obj["category"]}.png')
                img = Image.fromarray(black_mask.float().cpu().numpy(), mode='L')
                img.save(save_path)
                #torchvision.utils.save_image(black_mask.unsqueeze(0).float(), save_path)
                #plt.imsave(save_path, black_mask, cmap='gray', format='png', bits=8)
    # 保存分割结果数据
    #save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
