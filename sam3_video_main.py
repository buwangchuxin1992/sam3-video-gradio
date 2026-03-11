#!/usr/bin/env python3
"""
SAM3 Interactive Vision Studio - 增强版视频目标跟踪
基于 SAM3 的交互式图像分割与视频跟踪系统，支持点、框提示和多目标分割
"""

import os
import sys
import time
import io
import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
from pathlib import Path
import tempfile
import json
import logging
import zipfile
from datetime import datetime
# 修复中文编码问题 - 设置标准输入输出编码

# 添加当前目录到Python路径，以便导入sam3模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入SAM3相关模块
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import FindStage
    from sam3.visualization_utils import plot_results, visualize_formatted_frame_output, render_masklet_frame
    from sam3.model import box_ops
except ImportError as e:
    print(f"导入SAM3模块失败: {e}")
    print("请确保已正确安装SAM3依赖")
    sys.exit(1)

# 全局变量
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 添加MPS设备警告
if DEVICE == "mps":
    print("\n警告: 使用MPS设备可能会导致数值不稳定，建议使用CUDA设备。")
    print("更多信息: https://github.com/pytorch/pytorch/issues/84936")
    print("如果遇到问题，请尝试在终端运行: export PYTORCH_ENABLE_MPS_FALLBACK=1\n")

print(f"使用设备: {DEVICE}")

# 初始化模型
def initialize_models():
    """初始化SAM3图像和视频预测器"""
    try:
        # 检查模型文件是否存在
        # model_dir = current_dir / "models"
        # checkpoint_path = model_dir / "sam3.pt"
        bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        checkpoint_path = f"{current_dir}/checkpoints/sam3.pt"

        # print(f"111---------checkpoint_path:{checkpoint_path}---------")
        # print(f"111---------bpe_path:{bpe_path}---------")

        # if not checkpoint_path.exists():
        #     print(f"模型文件不存在: {checkpoint_path}")
        #     print("请下载SAM3模型文件到目录")
        #     return None, None
            
        # if not bpe_path.exists():
        #     print(f"BPE文件不存在: {bpe_path}")
        #     return None, None
            
        # 初始化图像模型
        # print("00")
        image_model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE
        )
        # print("01")


        # image_model = build_sam3_image_model(
        #     checkpoint_path=str(checkpoint_path),
        #     bpe_path=str(bpe_path),
        #     device=DEVICE
        # )
        
        # 创建图像处理器
        image_predictor = Sam3Processor(image_model, device=DEVICE)
        # print("02")
        # 初始化视频模型（关键修复：使用build_sam3_video_model）
        video_model = build_sam3_video_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=DEVICE
        )
        # print("03")
        # 获取视频预测器（关键修复：从video_model.tracker获取）
        video_predictor = video_model.tracker
        video_predictor.backbone = video_model.detector.backbone

        print("模型初始化成功")
        return image_predictor, video_predictor
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return None, None

# 全局预测器实例
image_predictor, video_predictor = initialize_models()

class VideoTrackingSession:
    """视频跟踪会话管理类"""
    def __init__(self):
        self.inference_state = None
        self.video_frames = []
        self.video_path = None
        self.prompts = {}  # 存储每个目标的提示信息
        self.next_obj_id = 1
        self.width = 0
        self.height = 0
        self.total_frames = 0
        
    def init_video(self, video_path):
        """初始化视频会话"""
        try:
            self.video_path = video_path
            self.video_frames = []
            
            # 读取视频帧
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if len(self.video_frames) == 0:
                return False, "无法读取视频帧"
                
            self.width = self.video_frames[0].shape[1]
            self.height = self.video_frames[0].shape[0]
            self.total_frames = len(self.video_frames)
            
            # 初始化推理状态（关键修复：使用video_predictor.init_state）
            # print("-----video_path:",video_path)
            self.inference_state = video_predictor.init_state(video_path=video_path)
            self.prompts = {}
            self.next_obj_id = 1
            
            return True, f"视频初始化成功，共{len(self.video_frames)}帧，分辨率{self.width}x{self.height}"
            
        except Exception as e:
            return False, f"视频初始化失败: {str(e)}"
    
    def add_point_prompt(self, frame_idx, x, y, is_positive=True, obj_id=None):
        """添加点提示"""
        try:
            # print("111")
            if obj_id is None:
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                
            if obj_id not in self.prompts:
                self.prompts[obj_id] = {"points": [], "labels": [], "boxes": []}
                
            # print("222")
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1 if is_positive else 0], np.int32)
            
            rel_points = [[x / self.width, y / self.height] for x, y in points]
            points_tensor = torch.tensor(rel_points, dtype=torch.float32)
            points_labels_tensor = torch.tensor(labels, dtype=torch.int32)
            


            _, out_obj_ids, low_res_masks, video_res_masks = video_predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_tensor,
                labels=points_labels_tensor,
                clear_old_points=False,
            )
            # print("333")

            self.prompts[obj_id]["points"].extend(points)
            self.prompts[obj_id]["labels"].extend(labels)
            # print("444")

            return True, f"目标{obj_id}: 添加{'正' if is_positive else '负'}点提示({x},{y})"
            
        except Exception as e:
            return False, f"添加点提示失败: {str(e)}"
    
    def add_box_prompt(self, frame_idx, x1, y1, x2, y2, obj_id=None):
        """添加框提示"""
        try:
            if obj_id is None:
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                
            if obj_id not in self.prompts:
                self.prompts[obj_id] = {"points": [], "labels": [], "boxes": []}
                
            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            rel_box = [[xmin / self.width, ymin / self.height, xmax / self.width, ymax / self.height] 
                      for xmin, ymin, xmax, ymax in box]
            rel_box = np.array(rel_box, dtype=np.float32)
            
            # 关键修复：使用video_predictor.add_new_points_or_box
            _, out_obj_ids, low_res_masks, video_res_masks = video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=rel_box,
            )
            
            self.prompts[obj_id]["boxes"].append(box[0])
            
            return True, f"目标{obj_id}: 添加框提示({x1},{y1},{x2},{y2})"
            
        except Exception as e:
            return False, f"添加框提示失败: {str(e)}"
    
    def refine_prompt(self, frame_idx, obj_id, points, labels):
        """精炼已有目标的提示"""
        try:
            if obj_id not in self.prompts:
                return False, f"目标{obj_id}不存在"
                
            rel_points = [[x / self.width, y / self.height] for x, y in points]
            points_tensor = torch.tensor(rel_points, dtype=torch.float32)
            points_labels_tensor = torch.tensor(labels, dtype=torch.int32)
            
            # 关键修复：使用video_predictor.add_new_points
            _, out_obj_ids, low_res_masks, video_res_masks = video_predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_tensor,
                labels=points_labels_tensor,
                clear_old_points=False,
            )
            
            self.prompts[obj_id]["points"].extend(points)
            self.prompts[obj_id]["labels"].extend(labels)
            
            return True, f"目标{obj_id}: 精炼提示成功"
            
        except Exception as e:
            return False, f"精炼提示失败: {str(e)}"
    
    def clear_prompts(self, obj_id=None):
        """清空提示"""
        try:
            if obj_id is None:
                # 清空所有提示
                video_predictor.clear_all_points_in_video(self.inference_state)
                self.prompts = {}
                self.next_obj_id = 1
                return True, "已清空所有提示"
            else:
                # 清空特定目标的提示
                if obj_id in self.prompts:
                    # 临时存储其他提示
                    temp_prompts = self.prompts.copy()
                    del temp_prompts[obj_id]
                    
                    # 清空当前状态
                    video_predictor.clear_all_points_in_video(self.inference_state)
                    self.prompts = {}
                    self.next_obj_id = 1
                    
                    # 重新添加其他提示
                    for temp_obj_id, prompt in temp_prompts.items():
                        if prompt["boxes"]:
                            for box in prompt["boxes"]:
                                self.add_box_prompt(0, *box, temp_obj_id)
                        elif prompt["points"]:
                            for i, point in enumerate(prompt["points"]):
                                self.add_point_prompt(0, point[0], point[1], 
                                                    prompt["labels"][i] == 1, temp_obj_id)
                    
                    return True, f"已清空目标{obj_id}的提示"
                else:
                    return False, f"目标{obj_id}不存在"
                    
        except Exception as e:
            return False, f"清空提示失败: {str(e)}"
    
    def propagate_video(self, max_frames=20):
        """传播分割结果到整个视频"""
        try:
            video_segments = {}
            
            # 关键修复：使用video_predictor.propagate_in_video
            for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in video_predictor.propagate_in_video(
                self.inference_state, start_frame_idx=0, max_frame_num_to_track=max_frames, 
                reverse=False, propagate_preflight=True
            ):
                video_segments[frame_idx] = {
                    out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(obj_ids)
                }
            
            return True, video_segments
            
        except Exception as e:
            return False, f"视频传播失败: {str(e)}"
    
    def visualize_frame(self, frame_idx, video_segments=None):
        """可视化特定帧的分割结果"""
        try:
            if frame_idx >= len(self.video_frames):
                return None, f"帧索引{frame_idx}超出范围"
                
            frame = self.video_frames[frame_idx].copy()
            
            # 显示提示点
            for obj_id, prompt in self.prompts.items():
                if prompt["points"]:
                    for i, point in enumerate(prompt["points"]):
                        color = (0, 255, 0) if prompt["labels"][i] == 1 else (0, 0, 255)
                        cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, -1)
                        # cv2.circle(frame, (int(point[0]), int(point[1])), 8, (255, 255, 255), 2)
                
                if prompt["boxes"]:
                    for box in prompt["boxes"]:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # 显示分割结果
            if video_segments and frame_idx in video_segments:
                import matplotlib.pyplot as plt
                from sam3.visualization_utils import show_mask
                
                plt.figure(figsize=(12, 8))
                plt.imshow(frame)
                
                for obj_id, mask in video_segments[frame_idx].items():
                    show_mask(mask, plt.gca(), obj_id=obj_id)
                
                plt.axis('off')
                plt.tight_layout(pad=0)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                result_image = Image.open(buf)
                plt.close()
                
                return result_image, f"帧{frame_idx}可视化完成"
            else:
                return Image.fromarray(frame), f"帧{frame_idx}（仅显示提示）"
                
        except Exception as e:
            return None, f"可视化失败: {str(e)}"


# 绘制坐标信息的辅助函数
# def draw_coordinate_info(frame, x, y, text_suffix="", is_current=False):
def draw_coordinate_info(frame, x, y, is_current=False):
    """在图像上绘制坐标信息"""
    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # 白色文字
    bg_color = (0, 0, 0)  # 黑色背景
    thickness = 1
    margin = 5
    
    # 创建坐标文本
    # coord_text = f"({x}, {y}){text_suffix}"
    coord_text = f"({x}, {y})"

    # 如果是当前鼠标位置，使用不同的颜色
    if is_current:
        font_color = (255, 255, 0)  # 黄色显示当前鼠标位置
        bg_color = (100, 100, 100)  # 灰色背景
    
    # 获取文本尺寸
    text_size = cv2.getTextSize(coord_text, font, font_scale, thickness)[0]
    
    # 计算文本背景矩形位置
    text_x = x + 10
    text_y = y - 10
    
    # 确保文本不超出图像边界
    img_height, img_width = frame.shape[:2]
    if text_x + text_size[0] + margin > img_width:
        text_x = x - text_size[0] - 15
    if text_y - text_size[1] - margin < 0:
        text_y = y + text_size[1] + 15
    
    # 绘制背景矩形
    bg_x1 = text_x - margin
    bg_y1 = text_y - text_size[1] - margin
    bg_x2 = text_x + text_size[0] + margin
    bg_y2 = text_y + margin
    
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # 绘制坐标文本
    cv2.putText(frame, coord_text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    return frame

# 重绘所有已存在的点和框及其坐标信息
def redraw_existing_annotations(frame, points_str, boxes_str):
    """重绘所有已存在的注释"""
    # 重绘已存在的点
    if points_str and points_str.strip():
        points_list = points_str.split(';')
        for i, point_str in enumerate(points_list):
            if point_str.strip():
                try:
                    point_coords = point_str.split(',')
                    if len(point_coords) == 2:
                        px, py = map(int, point_coords)
                        # 绘制点
                        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
                        # 绘制坐标信息（已存在的点）
                        frame = draw_coordinate_info(frame, px, py)
                except ValueError:
                    continue
    
    # 重绘已存在的框
    if boxes_str and boxes_str.strip():
        boxes_list = boxes_str.split(';')
        for i, box_str in enumerate(boxes_list):
            if box_str.strip():
                try:
                    box_coords = box_str.split(',')
                    if len(box_coords) == 4:
                        x1, y1, x2, y2 = map(int, box_coords)
                        # 绘制框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        # 在框的左上角显示坐标信息
                        frame = draw_coordinate_info(frame, x1, y1)
                except ValueError:
                    continue
    
    return frame




# 全局视频会话
video_session = VideoTrackingSession()

def handle_video_click(video_frame, evt: gr.SelectData, interaction_mode, current_points, current_boxes, click_state):
    """处理视频帧点击事件 - 增强版，支持实时坐标显示和点击后坐标保留"""
    if video_frame is None:
        return video_frame, current_points, current_boxes, click_state, "请先上传视频并选择帧"
    
    x, y = evt.index
    x, y = int(x), int(y)
    
    info_msg = ""
    # 确保视频帧是numpy数组格式
    if isinstance(video_frame, np.ndarray):
        vis_frame = video_frame.copy()
    else:
        vis_frame = np.array(video_frame)
    

    
    # 首先重绘所有已存在的注释
    vis_frame = redraw_existing_annotations(vis_frame, current_points, current_boxes)
    
    if interaction_mode == "📍 点提示 (Point)":
        # 添加当前鼠标位置的坐标显示（黄色，表示实时位置）
        vis_frame = draw_coordinate_info(vis_frame, x, y, is_current=True)
        
        # 添加新点
        new_point = f"{x},{y}"
        if current_points:
            current_points += f";{new_point}"
        else:
            current_points = new_point
            
        # 绘制新添加的点（绿色实心圆）
        cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
        # cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 2)
        
        # 为新点添加坐标显示（白色，表示已确认的点）
        vis_frame = draw_coordinate_info(vis_frame, x, y)
        
        info_msg = f"✅ 已添加点: ({x}, {y})"
        return Image.fromarray(vis_frame), current_points, current_boxes, None, info_msg
        
    elif interaction_mode == "🔲 框提示 (Box)":
        # 添加当前鼠标位置的坐标显示
        vis_frame = draw_coordinate_info(vis_frame, x, y, is_current=True)
        
        if click_state is None:
            # 第一次点击 - 记录起点
            click_state = [x, y]
            cv2.circle(vis_frame, (x, y), 1, (255, 0, 0), -1)  # 蓝色实心圆
            
            # 显示起点坐标
            vis_frame = draw_coordinate_info(vis_frame, x, y)
            
            info_msg = f"🔵 已记录起点: ({x}, {y})，请点击对角点完成框选"
            return Image.fromarray(vis_frame), current_points, current_boxes, click_state, info_msg
        else:
            # 第二次点击 - 完成框选
            x1, y1 = click_state
            x2, y2 = x, y
            
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            
            if xmin == xmax: xmax += 1
            if ymin == ymax: ymax += 1
            
            new_box = f"{xmin},{ymin},{xmax},{ymax}"
            if current_boxes:
                current_boxes += f";{new_box}"
            else:
                current_boxes = new_box
            
            # 绘制完成的框
            cv2.rectangle(vis_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
            
            # 在框的四个角显示坐标信息
            vis_frame = draw_coordinate_info(vis_frame, xmin, ymin)
            vis_frame = draw_coordinate_info(vis_frame, xmax, ymax)
            
            info_msg = f"✅ 已添加框: 左上({xmin},{ymin}) 右下({xmax},{ymax})"
            return Image.fromarray(vis_frame), current_points, current_boxes, None, info_msg
    
    return Image.fromarray(vis_frame), current_points, current_boxes, click_state, info_msg


def init_video_session(input_video, progress=gr.Progress()):
    """初始化视频会话"""
    if input_video is None:
        return None, None, "请上传视频文件", "", "", None, gr.update(maximum=0, value=0)
    
    try:
        progress(0.2, desc="正在初始化视频会话...")
        success, message = video_session.init_video(input_video)
        
        if not success:
            return None, None, message, "", "", None, gr.update(maximum=0, value=0)
        
        # 获取第一帧用于显示
        first_frame, _ = video_session.visualize_frame(0)
        
        # 更新帧选择器的最大值
        slider_update = gr.update(
            maximum=max(0, video_session.total_frames - 1),
            value=0
        )
        
        progress(0.5, desc="视频会话初始化完成")
        return first_frame, first_frame, message, "", "", None, slider_update
        
    except Exception as e:
        return None, None, f"初始化失败: {str(e)}", "", "", None, gr.update(maximum=0, value=0)



def add_video_prompt(frame_idx, point_prompt, box_prompt, is_positive=True, obj_id=None, progress=gr.Progress()):
    """添加视频提示 - 修复版本"""
    # print("🎯 add_video_prompt函数被调用")
    # print(f"point_prompt原始值: '{point_prompt}'")
    # print(f"box_prompt原始值: '{box_prompt}'")
    # print(f"------------is_positive: '{is_positive}'")

    try:
        if video_session.inference_state is None:
            return None, "请先初始化视频会话"
        
        progress(0.2, desc="正在添加提示...")
        
        # 修复1: 处理obj_id=-1的情况
        if obj_id == -1:
            obj_id = None
        
        # 修复2: 记录成功添加的提示数量
        points_added = 0
        boxes_added = 0
        target_obj_id = None
        
        # 修复3: 增强点提示解析 - 处理全角/半角逗号问题
        if point_prompt and point_prompt.strip():
            # 统一处理逗号格式：将全角逗号替换为半角逗号，并清理空格
            normalized_point_prompt = point_prompt.replace('，', ',').replace(' ', '')
            # print(f"规范化后的point_prompt: '{normalized_point_prompt}'")
            
            for point_str in normalized_point_prompt.split(';'):
                point_str = point_str.strip()
                if point_str:
                    try:
                        # 使用更健壮的坐标解析
                        coords = point_str.split(',')
                        if len(coords) == 2:
                            x, y = map(float, coords)
                            # print(f"解析坐标: x={x}, y={y}")
                            
                            success, message = video_session.add_point_prompt(
                                frame_idx, x, y, is_positive, obj_id
                            )

                            # print(f"add_point_prompt返回: success={success}, message='{message}'")
                            if success:
                                # print("add point success:", message)
                                points_added += 1
                                # 获取实际使用的obj_id
                                if target_obj_id is None and obj_id is not None:
                                    target_obj_id = obj_id
                                elif target_obj_id is None:
                                    # 从消息中提取obj_id
                                    import re
                                    match = re.search(r'目标(\d+)', message)
                                    if match:
                                        target_obj_id = int(match.group(1))
                        else:
                            print(f"❌ 坐标格式错误: {point_str}，应为 x,y 格式")
                            continue
                            
                    except ValueError as e:
                        print(f"❌ 解析点坐标失败: '{point_str}'，错误: {e}")
                        # 显示详细的错误信息
                        print(f"  坐标字符串长度: {len(point_str)}")
                        print(f"  坐标字符串repr: {repr(point_str)}")
                        continue
        
        # 修复4: 同样增强框提示解析
        if box_prompt and box_prompt.strip():
            normalized_box_prompt = box_prompt.replace('，', ',').replace(' ', '')
            # print(f"规范化后的box_prompt: '{normalized_box_prompt}'")
            
            for box_str in normalized_box_prompt.split(';'):
                box_str = box_str.strip()
                if box_str:
                    try:
                        coords = box_str.split(',')
                        if len(coords) == 4:
                            x1, y1, x2, y2 = map(float, coords)
                            # print(f"解析框坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            
                            success, message = video_session.add_box_prompt(
                                frame_idx, x1, y1, x2, y2, obj_id
                            )
                            if success:
                                boxes_added += 1
                                if target_obj_id is None and obj_id is not None:
                                    target_obj_id = obj_id
                                elif target_obj_id is None:
                                    import re
                                    match = re.search(r'目标(\d+)', message)
                                    if match:
                                        target_obj_id = int(match.group(1))
                        else:
                            print(f"❌ 框坐标格式错误: {box_str}，应为 x1,y1,x2,y2 格式")
                            continue
                            
                    except ValueError as e:
                        print(f"❌ 解析框坐标失败: '{box_str}'，错误: {e}")
                        continue
        
        progress(0.7, desc="提示添加完成")
        
        # 修复5: 确保会话中确实有提示
        if not video_session.prompts and points_added == 0 and boxes_added == 0:
            return None, "❌ 提示添加失败，请检查坐标格式（使用半角逗号分隔）"
        
        # 更新显示当前帧
        updated_frame, message = video_session.visualize_frame(frame_idx)
        
        info_msg = f"✅ 提示添加成功 | 点: {points_added}, 框: {boxes_added}"
        if target_obj_id is not None:
            info_msg += f" | 目标ID: {target_obj_id}"
        
        # 显示当前所有提示信息
        total_prompts = sum(len(p['points']) + len(p['boxes']) for p in video_session.prompts.values())
        info_msg += f" | 总提示数: {total_prompts}"
        
        return updated_frame, info_msg
        
    except Exception as e:
        import traceback
        print("🔥 add_video_prompt函数异常:")
        traceback.print_exc()
        return None, f"❌ 添加提示失败: {str(e)}"



def process_video_tracking(
    frame_idx,
    max_frames,
    confidence_threshold,
    progress=gr.Progress()
):
    """处理视频跟踪 - 修复版本"""
    if video_session.inference_state is None:
        return None, None, "请先初始化视频会话并添加提示"
    
    # 修复1: 更友好的提示检查
    if not video_session.prompts:
        return None, None, "❌ 请先添加点或框提示（至少需要1个提示）"
    
    # 修复2: 统计提示信息
    total_points = sum(len(p['points']) for p in video_session.prompts.values())
    total_boxes = sum(len(p['boxes']) for p in video_session.prompts.values())
    
    print(f"调试信息: prompts={video_session.prompts}")
    print(f"调试信息: total_points={total_points}, total_boxes={total_boxes}")
    
    # 修复3: 即使不在帧0，如果有足够的提示也可以开始跟踪
    # 但需要确保至少有一个目标有初始提示
    has_frame0_prompts = False
    for obj_id, prompt in video_session.prompts.items():
        # 检查是否有至少一个点或框提示在视频的某个帧上
        # 由于我们通常从帧0开始，我们假设所有提示都在当前帧（通常是帧0）
        if prompt['points'] or prompt['boxes']:
            has_frame0_prompts = True
            break
    
    if not has_frame0_prompts:
        return None, None, "❌ 至少需要一个目标有点或框提示"
    
    try:
        progress(0.1, desc="正在准备视频跟踪...")
        
        # 修复4: 确保推理状态已正确初始化
        # 如果需要，重新初始化推理状态
        if video_session.inference_state is None:
            success, msg = video_session.init_video(video_session.video_path)
            if not success:
                return None, None, f"❌ 重新初始化失败: {msg}"
        
        progress(0.3, desc="正在传播分割结果...")
        
        # 传播分割结果
        success, video_segments = video_session.propagate_video(max_frames)
        
        if not success:
            return None, None, video_segments  # 这里video_segments实际上是错误信息
        
        progress(0.6, desc="正在生成结果视频...")
        
        # 创建结果视频
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # 创建mask图像临时目录
        mask_temp_dir = tempfile.mkdtemp()
        mask_image_paths = []
        
        # 获取视频参数
        cap = cv2.VideoCapture(video_session.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f"视频参数: 宽度={width}, 高度={height}")
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'x264')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 渲染每一帧
        total_frames = min(max_frames, len(video_session.video_frames))
        # print("00")
        for i in range(total_frames):
            if i in video_segments:
                try:
                    # 使用简单的颜色覆盖渲染
                    # print("01")
                    vis_frame = video_session.video_frames[i].copy()
                    # print("02")

                    # 为每个目标分配不同的颜色
                    # colors = [
                    #     (255, 0, 0),    # 蓝色
                    #     (0, 255, 0),    # 绿色
                    #     (0, 0, 255),    # 红色
                    #     (255, 255, 0),  # 青色
                    #     (255, 0, 255),  # 品红
                    #     (0, 255, 255),  # 黄色
                    #     (128, 0, 0),    # 深蓝
                    #     (0, 128, 0),    # 深绿
                    #     (0, 0, 128),    # 深红
                    # ]
                    colors = [
                        (255, 0, 0),    # 蓝色
                        (0, 255, 0),    # 绿色
                        # (0, 0, 255),    # 红色
                        # (255, 255, 0),  # 青色
                        # (255, 0, 255),  # 品红
                        # (0, 255, 255),  # 黄色
                        (128, 0, 0),    # 深蓝
                        (0, 128, 0),    # 深绿
                        # (0, 0, 128),    # 深红
                    ]


                    # 创建纯mask图像（黑色背景，彩色mask）
                    mask_frame = np.zeros_like(vis_frame)
                    # print("03")
                    color_idx = 0
                    for obj_id, mask in video_segments[i].items():
                        color = colors[color_idx % len(colors)]
                        color_idx += 1
                        # print("04")
                        # 创建掩码叠加
                        overlay = vis_frame.copy()
                        # print("00004")
                        # print(f"overlay shape: {overlay.shape}, mask shape: {mask.shape},color: {color}")
                        # mask_adjusted = mask.squeeze(0)
                        mask = mask.squeeze(0)

                        # overlay[mask > 0] = color
                        overlay[mask > 0] = color
                        # print("004")
                        # 半透明叠加
                        # cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
                        cv2.addWeighted(overlay, 0.9, vis_frame, 0.1, 0, vis_frame)

                        # print("05")
                        # 添加目标ID标签

                        # 在mask帧上添加彩色mask（白色背景上的彩色区域）
                        # mask_frame[mask > 0] = color
                        mask_frame[mask > 0] = (255,255,255)


                        if mask.sum() > 0:
                            # 找到掩码中心
                            y_indices, x_indices = np.where(mask > 0)
                            # print("06")
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                center_x = int(np.mean(x_indices))
                                center_y = int(np.mean(y_indices))
                                
                                # print("07")
                                # print("ID:", obj_id, "Center:", center_x, center_y)
                                # 绘制目标ID
                                cv2.putText(vis_frame, f"ID:{obj_id}", 
                                          (center_x - 20, center_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                          (255, 255, 255), 1)
                                            # 保存mask图像
                    mask_filename = f"mask_frame_{i:05d}.png"
                    mask_filepath = os.path.join(mask_temp_dir, mask_filename)
                    cv2.imwrite(mask_filepath, cv2.cvtColor(mask_frame, cv2.COLOR_RGB2BGR))
                    mask_image_paths.append(mask_filepath)

                except Exception as e:
                    print(f"渲染帧{i}失败: {e}")
                    vis_frame = video_session.video_frames[i]
                    # 创建空的mask图像
                    mask_filename = f"mask_frame_{i:05d}.png"
                    mask_filepath = os.path.join(mask_temp_dir, mask_filename)
                    cv2.imwrite(mask_filepath, np.zeros_like(vis_frame))
                    mask_image_paths.append(mask_filepath)
            else:
                vis_frame = video_session.video_frames[i]

                # 创建空的mask图像
                mask_filename = f"mask_frame_{i:05d}.png"
                mask_filepath = os.path.join(mask_temp_dir, mask_filename)
                cv2.imwrite(mask_filepath, np.zeros_like(vis_frame))
                mask_image_paths.append(mask_filepath)
            
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            out.write(vis_frame_bgr)
            
            progress(0.6 + 0.4 * (i / total_frames), desc=f"渲染帧 {i+1}/{total_frames}")
        
        out.release()
        # 创建mask图像的ZIP文件
        mask_zip_path = os.path.join(tempfile.gettempdir(), f"mask_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        with zipfile.ZipFile(mask_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for mask_file in mask_image_paths:
                zipf.write(mask_file, os.path.basename(mask_file))

        # 清理临时文件
        for mask_file in mask_image_paths:
            try:
                os.remove(mask_file)
            except:
                pass
        try:
            os.rmdir(mask_temp_dir)
        except:
            pass
        
        # 生成预览帧
        preview_frame, _ = video_session.visualize_frame(frame_idx, video_segments)
        
        progress(1.0, desc="处理完成")
        
        info_msg = f"✨ 视频跟踪完成 | 总帧数: {total_frames} | 跟踪目标: {len(video_session.prompts)}"
        info_msg += f" | 点提示: {total_points} | 框提示: {total_boxes}"
        
        return str(output_path), preview_frame, info_msg, mask_zip_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ 处理失败: {str(e)}"



def clear_video_prompts(obj_id=None):
    """清空视频提示"""
    try:
        success, message = video_session.clear_prompts(obj_id)
        
        if success:
            # 更新显示当前帧
            updated_frame, _ = video_session.visualize_frame(0)
            return updated_frame, message, "", ""
        else:
            return None, message, "", ""
            
    except Exception as e:
        return None, f"清空失败: {str(e)}", "", ""

def change_video_frame(frame_idx):
    """切换视频帧显示 - 根据帧选择器更新当前帧显示"""
    try:
        if video_session.inference_state is None:
            return None, "请先初始化视频会话"
        
        # 确保帧索引在有效范围内
        if frame_idx < 0 or frame_idx >= video_session.total_frames:
            return None, f"帧索引 {frame_idx} 超出范围 (0-{video_session.total_frames-1})"
        
        # 获取并显示指定帧
        updated_frame, message = video_session.visualize_frame(frame_idx)
        return updated_frame, f"已切换到帧 {frame_idx}/{video_session.total_frames-1}"
        
    except Exception as e:
        return None, f"切换帧失败: {str(e)}"

def update_frame_on_slider_change(frame_idx):
    """当帧选择器值改变时更新当前帧显示"""
    return change_video_frame(frame_idx)

def create_enhanced_video_demo():
    """创建增强版视频跟踪演示界面"""
    
    custom_css = """
    .container { max-width: 1400px; margin: auto; padding-top: 20px; }
    h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin-bottom: 10px; }
    .description { text-align: center; font-size: 1.1em; color: #4a5568; margin-bottom: 30px; }
    .gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
    .gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    #interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
    .mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
    .mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
    .coordinate-display { background: #f7fafc; border: 1px solid #e2e8f0; padding: 8px; border-radius: 5px; margin-top: 5px; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
    )

    with gr.Blocks(theme=theme, css=custom_css, title="SAM3 增强版视频跟踪工作台") as demo:
        
        with gr.Column(elem_classes="container"):
            gr.Markdown("# 🎬 SAM3 增强版视频目标跟踪")
            gr.Markdown("支持点、框提示和多目标分割的下一代视频跟踪系统", elem_classes="description")
            
            with gr.Tabs():
                # ================= 增强版视频跟踪标签页 =================
                with gr.TabItem("🎬 智能视频跟踪", id="tab_video_enhanced"):
                    with gr.Row():
                        # 左侧控制栏
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### 📂 视频输入")
                                video_input = gr.Video(label="上传视频文件", interactive=True)
                                init_video_btn = gr.Button("🚀 初始化视频会话", variant="primary")
                            
                            with gr.Group():
                                gr.Markdown("### 🎮 交互设置")
                                video_frame_slider = gr.Slider(
                                    minimum=0, maximum=100, value=0, step=1,
                                    label="帧选择器", interactive=True
                                )
                                
                                video_interaction_mode = gr.Radio(
                                    choices=["📍 点提示 (Point)", "🔲 框提示 (Box)"],
                                    value="📍 点提示 (Point)",
                                    label="交互模式",
                                    show_label=False,
                                    elem_classes="mode-radio"
                                )
                                
                                with gr.Row():
                                    gr.Markdown("提示类型:")
                                    positive_toggle = gr.Checkbox(value=True, label="正提示", show_label=True)
                                    obj_id_input = gr.Number(value=-1, label="目标ID (-1=自动)", precision=0)
                                
                                with gr.Row():
                                    add_prompt_btn = gr.Button("✅ 添加提示", variant="primary")
                                    # print("222")
                                    clear_prompts_btn = gr.Button("🗑️ 清空提示", variant="secondary")
                            

                            
                            with gr.Accordion("⚙️ 跟踪参数", open=True):
                                max_frames_slider = gr.Slider(
                                    minimum=10, maximum=1000, value=300, step=10,
                                    label="最大跟踪帧数"
                                )
                                video_confidence_threshold = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                    label="置信度阈值"
                                )
                            
                            process_video_btn = gr.Button("▶️ 开始跟踪处理", variant="primary", size="lg")
                        
                        # 右侧显示栏
                        with gr.Column(scale=1):
                            with gr.Tabs():
                                with gr.TabItem("🖼️ 当前帧"):
                                    video_frame_display = gr.Image(
                                        type="pil", label="当前帧预览", interactive=True
                                    )

                                    video_interaction_info = gr.Markdown(
                                        "👆 点击图像添加点/框提示", elem_id="interaction-info"
                                    )
                                    # print(f"----video_interaction_info:{video_interaction_info}")
                                with gr.TabItem("📊 跟踪结果"):
                                    video_result_display = gr.Image(
                                        type="pil", label="跟踪结果预览"
                                    )
                                    video_output = gr.Video(label="✨ 最终跟踪视频")
                                    # 添加mask图像下载按钮
                                    mask_download = gr.File(
                                        label="📥 下载Mask图像包", 
                                        file_types=[".zip"],
                                        visible=False
                                    )
                            
                            video_info = gr.Textbox(label="📊 处理报告", interactive=False, lines=3)
                    
                    # 存储状态
                    video_click_state = gr.State(None)
                    video_current_points = gr.State("")
                    video_current_boxes = gr.State("")
                    
                    # 事件绑定
                    # "初始化视频会话调用init_video_session"
                    init_video_btn.click(
                        fn=init_video_session,
                        inputs=[video_input],
                        outputs=[video_frame_display, video_result_display, video_info, 
                                video_current_points, video_current_boxes, video_click_state, video_frame_slider]
                    )
                    
                    # 关键修改：帧选择器变化时更新当前帧显示
                    video_frame_slider.change(
                        fn=update_frame_on_slider_change,
                        inputs=[video_frame_slider],
                        outputs=[video_frame_display, video_interaction_info]
                    )
                    

                    # 关键修复：修正点击事件参数
                    # handle_video_click,添加鼠标点击信息；对图片点击，加入信息
                    # return Image.fromarray(vis_frame), current_points, current_boxes, click_state, info_msg

                    video_frame_display.select(
                        fn=handle_video_click,
                        inputs=[video_frame_display, video_interaction_mode, video_current_points, 
                               video_current_boxes, video_click_state],
                        outputs=[video_frame_display, video_current_points, video_current_boxes, 
                                video_click_state, video_interaction_info]
                    )
                    # 对应按钮坐标
                    add_prompt_btn.click(
                        fn=add_video_prompt,
                        inputs=[video_frame_slider, video_current_points, video_current_boxes, 
                               positive_toggle, obj_id_input],
                        outputs=[video_frame_display, video_interaction_info]
                    ).then(
                        fn=lambda: ("", ""),
                        outputs=[video_current_points, video_current_boxes]
                    )
                    
                    clear_prompts_btn.click(
                        fn=clear_video_prompts,
                        inputs=[obj_id_input],
                        outputs=[video_frame_display, video_interaction_info, video_current_points, video_current_boxes]
                    )
                    
                    process_video_btn.click(
                        fn=process_video_tracking,
                        inputs=[video_frame_slider, max_frames_slider, video_confidence_threshold],
                        outputs=[video_output, video_result_display, video_info, mask_download]
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=[mask_download]
                    )
        # 页脚
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #718096; font-size: 0.9em;">
            Powered by <strong>SAM3 增强版</strong> | 2025 SAM3 Interactive Studio | 支持多目标点框提示跟踪
        </div>
        """)
    
    return demo

def main():
    """主函数"""
    # 检查模型文件
    model_dir = current_dir / "models"
    if not model_dir.exists():
        print(f"创建模型目录: {model_dir}")
        model_dir.mkdir(exist_ok=True)
        
    checkpoint_path = model_dir / "sam3.pt"
    bpe_path = current_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    
    if not checkpoint_path.exists() or not bpe_path.exists():
        print("⚠️ 模型文件缺失")
        print(f"请确保以下文件存在:\n1. {checkpoint_path}\n2. {bpe_path}")
        
        response = input("是否尝试自动下载模型文件？(y/n): ").lower().strip()
        if response == 'y':
            try:
                import download_models
                download_models.main()
            except Exception as e:
                print(f"自动下载失败: {e}")
                return
        else:
            return
    
    print("🚀 正在启动 SAM3 增强版视频跟踪工作台...")
    demo = create_enhanced_video_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=True,
        debug=True,
        allowed_paths=[str(current_dir)]
    )

if __name__ == "__main__":  
    main()  
