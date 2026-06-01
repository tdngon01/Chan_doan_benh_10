import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from modules.config import System_Config as cfg
import modules.dataset as dataset
from modules.utils import load_moco_pretrained_weights

# Đảm bảo import đúng từ thư viện grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("LỖI: Vui lòng cài đặt thư viện grad-cam bằng cách chạy: pip install grad-cam")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_backbone_model(backbone_name):
    if backbone_name == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
        target_layers = [model.features[-1]]
    elif backbone_name == "ResNet":
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()
        target_layers = [model.layer4[-1]]
    elif backbone_name == "DenseNet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Identity()
        target_layers = [model.features.norm5]
    elif backbone_name == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
        target_layers = [model.features[-1]]
    elif backbone_name == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Identity()
        target_layers = [model.inception5b]
    elif backbone_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier = nn.Identity()
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Backbone {backbone_name} không hỗ trợ.")
    return model, target_layers

class ContrastiveSimilarityTarget:
    def __init__(self, target_embedding):
        # Chuyển embedding đích thành tensor và detached
        self.target_embedding = target_embedding.detach().to(device)

    def __call__(self, model_output):
        # model_output: (B, D) là embedding của view_q
        # target_embedding: (1, D) hoặc (D,) là embedding của view_k
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
            
        q_norm = nn.functional.normalize(model_output, dim=1)
        k_norm = nn.functional.normalize(self.target_embedding, dim=1)
        
        # Tích vô hướng (cosine similarity) giữa hai vector đặc trưng
        similarity = torch.sum(q_norm * k_norm, dim=1)
        return similarity

def main():
    backbone_name = cfg.PRETRAIN_CONFIG["BACKBONE"]
    checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_100.pth.tar")
    
    print(f"Khởi tạo mô hình: {backbone_name}")
    model, target_layers = get_backbone_model(backbone_name)
    
    if os.path.exists(checkpoint_path):
        model = load_moco_pretrained_weights(model, checkpoint_path)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {checkpoint_path}. Sử dụng trọng số ngẫu nhiên!")
        
    model = model.to(device)
    model.eval()
    
    # 1. Tìm một ảnh bệnh lý từ test set để làm mẫu trực quan
    test_df = pd.read_csv(cfg.TEST_CSV)
    # Lấy các mẫu không phải "No Finding"
    diseased_df = test_df[test_df['No finding'] == 0]
    
    if len(diseased_df) > 0:
        sample_row = diseased_df.iloc[0]
        # Tìm nhãn bệnh của mẫu
        diseases = [col for col in cfg.CLASS_NAMES[:-1] if sample_row[col] == 1]
        label_text = ", ".join(diseases)
    else:
        sample_row = test_df.iloc[0]
        label_text = "No finding"
        
    image_id = sample_row["image_id"].replace(".png", "")
    image_index = dataset.get_image_index_VIN()
    img_path = image_index.get(image_id)
    
    if img_path is None or not os.path.exists(img_path):
        print(f"LỖI: Không tìm thấy file ảnh cho image_id: {image_id}")
        return
        
    print(f"Chọn ảnh mẫu: {img_path} (Nhãn: {label_text})")
    
    # Load ảnh gốc
    orig_img = Image.open(img_path).convert("RGB")
    # Resize để trực quan hóa
    orig_img_resized = orig_img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    orig_img_np = np.array(orig_img_resized, dtype=np.float32) / 255.0
    
    # 2. Tạo 2 views tăng cường bằng TwoCropsTransform của MoCo
    moco_transform = dataset.get_transforms(stage='pre_train_moco')
    
    # moco_transform(orig_img) trả về [img_q, img_k] là 2 PyTorch Tensors đã chuẩn hóa
    img_q, img_k = moco_transform(orig_img)
    
    # Đưa vào GPU và thêm batch dimension
    img_q_tensor = img_q.unsqueeze(0).to(device)
    img_k_tensor = img_k.unsqueeze(0).to(device)
    
    # 3. Trích xuất embedding của view_k để làm vector đích
    with torch.no_grad():
        embedding_k = model(img_k_tensor)
        if len(embedding_k.shape) > 2:
            embedding_k = torch.flatten(embedding_k, start_dim=1)
            
    # 4. Cấu hình Grad-CAM cho view_q để tối đa hóa độ tương đồng với view_k
    target = ContrastiveSimilarityTarget(embedding_k)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Chạy Grad-CAM trên view_q
    grayscale_cam = cam(input_tensor=img_q_tensor, targets=[target])
    
    # Lấy bản đồ cam cho ảnh đầu tiên
    grayscale_cam = grayscale_cam[0, :]
    
    # Chuẩn bị ảnh view_q ở định dạng numpy [0,1] để làm ảnh nền
    # Ta cần đảo ngược quá trình Normalize để ảnh hiển thị đẹp hơn
    mean = np.array(cfg.MEAN)
    std = np.array(cfg.STD)
    view_q_np = img_q.permute(1, 2, 0).numpy()
    view_q_np = (view_q_np * std) + mean
    view_q_np = np.clip(view_q_np, 0, 1)
    
    # Tạo overlay màu
    cam_image = show_cam_on_image(view_q_np, grayscale_cam, use_rgb=True)
    
    # 5. Vẽ biểu đồ hiển thị kết quả
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img_resized)
    plt.title(f"Ảnh X-quang Gốc\n(Nhãn: {label_text})")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(view_q_np)
    plt.title("View Q (Đã tăng cường)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cam_image)
    plt.title("Grad-CAM (Vùng quyết định tương đồng)")
    plt.axis("off")
    
    # Lưu kết quả
    charts_dir = os.path.join(cfg.BASE_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    save_path = os.path.join(charts_dir, f"gradcam_contrastive_{backbone_name}.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu hình ảnh phân tích Grad-CAM tại: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
