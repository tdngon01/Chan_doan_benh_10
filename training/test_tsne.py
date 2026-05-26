import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from modules.config import System_Config as cfg
import modules.dataset as dataset
from modules.utils import load_moco_pretrained_weights

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_backbone(backbone_name):
    if backbone_name == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
        feature_dim = 1280
    elif backbone_name == "ResNet":
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()
        feature_dim = 512
    elif backbone_name == "DenseNet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Identity()
        feature_dim = 1024
    elif backbone_name == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
        feature_dim = 1280
    elif backbone_name == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Identity()
        feature_dim = 1024
    elif backbone_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier = nn.Identity()
        feature_dim = 25088
    else:
        raise ValueError(f"Backbone {backbone_name} không hỗ trợ.")
    return model, feature_dim

def main():
    backbone_name = cfg.PRETRAIN_CONFIG["BACKBONE"]
    checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_50.pth.tar")
    
    print(f"Đang chuẩn bị mô hình: {backbone_name}")
    model, _ = get_backbone(backbone_name)
    
    if os.path.exists(checkpoint_path):
        model = load_moco_pretrained_weights(model, checkpoint_path)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {checkpoint_path}. Sử dụng trọng số ngẫu nhiên!")
        
    model = model.to(device)
    model.eval()
    
    # Lấy DataLoader
    _, val_loader, test_loader = dataset.Fine_Tune_DataLoader()
    # Sử dụng test loader để trực quan hóa
    loader = test_loader
    print(f"Đang trích xuất embeddings từ tối đa 5000 mẫu trong test dataset...")
    
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            features = model(images)
            
            # Flatten features nếu cần
            if len(features.shape) > 2:
                features = torch.flatten(features, start_dim=1)
                
            embeddings.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            # Giới hạn số lượng mẫu để t-SNE chạy nhanh (khoảng 3000-5000 mẫu)
            current_count = sum(len(x) for x in embeddings)
            if current_count >= 5000:
                break
                
    embeddings = np.concatenate(embeddings)[:5000]
    labels = np.concatenate(labels_list)[:5000]
    
    print(f"Tổng số mẫu trích xuất: {embeddings.shape[0]} với kích thước vector nhúng: {embeddings.shape[1]}")
    print("Đang chạy thuật toán t-SNE (quá trình này có thể mất vài phút)...")
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Chuẩn bị vẽ biểu đồ
    plt.figure(figsize=(16, 7))
    
    # 1. Subplot 1: "No Finding" vs "Có Bệnh Lý"
    # Lớp "No finding" là phần tử cuối cùng trong CLASS_NAMES
    is_no_finding = labels[:, -1] == 1
    
    plt.subplot(1, 2, 1)
    plt.scatter(embeddings_2d[is_no_finding, 0], embeddings_2d[is_no_finding, 1], c='#1f77b4', label='No Finding', alpha=0.5, s=8)
    plt.scatter(embeddings_2d[~is_no_finding, 0], embeddings_2d[~is_no_finding, 1], c='#d62728', label='Disease (Có bệnh)', alpha=0.5, s=8)
    plt.title("t-SNE: Không gian biểu diễn Nhị phân (Bình thường vs Bệnh lý)", fontsize=12)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # 2. Subplot 2: Phân cụm các lớp bệnh lý phổ biến nhất
    plt.subplot(1, 2, 2)
    
    # Tính tần suất xuất hiện của mỗi bệnh lý (loại bỏ No Finding)
    class_frequencies = labels[:, :-1].sum(axis=0)
    top_class_indices = np.argsort(class_frequencies)[-4:]  # Lấy top 4 bệnh nhiều nhất
    
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#e377c2']
    
    # Vẽ các mẫu thuộc Top 4 lớp này (chú ý chỉ vẽ mẫu có nhãn của lớp đó)
    for idx, class_idx in enumerate(top_class_indices):
        class_name = cfg.CLASS_NAMES[class_idx]
        mask = labels[:, class_idx] == 1
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=colors[idx], label=f"{class_name} ({int(class_frequencies[class_idx])} mẫu)", alpha=0.7, s=12)
        
    # Các mẫu còn lại hoặc No finding
    other_mask = (labels[:, top_class_indices].sum(axis=1) == 0)
    plt.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1], c='lightgray', label='Others / No Finding', alpha=0.2, s=5)
    
    plt.title("t-SNE: Sự tách biệt của các Bệnh lý phổ biến nhất", fontsize=12)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Tạo thư mục và lưu ảnh
    charts_dir = os.path.join(cfg.BASE_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    save_path = os.path.join(charts_dir, f"tsne_moco_{backbone_name}.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ phân tích t-SNE tại: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
