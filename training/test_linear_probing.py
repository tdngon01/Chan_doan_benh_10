import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.models as models
from torch import nn, optim
import modules.dataset as dataset
from modules.config import System_Config as cfg
from modules.trainer import *

if cfg.TYPE == "moco":
    from modules.utils import load_moco_pretrained_weights as load_weights
    PRETRAIN_CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_50.pth.tar")
elif cfg.TYPE == "spark":
    from modules.utils import load_spark_pretrained_weights as load_weights
    PRETRAIN_CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_spark_epoch_50.pth.tar")

DEVICE = cfg.DEVICE
LOG_DIR = os.path.join(cfg.LOGS_DIR, f"linear_probing_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")
TENSORBOARD_DIR = os.path.join(LOG_DIR, f"tensorboard_linear_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")
METRICS_CSV_PATH = os.path.join(LOG_DIR, f"metrics_linear_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.csv")
METRICS_JSON_PATH = os.path.join(LOG_DIR, f"metrics_linear_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.json")
RUN_INFO_PATH = os.path.join(LOG_DIR, f"run_info_linear_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.json")
CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR, f"linear_probing_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")

RESUME_PATH = None
SUPPORTED_BACKBONES = ("EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16")

def model_linear_probing():
    backbone = cfg.FINE_TUNE_CONFIG["MODEL_NAME"]
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    else:
        raise ValueError(
            f"Không có backbone '{backbone}'. Chỉ có các backbone: {', '.join(SUPPORTED_BACKBONES)}"
        )
    
    # Tải trọng số pre-train từ MoCo/SparK
    model = load_weights(model, PRETRAIN_CHECKPOINT_PATH)
    
    # DÒNG QUAN TRỌNG NHẤT: Đóng băng toàn bộ Backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở băng duy nhất lớp phân loại (Classifier Head)
    if backbone in ("EfficientNet", "DenseNet", "MobileNet"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif backbone in ("ResNet", "GoogleNet"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif backbone == "VGG16":
        for param in model.classifier[6].parameters():
            param.requires_grad = True
            
    model = model.to(DEVICE)
    return model

def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(int(cfg.SEED))

    train_dataloader, val_dataloader, test_dataloader = dataset.Fine_Tune_DataLoader()
    print("Dữ liệu huấn luyện cho Linear Probing đã sẵn sàng.")
    
    model = model_linear_probing()
    print("Model đã được cấu hình cho Linear Probing (Backbone đã đóng băng hoàn toàn).")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Số tham số huấn luyện (chỉ Classifier Head): {trainable_params:,}")

    # Thống kê mẫu dương tính
    counts = {}
    for disease in cfg.CLASS_NAMES:
        try:
            counts[disease] = dataset.count_samples(cfg.TRAIN_CSV, disease)
        except Exception as e:
            counts[disease] = None

    # Lưu run info
    run_info = save_run_info(RUN_INFO_PATH, {
        "seed": int(cfg.SEED),
        "device": str(DEVICE),
        "num_classes": int(cfg.NUM_CLASSES),
        "class_names": cfg.CLASS_NAMES,
        "img_size": int(cfg.IMG_SIZE),
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "train_csv": cfg.TRAIN_CSV,
        "val_csv": cfg.VAL_CSV,
        "test_csv": cfg.TEST_CSV,
        "image_root": cfg.TRAIN_FINE_TUNE_DIR_IMG,
        "checkpoint_pretrain": PRETRAIN_CHECKPOINT_PATH,
        "fine_tune_config": cfg.FINE_TUNE_CONFIG,
        "counts_train_csv": counts,
        "train_samples": len(train_dataloader.dataset),
        "val_samples": len(val_dataloader.dataset),
        "test_samples": len(test_dataloader.dataset),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_mode": "linear_probing",
    })
    
    # Cho phép sử dụng LR lớn hơn một chút cho Linear Probing (ví dụ: 1e-3)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, 
        weight_decay=1e-4
    )

    def ckpt_extra_fn(run_info):
        return {
            "run_info_path": RUN_INFO_PATH,
            "run_info": run_info,
            "num_classes": int(cfg.NUM_CLASSES),
            "class_names": cfg.CLASS_NAMES,
            "fine_tune_config": cfg.FINE_TUNE_CONFIG,
        }

    run_training(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=DEVICE,
        log_dir=LOG_DIR,
        tensorboard_dir=TENSORBOARD_DIR,
        metrics_csv_path=METRICS_CSV_PATH,
        metrics_json_path=METRICS_JSON_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        run_info=run_info,
        ckpt_extra_fn=ckpt_extra_fn,
        best_ckpt="linear_best_auc",
        last_ckpt="last_linear_probing",
        train_mode_name="Linear Probing",
        resume_path=RESUME_PATH,
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
