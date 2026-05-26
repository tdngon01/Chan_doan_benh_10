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
else:
    from modules.utils import load_spark_pretrained_weights as load_weights

DEVICE = cfg.DEVICE
SUPPORTED_BACKBONES = ("EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16")

def model_full(backbone, pretrain_checkpoint_path):
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
    
    # Tải trọng số pre-train
    model = load_weights(model, pretrain_checkpoint_path)
    model = model.to(DEVICE)
    return model

def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    set_seed(int(cfg.SEED))

    # Tải các loader chung
    train_dataloader, val_dataloader, test_dataloader = dataset.Fine_Tune_DataLoader()
    print("Dữ liệu huấn luyện đã sẵn sàng.")
    
    # Danh sách 6 backbone cần full fine-tune
    backbones = ["EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16"]
    
    for backbone in backbones:
        print(f"\n======================================================================")
        print(f" BẮT ĐẦU FULL FINE-TUNE BACKBONE: {backbone}")
        print(f"======================================================================")
        
        # Cấu hình động cho backbone hiện tại
        cfg.FINE_TUNE_CONFIG["MODEL_NAME"] = backbone
        cfg.PRETRAIN_CONFIG["BACKBONE"] = backbone
        cfg.CHECKPOINT_SAVE_PRETRAIN = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_pretrain_{backbone}")
        
        if cfg.TYPE == "moco":
            pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_50.pth.tar")
        else:
            pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_spark_epoch_50.pth.tar")
            
        LOG_DIR = os.path.join(cfg.LOGS_DIR, f"full_finetune_{backbone}")
        TENSORBOARD_DIR = os.path.join(LOG_DIR, f"tensorboard_full_{backbone}")
        METRICS_CSV_PATH = os.path.join(LOG_DIR, f"metrics_full_{backbone}.csv")
        METRICS_JSON_PATH = os.path.join(LOG_DIR, f"metrics_full_{backbone}.json")
        RUN_INFO_PATH = os.path.join(LOG_DIR, f"run_info_full_{backbone}.json")
        CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR, f"full_finetune_{backbone}")
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Thiết lập resume_path nếu có checkpoint cũ của full fine-tuning
        resume_path = os.path.join(CHECKPOINT_DIR, "last_fine_tuning.pth.tar")
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=DEVICE)
                start_epoch = int(checkpoint.get("epoch", 0))
                if start_epoch >= int(cfg.FINE_TUNE_CONFIG["EPOCHS"]):
                    print(f"Backbone {backbone} đã hoàn thành huấn luyện Full Fine-tuning đủ {cfg.FINE_TUNE_CONFIG['EPOCHS']} epochs. Chuyển sang mô hình tiếp theo.")
                    continue
            except Exception as e:
                print(f"Không thể đọc checkpoint để kiểm tra: {e}. Sẽ tiến hành resume huấn luyện.")
        else:
            resume_path = None

        if not os.path.exists(pretrain_checkpoint_path):
            print(f"CẢNH BÁO: Không tìm thấy checkpoint pre-train tại: {pretrain_checkpoint_path}. Bỏ qua backbone {backbone}!")
            continue

        model = model_full(backbone, pretrain_checkpoint_path)
        print(f"Model {backbone} đã được khởi tạo và tải trọng số tiền huấn luyện.")
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Tổng số tham số phải huấn luyện: {total_params:,} tham số")

        # Thống kê số mẫu dương
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
            "checkpoint_pretrain": pretrain_checkpoint_path,
            "fine_tune_config": cfg.FINE_TUNE_CONFIG,
            "counts_train_csv": counts,
            "train_samples": len(train_dataloader.dataset),
            "val_samples": len(val_dataloader.dataset),
            "test_samples": len(test_dataloader.dataset),
            "train_batches": len(train_dataloader),
            "val_batches": len(val_dataloader),
            "test_batches": len(test_dataloader),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": total_params,
            "train_mode": "full_finetune",
        })
        print(f"Đã lưu run info: {RUN_INFO_PATH}")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.FINE_TUNE_CONFIG["LR_FULL"],
            weight_decay=cfg.FINE_TUNE_CONFIG["WEIGHT_DECAY_FULL"]
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
            best_ckpt="full_best_auc",
            last_ckpt="last_fine_tuning",
            train_mode_name=f"Full Fine-Tune [{backbone}]",
            resume_path=resume_path,
        )

        print(f"\n Hoàn thành huấn luyện Full Fine-tuning cho Backbone: {backbone}\n")

    print("\nHuấn luyện Full Fine-tuning trên toàn bộ 6 mạng hoàn tất!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
