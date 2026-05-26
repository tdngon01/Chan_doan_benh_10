import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.models as models
from peft import LoraConfig, get_peft_model
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

def get_target_modules_for_lora(model, backbone):
    target_modules = []
    if backbone == "EfficientNet":
        prefixes = ("features.6", "features.7", "features.8")
    elif backbone == "ResNet":
        prefixes = ("layer3", "layer4")
    elif backbone == "DenseNet":
        prefixes = ("features.denseblock3", "features.denseblock4")
    elif backbone == "MobileNet":
        prefixes = ("features.12", "features.13", "features.14", "features.15", "features.16", "features.17")
    elif backbone == "GoogleNet":
        prefixes = ("inception4a","inception4b","inception4c","inception4d","inception4e","inception5a", "inception5b")
    elif backbone == "VGG16":
        prefixes = ("features.28")
    else:
        raise ValueError(f"Không có backbone '{backbone}'. Chỉ hỗ trợ: {', '.join(SUPPORTED_BACKBONES)}")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.groups == 1 and name.startswith(prefixes):
            target_modules.append(name)
    return target_modules

def model_lora(backbone, pretrain_checkpoint_path):
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    else:
        raise ValueError(
            f"Không có backbone '{backbone}'. Chỉ có các backbone: {', '.join(SUPPORTED_BACKBONES)}"
        )
        
    model = load_weights(model, pretrain_checkpoint_path)
    
    # Đóng băng mạng gốc và chỉ dùng adapter LoRA và classifier mới
    for param in model.parameters():
        param.requires_grad = False
        
    target_modules = get_target_modules_for_lora(model, backbone)
    if backbone == "ResNet" or backbone == "GoogleNet":
        modules = ["fc"]
    elif backbone == "VGG16":
        modules = ["classifier.6"]
    else:
        modules = ["classifier"]
        
    lora_config = LoraConfig(
        r=cfg.LORA_CONFIG["RANK"],
        lora_alpha=cfg.LORA_CONFIG["LORA_ALPHA"],
        target_modules=target_modules,
        bias="none",
        lora_dropout=cfg.LORA_CONFIG["DROPOUT"],
        modules_to_save=modules,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(DEVICE)
    return model, target_modules, lora_config

def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    set_seed(int(cfg.SEED))

    # Tải các loader chung
    train_dataloader, val_dataloader, test_dataloader = dataset.Fine_Tune_DataLoader()
    print("Dữ liệu huấn luyện đã sẵn sàng.")
    
    # Danh sách 6 backbone cần fine-tune
    backbones = ["EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16"]
    
    for backbone in backbones:
        print(f"\n======================================================================")
        print(f" BẮT ĐẦU FINE-TUNE LORA BACKBONE: {backbone}")
        print(f"======================================================================")
        
        # Cấu hình động cho backbone hiện tại
        cfg.FINE_TUNE_CONFIG["MODEL_NAME"] = backbone
        cfg.PRETRAIN_CONFIG["BACKBONE"] = backbone
        cfg.CHECKPOINT_SAVE_PRETRAIN = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_pretrain_{backbone}")
        
        if cfg.TYPE == "moco":
            pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_50.pth.tar")
        else:
            pretrain_checkpoint_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_spark_epoch_50.pth.tar")
            
        LOG_DIR = os.path.join(cfg.LOGS_DIR, f"lora_finetune_{backbone}")
        TENSORBOARD_DIR = os.path.join(LOG_DIR, f"tensorboard_lora_{backbone}")
        METRICS_CSV_PATH = os.path.join(LOG_DIR, f"metrics_lora_{backbone}.csv")
        METRICS_JSON_PATH = os.path.join(LOG_DIR, f"metrics_lora_{backbone}.json")
        RUN_INFO_PATH = os.path.join(LOG_DIR, f"run_info_lora_{backbone}.json")
        CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR, f"lora_finetune_{backbone}")
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Thiết lập resume_path nếu có checkpoint cũ của lora
        resume_path = os.path.join(CHECKPOINT_DIR, "last_fine_tuning.pth.tar")
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=DEVICE)
                start_epoch = int(checkpoint.get("epoch", 0))
                if start_epoch >= int(cfg.FINE_TUNE_CONFIG["EPOCHS"]):
                    print(f"Backbone {backbone} đã hoàn thành huấn luyện LoRA đủ {cfg.FINE_TUNE_CONFIG['EPOCHS']} epochs. Chuyển sang mô hình tiếp theo.")
                    continue
            except Exception as e:
                print(f"Không thể đọc checkpoint để kiểm tra: {e}. Sẽ tiến hành resume huấn luyện.")
        else:
            resume_path = None

        if not os.path.exists(pretrain_checkpoint_path):
            print(f"CẢNH BÁO: Không tìm thấy checkpoint pre-train tại: {pretrain_checkpoint_path}. Bỏ qua backbone {backbone}!")
            continue

        model, target_modules, lora_config = model_lora(backbone, pretrain_checkpoint_path)
        print(f"Model {backbone} đã được khởi tạo và tải trọng số tiền huấn luyện với LoRA.")
        model.print_trainable_parameters()

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
            "lora_config": {
                "r": cfg.LORA_CONFIG["RANK"],
                "lora_alpha": cfg.LORA_CONFIG["LORA_ALPHA"],
                "dropout": cfg.LORA_CONFIG["DROPOUT"],
                "modules_to_save": lora_config.modules_to_save if hasattr(lora_config, "modules_to_save") else [],
            },
            "lora_target_modules": target_modules,
            "counts_train_csv": counts,
            "train_samples": len(train_dataloader.dataset),
            "val_samples": len(val_dataloader.dataset),
            "test_samples": len(test_dataloader.dataset),
            "train_batches": len(train_dataloader),
            "val_batches": len(val_dataloader),
            "test_batches": len(test_dataloader),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_mode": "lora_finetune",
        })
        print(f"Đã lưu run info: {RUN_INFO_PATH}")
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.FINE_TUNE_CONFIG["LR_LORA"],
            weight_decay=cfg.FINE_TUNE_CONFIG["WEIGHT_DECAY_LORA"]
        )

        def ckpt_extra_fn(run_info):
            return {
                "run_info_path": RUN_INFO_PATH,
                "run_info": run_info,
                "num_classes": int(cfg.NUM_CLASSES),
                "class_names": cfg.CLASS_NAMES,
                "lora_target_modules": target_modules,
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
            best_ckpt="lora_best_auc",
            last_ckpt="last_fine_tuning",
            train_mode_name=f"LoRA Fine-Tune [{backbone}]",
            resume_path=resume_path,
        )

        print(f"\n Hoàn thành huấn luyện LoRA cho Backbone: {backbone}\n")

    print("\nHuấn luyện LoRA trên toàn bộ 6 mạng hoàn tất!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
