import torch
import os

class System_Config:
    PROJECT_NAME = "App_Demo"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True
    IMG_SIZE= 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    TYPE = "moco" # "moco" hoặc "spark"
    temp_vram = "lora" # "full" hoặc "lora"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data", "train_csv")
    CVS_DIR = os.path.join(DATA_DIR, "data_csv10")

    PRE_TRAIN_CSV = os.path.join(CVS_DIR, "pre_train.csv")
    TRAIN_CSV = os.path.join(CVS_DIR, "train_split.csv")
    VAL_CSV = os.path.join(CVS_DIR, "val_split.csv")
    TEST_CSV = os.path.join(CVS_DIR, "test_split.csv")

    NIH_IMAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "pretrain_images"))
    Vin_IMAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "finetune_images"))

    TRAIN_PRE_TRAIN_DIR_IMG = NIH_IMAGE_ROOT
    TRAIN_FINE_TUNE_DIR_IMG = Vin_IMAGE_ROOT

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_moco")
    CHECKPOINT_SAVE = os.path.join(CHECKPOINT_DIR)
    LOGS_DIR =  os.path.join(BASE_DIR, "logs_moco")

    CLASS_NAMES = ['Aortic enlargement', 'Atelectasis', 'Calcification', 
                'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 
        'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis','No finding']
    
    NUM_CLASSES = len(CLASS_NAMES)

    PRETRAIN_CONFIG = {
        "BACKBONE": "EfficientNet",  # VGG16, ResNet, DenseNet, MobileNet, GoogleNet, VGG16
        "PRE_TRAIN": 25000,
        "EPOCHS": 100,
        "BATCH_SIZE": 32,
        "LR": 1e-3,
        #"LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        
        "m": 0.999,
        "T": 0.07,
        "K": 4096,
        "DIM": 128,
        "mlp": True,
        "pretrained": True
    }
    CHECKPOINT_SAVE_PRETRAIN = os.path.join(CHECKPOINT_DIR, f"checkpoint_pretrain_{PRETRAIN_CONFIG['BACKBONE']}")
    FINE_TUNE_CONFIG = {
        "EPOCHS": 50,
        "BATCH_SIZE": 32,
        "LR_LORA": 1e-4,
        "LR_FULL": 1e-5,
        "WEIGHT_DECAY_LORA": 1e-4,
        "WEIGHT_DECAY_FULL": 5e-4,
        "MODEL_NAME": PRETRAIN_CONFIG["BACKBONE"]
    }
    CHECKPOINT_SAVE_FINE_TUNE = os.path.join(CHECKPOINT_DIR, f"checkpoint_finetune_{FINE_TUNE_CONFIG['MODEL_NAME']}")
    LORA_CONFIG = {
        "RANK": 8,
        "LORA_ALPHA": 16,
        "DROPOUT": 0.3,
        "MODULES_TO_SAVE": []
    }
