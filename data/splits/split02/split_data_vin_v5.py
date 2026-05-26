# https://www.kaggle.com/datasets/nih-chest-xrays/sample
# https://www.kaggle.com/datasets/rishabhrp/chest-x-ray-dataset
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def resolve_image_path(source_image_folder, image_id):
    image_name = str(image_id)
    candidates = [image_name]
    if not os.path.splitext(image_name)[1]:
        candidates.append(f"{image_name}.png")

    for candidate in candidates:
        src_path = os.path.join(source_image_folder, candidate)
        if os.path.exists(src_path):
            return src_path, candidate

    return None, candidates[-1]

def copy_images_from_csv(image_ids, source_image_folder, dest_image_folder):
    os.makedirs(dest_image_folder, exist_ok=True)
    print(f"Đang tiến hành copy ảnh sang: {dest_image_folder}")

    count_success = 0
    missing_images = []

    for image_id in tqdm(image_ids, desc="Copying Images"):
        src_path, image_name = resolve_image_path(source_image_folder, image_id)
        dst_path = os.path.join(dest_image_folder, image_name)

        if src_path is not None:
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            count_success += 1
        else:
            missing_images.append(image_id)

    print("\n--- TỔNG KẾT COPY ẢNH ---")
    print(f"Đã copy thành công: {count_success}/{len(image_ids)} ảnh.")
    if missing_images:
        print(f"Cảnh báo: Không tìm thấy {len(missing_images)} ảnh trong thư mục gốc.")

def main():
    df = pd.read_csv(r'D:\Khoa_Luan\Du_Lieu\VinBigData_CXR_01\train_csv\train.csv')
    TARGETS = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung Opacity",
        "Nodule/Mass", "Other lesion", "Pleural effusion",
        "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
        "No finding"
    ]
    for i in TARGETS:
        df[i] = (df['class_name'] == i).astype(int) #tạo cột cho bệnh

    df = df.groupby('image_id')[TARGETS].max().reset_index() #hợp nhất nhãn
    print(f"Tổng số ảnh: {len(df)}")

    np.random.seed(42)
    image_id = df['image_id'].values
    train_id, temp_id = train_test_split(
        image_id,
        test_size=0.3,
        random_state=42
    )
    val_id, test_id = train_test_split(
        temp_id,
        test_size=0.5,
        random_state=42
    )
    
    df_train = df[df['image_id'].isin(train_id)] #lấy ảnh thuộc id
    df_val = df[df['image_id'].isin(val_id)]
    df_test = df[df['image_id'].isin(test_id)]

    save_dir = os.path.join(r'data/train_csv', 'data_csv10')
    os.makedirs(save_dir, exist_ok=True)
    df_train.to_csv(os.path.join(save_dir, 'train_split.csv'), index=False)
    df_val.to_csv(os.path.join(save_dir, 'val_split.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'test_split.csv'), index=False)

    source_image_folder = r'D:\Khoa_Luan\Du_Lieu\VinBigData_CXR_01\train'
    dest_image_folder = r'D:\Khoa_Luan\03_Code\finetune_images'
    split_image_ids = pd.concat([
        df_train['image_id'],
        df_val['image_id'],
        df_test['image_id'],
    ]).drop_duplicates().tolist()
    copy_images_from_csv(split_image_ids, source_image_folder, dest_image_folder)

    print(f"Train: {len(df_train)} ảnh")
    print(f"val: {len(df_val)} ảnh")
    print(f"test: {len(df_test)} ảnh")
    print(f"Đã lưu tại: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
