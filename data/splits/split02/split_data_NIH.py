# https://www.kaggle.com/datasets/nih-chest-xrays/sample
# https://www.kaggle.com/datasets/rishabhrp/chest-x-ray-dataset
import shutil

import pandas as pd
import os
from tqdm import tqdm


def build_image_index(source_image_folder):
    image_index = {}
    print(f"Đang lập chỉ mục ảnh từ: {source_image_folder}")

    for root, _, files in os.walk(source_image_folder):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_index[file_name] = os.path.join(root, file_name)

    print(f"Tìm thấy {len(image_index)} ảnh trong thư mục gốc.")
    return image_index


def copy_images_from_csv(image_ids, source_image_folder, dest_image_folder):
    os.makedirs(dest_image_folder, exist_ok=True)
    print(f"Đang tiến hành copy ảnh sang: {dest_image_folder}")

    image_index = build_image_index(source_image_folder)
    count_success = 0
    missing_images = []

    for img_name in tqdm(image_ids, desc="Copying Images"):
        src_path = image_index.get(img_name)
        dst_path = os.path.join(dest_image_folder, img_name)

        if src_path is not None:
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            count_success += 1
        else:
            missing_images.append(img_name)

    print("\n--- TỔNG KẾT COPY ẢNH ---")
    print(f"Đã copy thành công: {count_success}/{len(image_ids)} ảnh.")
    if missing_images:
        print(f"Cảnh báo: Không tìm thấy {len(missing_images)} ảnh trong thư mục gốc.")
        print(f"Ví dụ ảnh bị thiếu: {missing_images[:5]}")


def main():
    df = pd.read_csv(r'D:\Khoa_Luan\Du_Lieu\archive\Data_Entry_2017.csv')
    df = df[df['Finding Labels'] == 'No Finding'].copy()

    df = df[['Image Index', 'Finding Labels', 'Patient ID']]
    df = df.rename(columns = {'Image Index': 'image_id'})

    n_pretrain = 5000
    df_pretrain = df.sample(n=n_pretrain, random_state=42).reset_index(drop=True)
    save_csv_path = r'data/train_csv/data_csv10/pre_train.csv'
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df_pretrain.to_csv(save_csv_path, index=False)
    print(f"Pre-train có: {len(df_pretrain)} ảnh")

    source_image_folder = r'D:\Khoa_Luan\Du_Lieu\archive' 
    
    # Thư mục bạn muốn lưu các ảnh đã lọc ra
    dest_image_folder = r'D:\Khoa_Luan\03_Code\pretrain_images'
    copy_images_from_csv(df_pretrain['image_id'].tolist(), source_image_folder, dest_image_folder)
        
if __name__ == "__main__":
    main()
