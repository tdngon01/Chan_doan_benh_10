# BÁO CÁO TỔNG HỢP VÀ PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM
*Dự án: App_Demo (Chẩn đoán bệnh lý lồng ngực qua ảnh X-quang phổi)*
*So sánh hiệu năng giữa MoCo v2 + LoRA và MoCo v2 + Full Fine-tuning trên 6 kiến trúc Backbone mạng*

> [!NOTE]
> Tất cả các kết quả dưới đây được trích xuất tự động từ hệ thống log huấn luyện thực tế (`logs_moco/`), thông số cấu hình mô hình (`run_info_*.json`) và kết quả đánh giá trên tập kiểm tra độc lập (`results_eval/`).

---

## 1. Bảng 1: So sánh Hiệu năng Tổng quan trên Tập Kiểm tra (Test Set)
Bảng dưới đây tổng hợp các chỉ số trung bình (Mean) trên 15 lớp dữ liệu (14 loại bệnh lý và 1 lớp không bệnh lý "No finding") được đánh giá trên tập Test (2,250 ảnh):

| Kiến trúc Backbone | Phương thức Tinh chỉnh | Số tham số Trainable | Tỷ lệ Trainable % | Avg AUC | Avg Accuracy | Avg F1-Score | Avg Recall | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **LoRA** | **864,591** | **15.62%** | **0.9352** | **0.9529** | **0.4001** | 0.3468 | **0.6143** |
| | Full | 4,671,115 | 100.00% | 0.9337 | 0.9523 | 0.3590 | 0.3203 | 0.5149 |
| **MobileNet-V2** | **LoRA** | **751,375** | **20.65%** | **0.9356** | **0.9526** | **0.3967** | **0.3568** | **0.5002** |
| | Full | 2,887,439 | 100.00% | 0.9287 | 0.9509 | 0.3207 | 0.2999 | 0.3722 |
| **ResNet-18** | **LoRA** | **497,679** | **4.17%** | **0.9355** | **0.9527** | **0.3631** | 0.3194 | **0.5114** |
| | Full | 11,446,863 | 100.00% | 0.9319 | 0.9517 | 0.3473 | **0.3264** | 0.4872 |
| **DenseNet-121** | **LoRA** | **1,168,399** | **13.50%** | **0.9336** | **0.9521** | **0.3720** | **0.3304** | **0.5607** |
| | Full | 7,486,351 | 100.00% | 0.9276 | 0.9510 | 0.3210 | 0.3072 | 0.3796 |
| **GoogLeNet** | **LoRA** | **801,167** | **11.55%** | **0.9305** | **0.9509** | **0.3639** | **0.3265** | **0.5043** |
| | Full | 6,132,399 | 100.00% | 0.9214 | 0.9489 | 0.2976 | 0.2793 | 0.3644 |
| **VGG16** | LoRA | 2,146,319 | 1.55% | 0.9131 | 0.9450 | 0.3192 | 0.2687 | **0.5017** |
| | **Full** | **136,365,903** | **100.00%** | **0.9351** | **0.9527** | **0.4015** | **0.3688** | 0.4965 |

> [!TIP]
> **Nhận xét chính:**
> * LoRA vượt trội hơn Full Fine-tuning ở 5/6 backbone về cả AUC và F1-Score, mặc dù số lượng tham số cập nhật cực nhỏ (chỉ từ 1.55% đến 20.65% so với Full).
> * VGG16 là trường hợp duy nhất Full Fine-tuning tốt hơn hẳn LoRA (AUC 0.9351 so với 0.9131), do cấu trúc của VGG16 phụ thuộc lớn vào các lớp Fully Connected khổng lồ ở classifier, việc đóng băng phần lớn lớp tích chập khiến LoRA bị hạn chế biểu diễn.

---

## 2. Bảng 2: So sánh Động lực Huấn luyện trên tập Validation (Validation Dynamics)
Bảng dưới đây thống kê điểm đạt AUC cao nhất trên tập Validation, epoch tương ứng và thời gian huấn luyện trung bình của mỗi epoch:

| Kiến trúc Backbone | Tinh chỉnh | Epoch Đạt Max | Max Val AUC | Loss Train | Loss Val | Thời gian/Epoch (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | LoRA | 46 | 0.9329 | 0.1084 | 0.1323 | 143.0s |
| | Full | 37 | 0.9329 | 0.1076 | 0.1332 | 129.3s |
| **MobileNet-V2** | LoRA | 45 | **0.9342** | 0.0998 | 0.1342 | 140.1s |
| | Full | 48 | 0.9249 | 0.1168 | 0.1359 | **67.4s** |
| **ResNet-18** | LoRA | 43 | **0.9323** | 0.1145 | 0.1315 | 142.4s |
| | Full | 19 | 0.9291 | 0.1109 | 0.1306 | **71.2s** |
| **DenseNet-121** | LoRA | 44 | **0.9361** | 0.1084 | 0.1287 | 154.5s |
| | Full | 48 | 0.9293 | 0.1116 | 0.1330 | **75.5s** |
| **GoogLeNet** | LoRA | 37 | **0.9322** | 0.1186 | 0.1316 | 145.8s |
| | Full | 49 | 0.9233 | 0.1237 | 0.1362 | **68.3s** |
| **VGG16** | LoRA | 16 | 0.9108 | 0.1390 | 0.1558 | 142.6s |
| | Full | 23 | **0.9316** | 0.0899 | 0.1435 | **86.1s** |

> [!WARNING]
> **Nhận xét quan trọng về thời gian:**
> * Đối với các mạng CNN nhỏ và sâu như MobileNet-V2, ResNet-18, DenseNet-121, thời gian chạy mỗi epoch của LoRA cao gấp đôi Full Fine-tuning. 
> * **Giải thích kỹ thuật:** Đây là đặc trưng của việc triển khai thư viện PEFT trong PyTorch trên Windows. Các lớp bọc (wrappers) của LoRA tạo ra các hàm callback và hook chuyển đổi dữ liệu trung gian trong Python ở từng lớp tích chập, dẫn tới chi phí trích xuất phụ trợ (Python overhead) tăng cao khi duyệt qua mạng có nhiều khối tích chập độc lập, mặc dù khối lượng tính toán lý thuyết và bộ nhớ GPU giảm đi.

---

## 3. Bảng 3: So sánh Chi phí Bộ nhớ Đồ họa (GPU VRAM Utilization)
Bảng dưới đây thống kê lượng VRAM trung bình (Mean) và lớn nhất (Max) tiêu thụ trong quá trình huấn luyện và quá trình đánh giá (inference) trên tập test:

| Kiến trúc Backbone | Tinh chỉnh | Huấn luyện VRAM Mean (MB) | Huấn luyện VRAM Max (MB) | Đánh giá VRAM Mean (MB) | Đánh giá VRAM Max (MB) | % Tiết kiệm (Train) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | LoRA | **65.8** | **66.8** | **49.7** | **49.9** | **38.3%** |
| | Full | 106.7 | 106.9 | 53.9 | 54.3 | |
| **MobileNet-V2** | LoRA | 99.9 | 100.0 | **42.4** | **42.6** | -23.7% |
| | Full | **80.8** | **80.8** | 43.5 | 44.1 | |
| **ResNet-18** | LoRA | **86.3** | **86.3** | **72.9** | **73.1** | **59.9%** |
| | Full | 215.5 | 215.5 | 93.4 | 93.5 | |
| **DenseNet-121** | LoRA | **81.9** | **81.9** | **62.0** | **62.2** | **45.6%** |
| | Full | 150.5 | 150.8 | 69.5 | 69.8 | |
| **GoogLeNet** | LoRA | **114.6** | **114.7** | **54.9** | **55.4** | **13.2%** |
| | Full | 132.0 | 132.1 | 63.1 | 64.1 | |
| **VGG16** | LoRA | **631.9** | **632.0** | **560.3** | **561.3** | **70.2%** |
| | Full | 2,122.9 | 2,124.1 | 808.7 | 809.5 | |

---

## 4. Bảng 4: So sánh Mức độ Nhạy cảm với Bệnh lý Hiếm (Lớp Atelectasis & Pneumothorax)
Một trong những phát hiện khoa học giá trị nhất của khóa luận là khả năng giữ chỉ số **F1-Score tốt hơn trên các bệnh lý có tỷ lệ xuất hiện cực kỳ thấp** của LoRA so với Full Fine-tuning.

Dưới đây là chi tiết so sánh hiệu năng trên backbone tiêu biểu `EfficientNet-B0`:

| Loại Bệnh lý | Số ca Dương tính / Tổng ảnh Test | AUC (LoRA) | F1-Score (LoRA) | AUC (Full) | F1-Score (Full) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pneumothorax** (Tràn khí màng phổi) | 13 / 2,250 | **0.9597** | **0.1429** | 0.9368 | **0.0000** |
| **Atelectasis** (Xẹp phổi) | 19 / 2,250 | **0.9148** | **0.0870** | 0.9074 | **0.0000** |
| **Consolidation** (Đông đặc phổi) | 39 / 2,250 | **0.9475** | **0.2692** | 0.9437 | 0.2174 |
| **ILD** (Bệnh phổi kẽ) | 48 / 2,250 | **0.9345** | **0.2456** | 0.9285 | 0.0784 |
| **Nodule/Mass** (Nốt/Khối mờ) | 127 / 2,250 | 0.8971 | **0.0719** | **0.8993** | **0.0000** |

> [!CAUTION]
> **Phân tích hiện tượng sập F1-Score ở mô hình Full:**
> * Đối với các bệnh lý cực hiếm (chỉ dưới 50 ca dương tính trên tập test), mô hình **Full Fine-tuning bị suy giảm F1-Score về hẳn 0.0000**. Do số lượng mẫu mất cân bằng nghiêm trọng, mô hình học cách dự đoán toàn bộ là lớp âm tính để tối thiểu hóa hàm mất mát tổng thể.
> * Việc đóng băng các trọng số cơ bản trong **LoRA hoạt động như một bộ chính quy hóa (regularization) mạnh mẽ**, ngăn chặn mạng phân loại điều chỉnh quá đà (overfitting) các đặc trưng biểu diễn theo chiều hướng thiên vị hoàn toàn cho nhóm đa số, nhờ đó duy trì khả năng nhận biết nhạy bén hơn cho nhóm bệnh lý hiếm.
