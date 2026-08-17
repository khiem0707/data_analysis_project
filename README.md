# Data Analysis Report Project - README

## 📋 Mục lục
1. [Giới thiệu dự án](#giới-thiệu-dự-án)
2. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Hướng dẫn chạy code](#hướng-dẫn-chạy-code)
5. [Kiểm tra Requirements](#-kiểm-tra-requirements)
6. [Xử lý lỗi thường gặp](#️-xử-lý-lỗi-thường-gặp)
7. [Các lệnh hữu ích](#️-các-lệnh-hữu-ích)
8. [Câu hỏi thường gặp](#-câu-hỏi-thường-gặp-faq)
9. [Dữ liệu và kích thước](#-dữ-liệu-và-kích-thước)
10. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
11. [Liên hệ & Hỗ trợ](#-liên-hệ--hỗ-trợ)

---

## 🎯 Giới thiệu dự án

Dự án này thực hiện **phân tích dữ liệu toàn diện** trên 3 dataset lớn:
- **A3**: Customer Personality Analysis (phân loại, clustering)
- **B3**: Bike Sharing Dataset (dự báo chuỗi thời gian)
- **C3**: UrbanSound8K (phân tích audio, clustering âm thanh)

---

## 📁 Cấu trúc thư mục

```
data_analysis_project/
│
├── data/                          # Chứa dữ liệu thô
│   ├── raw/                       # Dữ liệu gốc chưa xử lý
│   │   ├── A3_customer_personality/
│   │   ├── B3_bike_sharing/
│   │   └── C3_urbansound8k/
│   └── processed/                 # Dữ liệu đã xử lý (để trống, chờ tạo)
│
├── notebooks/                     # Chứa các script Python chính
│   ├── 01_A3_customer_personality_audit.py      # Kiểm kê A3
│   ├── 01_A3_customer_personality_eda.py        # EDA A3
│   ├── 02_B3_bike_sharing_audit.py              # Kiểm kê B3
│   ├── 02_B3_bike_sharing_eda_timeseries.py     # EDA B3
│   ├── 03_C3_urbansound8k_audit.py              # Kiểm kê C3
│   ├── 03_C3_urbansound8k_feature_extraction_eda.py  # EDA C3
│   ├── chapter4_A3_classification.py            # ML: phân loại A3
│   ├── chapter4_A3_clustering.py                # ML: clustering A3
│   ├── chapter4_B3_regression.py                # ML: hồi quy B3
│   ├── chapter4_B3_clustering.py                # ML: clustering B3
│   ├── chapter4_C3_clustering.py                # ML: clustering C3
│   └── add_advanced_plots.py                    # Vẽ biểu đồ nâng cao
│
├── outputs/                       # Kết quả từ các script
│   ├── round_01_data_audit/       # Kết quả kiểm kê A3, B3, C3
│   ├── round_02_A3_eda/           # Dữ liệu, hình ảnh, bảng từ EDA A3
│   ├── round_03_B3_eda/           # Dữ liệu, hình ảnh, bảng từ EDA B3
│   ├── round_04_C3_audio_eda/     # Dữ liệu, hình ảnh, bảng từ EDA C3
│   └── round_05_chapter4_ml/      # Kết quả ML, bảng metrics
│
├── images/                        # Chứa hình ảnh cho báo cáo
│   ├── nhom-a-3/                  # Hình A3
│   ├── nhom-b-3/                  # Hình B3
│   └── nhom-c-3/                  # Hình C3
│
├── excel/                         # Tệp Excel tổng hợp (nếu có)
│
├── report_assets/                 # Tài liệu báo cáo
│
├── requirements.txt               # Danh sách thư viện cần cài
└── README.md                      # File này
```

**Giải thích từng thư mục chính:**

| Thư mục | Mục đích |
|---------|---------|
| `data/raw/` | Chứa dữ liệu gốc (CSV, WAV) không được sửa đổi |
| `data/processed/` | Chứa dữ liệu đã làm sạch (hiện tại để trống) |
| `notebooks/` | Chứa tất cả script Python thực hiện phân tích |
| `outputs/` | Chứa kết quả sau khi chạy script (CSV, PNG, TXT logs) |
| `images/` | Chứa hình ảnh được lưu từ outputs để tổng hợp báo cáo |
| `excel/` | Tệp Excel tổng hợp kết quả (nếu có) |
| `report_assets/` | Tài liệu báo cáo cuối cùng |

---

## 🔧 Cài đặt môi trường

### Bước 1: Cài đặt Python (3.8+)
Đảm bảo bạn đã cài đặt Python 3.8 trở lên. Kiểm tra:
```bash
python --version
```

### Bước 2: Tạo Virtual Environment (tuỳ chọn nhưng khuyến khích)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện
Cài đặt tất cả thư viện từ `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Các thư viện chính:**
- pandas, numpy: xử lý dữ liệu
- matplotlib, seaborn: vẽ biểu đồ
- scikit-learn: machine learning
- librosa: xử lý audio
- scipy: tính toán thống kê

---

## 🚀 Hướng dẫn chạy code

### Thứ tự thực hiện (từng vòng):

#### **Vòng 1: Kiểm kê dữ liệu** ✅ (Đã hoàn thành)
```bash
# Chạy kiểm kê A3
python notebooks/01_A3_customer_personality_audit.py

# Chạy kiểm kê B3
python notebooks/02_B3_bike_sharing_audit.py

# Chạy kiểm kê C3
python notebooks/03_C3_urbansound8k_audit.py
```
**Kết quả:** Kiểm tra số dòng, cột, missing values, duplicates → Lưu vào `outputs/round_01_data_audit/`

---

#### **Vòng 2: EDA cho A3** ✅ (Đã hoàn thành)
```bash
python notebooks/01_A3_customer_personality_eda.py
```
**Kết quả:** 
- Dữ liệu làm sạch → `outputs/round_02_A3_eda/data/`
- Biểu đồ EDA → `outputs/round_02_A3_eda/figures/`
- Bảng thống kê → `outputs/round_02_A3_eda/tables/`

---

#### **Vòng 3: EDA cho B3** ✅ (Đã hoàn thành)
```bash
python notebooks/02_B3_bike_sharing_eda_timeseries.py
```
**Kết quả:** 
- Dữ liệu với time features → `outputs/round_03_B3_eda/data/`
- Biểu đồ chuỗi thời gian → `outputs/round_03_B3_eda/figures/`
- Bảng phân tích → `outputs/round_03_B3_eda/tables/`

---

#### **Vòng 4: EDA cho C3 (Audio)** ✅ (Đã hoàn thành)
```bash
python notebooks/03_C3_urbansound8k_feature_extraction_eda.py
```
**Kết quả:**
- Feature audio → `outputs/round_04_C3_audio_eda/data/`
- Biểu đồ waveform, spectrogram → `outputs/round_04_C3_audio_eda/figures/`
- Bảng chất lượng audio → `outputs/round_04_C3_audio_eda/tables/`

---

#### **Vòng 5: Machine Learning** 🔄 (Đang thực hiện)
Chạy các script ML theo thứ tự:

```bash
# A3: Classification
python notebooks/chapter4_A3_classification.py

# A3: Clustering
python notebooks/chapter4_A3_clustering.py

# B3: Regression
python notebooks/chapter4_B3_regression.py

# B3: Clustering
python notebooks/chapter4_B3_clustering.py

# C3: Clustering
python notebooks/chapter4_C3_clustering.py
```
**Kết quả:** Mô hình, metrics, biểu đồ → `outputs/round_05_chapter4_ml/`

---

#### **Vòng 6: Tổng hợp báo cáo** ⏳ (Chưa thực hiện)
Tổng hợp tất cả kết quả vào báo cáo cuối cùng.

---

### ⚡ Chạy toàn bộ một lần (tùy chọn)
```bash
# Chạy tất cả script kiểm kê
for script in notebooks/0*_*_audit.py; do python "$script"; done

# Chạy tất cả script EDA
for script in notebooks/0*_*_eda*.py; do python "$script"; done

# Chạy tất cả script ML
for script in notebooks/chapter4_*.py; do python "$script"; done
```

---

## 📝 Kiểm tra Requirements

### Kiểm tra thư viện đã cài chưa:
```bash
pip list | findstr pandas numpy matplotlib scikit-learn librosa
```

### Cài đặt hoặc cập nhật:
```bash
pip install -r requirements.txt --upgrade
```

### Gỡ cài đặt:
```bash
pip uninstall -y -r requirements.txt
```

---

## ⚠️ Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-----------|---------|
| `ModuleNotFoundError: No module named 'pandas'` | Thư viện chưa cài | Chạy `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/...` | Đường dẫn không tìm thấy | Kiểm tra file tồn tại trong `data/raw/` |
| `MemoryError` | Dataset quá lớn, RAM không đủ | Giảm số dòng xử lý hoặc sử dụng `chunk` |
| `Cannot read audio file` | File WAV bị lỗi hoặc không tồn tại | Kiểm tra file audio trong `C3_urbansound8k/fold*` |
| `KeyError: column name` | Cột dữ liệu không tồn tại | Kiểm tra tên cột trong dữ liệu gốc |
| `RuntimeWarning` | Cảnh báo từ thư viện | Có thể bỏ qua nếu script chạy xong |

---

## 🛠️ Các lệnh hữu ích

### Xem dữ liệu nhanh:
```bash
# Xem 5 dòng đầu tiên
python -c "import pandas as pd; print(pd.read_csv('data/raw/A3_customer_personality/marketing_campaign.csv').head())"
```

### Kiểm tra dung lượng thư mục:
```bash
# Windows PowerShell
Get-ChildItem -Path "outputs" -Recurse | Measure-Object -Sum -Property Length

# Windows CMD
dir /s outputs
```

### Xóa tất cả output cũ (cẩn thận!):
```bash
# Windows PowerShell
Remove-Item "outputs\*" -Recurse -Force

# macOS/Linux
rm -rf outputs/*
```

### Chạy script với log:
```bash
# Lưu output vào file log
python notebooks/01_A3_customer_personality_eda.py > logs/A3_run.log 2>&1
```

---

## ❓ Câu hỏi thường gặp (FAQ)

**Q: Mình bắt đầu từ đâu?**
A: Hãy làm theo "Hướng dẫn chạy code" từ Vòng 1 đến Vòng 5 theo thứ tự.

**Q: Mình có thể chạy riêng từng vòng không?**
A: Có, nhưng khuyến khích chạy theo thứ tự vì mỗi vòng phụ thuộc vào kết quả của vòng trước.

**Q: Output nằm ở đâu?**
A: Tất cả kết quả được lưu trong thư mục `outputs/` theo từng vòng (round_01, round_02, ...).

**Q: Mình muốn xóa kết quả cũ và chạy lại?**
A: Xóa thư mục `outputs/` và chạy lại script. Hoặc xóa riêng thư mục round cụ thể.

**Q: Mất bao lâu để chạy hết?**
A: Khoảng 10-30 phút tùy máy tính, đặc biệt là Vòng 5 (ML) chậm nhất.

**Q: Mình có thể chỉnh sửa code không?**
A: Hoàn toàn được, nhưng lưu ý là một số output có thể thay đổi hoặc không tạo ra.

**Q: Làm sao biết script chạy xong?**
A: Khi không có lỗi (error) và terminal trở về dòng nhập lệnh.

---

## 📊 Dữ liệu và kích thước

| Dataset | Loại | Dòng | Cột | Dung lượng |
|---------|------|------|-----|-----------|
| A3 | CSV | 2,240 | 29 | ~500 KB |
| B3 (day) | CSV | 731 | 16 | ~50 KB |
| B3 (hour) | CSV | 17,379 | 17 | ~1 MB |
| C3 | CSV + WAV | 8,732 | 10 | ~10 GB (tất cả file audio) |

---

## 🎓 Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ chính
- **pandas**: Xử lý dữ liệu bảng (CSV)
- **numpy**: Tính toán số học
- **matplotlib & seaborn**: Vẽ biểu đồ
- **scikit-learn**: Machine Learning
- **librosa & soundfile**: Xử lý audio
- **scipy**: Phân tích thống kê

---

## 📞 Liên hệ & Hỗ trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra lỗi trong mục "Xử lý lỗi thường gặp"
2. Xem log output trong thư mục `outputs/round_*/logs/`
3. Đảm bảo tất cả thư viện đã cài đúng: `pip install -r requirements.txt`
4. Thử chạy lại script
