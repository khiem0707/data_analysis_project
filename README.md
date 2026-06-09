# Data Analysis Report Project - README cập nhật sau Vòng 4

## 1. Trạng thái hiện tại

Dự án đang đi theo 6 vòng đã chốt:

1. **Vòng 1 - Chuẩn bị và kiểm kê dữ liệu A3, B3, C3**: Đã hoàn thành.
2. **Vòng 2 - EDA cho A3 Customer Personality Analysis**: Đã hoàn thành.
3. **Vòng 3 - EDA cho B3 Bike Sharing Dataset**: Đã hoàn thành.
4. **Vòng 4 - Audio feature extraction và EDA cho C3 UrbanSound8K**: Đã hoàn thành.
5. **Vòng 5 - Machine Learning**: Chuẩn bị thực hiện.
6. **Vòng 6 - Tổng hợp báo cáo**: Chưa thực hiện.

---

## 2. Cấu trúc output hiện tại

```text
outputs/
|-- round_01_data_audit/
|   |-- A3/
|   |   |-- tables/
|   |   |-- logs/
|   |-- B3/
|   |   |-- tables/
|   |   |-- logs/
|   |-- C3/
|       |-- tables/
|       |-- logs/
|
|-- round_02_A3_eda/
|   |-- data/
|   |-- figures/
|   |   |-- raw/
|   |   |-- cleaned/
|   |-- logs/
|   |-- tables/
|       |-- raw/
|       |-- cleaned/
|       |-- comparison/
|
|-- round_03_B3_eda/
|   |-- data/
|   |-- figures/
|   |   |-- raw/
|   |   |-- transformed/
|   |-- logs/
|   |-- tables/
|       |-- raw/
|       |-- transformed/
|       |-- comparison/
|
|-- round_04_C3_audio_eda/
|   |-- data/
|   |-- figures/
|   |   |-- dataset_level/
|   |   |-- class_level/
|   |   |-- waveform_spectrogram/
|   |-- logs/
|   |-- tables/
|       |-- dataset_level/
|       |-- class_level/
|       |-- quality_check/
```

---

## 3. Vòng 1 - Kết quả kiểm kê dữ liệu

### 3.1. A3 - Customer Personality Analysis

- Dữ liệu chính: `data/raw/A3_customer_personality/marketing_campaign.csv`
- Số dòng: 2,240.
- Số cột: 29.
- Duplicate rows: 0.
- Missing value: chỉ có `Income` thiếu 24 giá trị, tương ứng 1.071429%.
- Target `Response`:
  - `Response = 0`: 1,906 mẫu, tương ứng 85.089286%.
  - `Response = 1`: 334 mẫu, tương ứng 14.910714%.
- Cột hằng số:
  - `Z_CostContact`
  - `Z_Revenue`

### 3.2. B3 - Bike Sharing Dataset

- `day.csv`: 731 dòng, 16 cột.
- `hour.csv`: 17,379 dòng, 17 cột.
- Duplicate rows: 0.
- Constant columns: không có.
- Missing values: không có.
- Date range: 2011-01-01 đến 2012-12-31.
- Số ngày duy nhất: 731.
- Biến mục tiêu `cnt` trong `hour.csv`:
  - Count: 17,379
  - Mean: 189.463088
  - Std: 181.387599
  - Min: 1
  - Q1: 40
  - Median: 142
  - Q3: 281
  - Max: 977

### 3.3. C3 - UrbanSound8K

- Metadata: 8,732 dòng.
- File WAV tìm thấy: 8,732.
- Tất cả metadata rows đều có file audio tương ứng.
- Missing values: 0.
- Duplicate theo `slice_file_name` và `fold`: 0.
- Số lớp âm thanh: 10.
- Lớp ít mẫu nhất:
  - `gun_shot`: 374 mẫu.
  - `car_horn`: 429 mẫu.
- Duration metadata:
  - Mean: 3.607904 giây.
  - Std: 0.973570.
  - Min: 0.054517.
  - Q1/Median/Q3/Max: 4.0 giây.

---

## 4. Vòng 2 - EDA cho A3 Customer Personality Analysis

### 4.1. Script đã chạy

```bash
python notebooks/01_A3_customer_personality_eda.py
```

### 4.2. Output chính

```text
outputs/round_02_A3_eda/
|-- data/
|   |-- A3_cleaned.csv
|   |-- A3_cleaned_with_scaled_features.csv
|-- logs/
|   |-- A3_round2_eda_log.txt
|-- tables/
|   |-- raw/
|   |-- cleaned/
|   |-- comparison/
|-- figures/
|   |-- raw/
|   |-- cleaned/
```

### 4.3. Kết quả chính

- Raw shape sau khi tạo biến dẫn xuất: 2,240 dòng, 36 cột.
- Missing `Income` trước xử lý: 24.
- Median dùng để điền `Income`: 51,381.5.
- Missing `Income` sau xử lý: 0.
- Đã tạo các biến:
  - `Age`
  - `Customer_Tenure_Days`
  - `Total_Spending`
  - `Total_Children`
  - `Total_Purchases`
  - `Total_Accepted_Campaigns`
- Đã gộp `Alone`, `Absurd`, `YOLO` trong `Marital_Status` thành `Other`.
- Đã xóa các cột hằng số:
  - `Z_CostContact`
  - `Z_Revenue`
- Đã làm mịn outliers theo quy tắc IQR.
- Cleaned shape: 2,240 dòng, 35 cột.

---

## 5. Vòng 3 - EDA cho B3 Bike Sharing Dataset

### 5.1. Script đã chạy

```bash
python notebooks/02_B3_bike_sharing_eda_timeseries.py
```

### 5.2. Output chính

```text
outputs/round_03_B3_eda/
|-- data/
|   |-- B3_day_raw_with_datetime.csv
|   |-- B3_hour_raw_with_datetime.csv
|   |-- B3_hour_transformed_time_features.csv
|   |-- B3_daily_aggregated_from_hour.csv
|-- logs/
|   |-- B3_round3_eda_log.txt
|-- tables/
|   |-- raw/
|   |-- transformed/
|   |-- comparison/
|-- figures/
|   |-- raw/
|   |-- transformed/
```

### 5.3. Kết quả chính

- `hour.csv` sau tạo `datetime`: 17,379 dòng, 18 cột.
- Khoảng thời gian: 2011-01-01 00:00:00 đến 2012-12-31 23:00:00.
- Missing values trong `hour.csv`: 0.
- Duplicate rows trong `hour.csv`: 0.
- Số bản ghi theo giờ kỳ vọng: 17,544.
- Số bản ghi theo giờ quan sát được: 17,379.
- Số mốc giờ bị thiếu: 165.
- Transformed hourly shape: 17,544 dòng, 43 cột.
- Missing `cnt_interpolated` sau xử lý: 0.

### 5.4. Biến đổi chuỗi thời gian đã tạo

- `datetime`
- Full hourly reindex
- Linear interpolation cho biến numeric
- Forward/backward fill cho biến categorical
- `cnt_log1p`
- `cnt_diff_1`
- `lag_1`
- `lag_24`
- `lag_168`
- `rolling_mean_24`
- `rolling_std_24`
- `rolling_mean_168`

---

## 6. Vòng 4 - Audio feature extraction và EDA cho C3 UrbanSound8K

### 6.1. Script đã chạy

```bash
python notebooks/03_C3_urbansound8k_feature_extraction_eda.py
```

### 6.2. Output chính

```text
outputs/round_04_C3_audio_eda/
|-- data/
|   |-- C3_audio_features.csv
|   |-- C3_metadata_with_audio_path.csv
|-- logs/
|   |-- C3_round4_audio_eda_log.txt
|-- tables/
|   |-- dataset_level/
|   |-- class_level/
|   |-- quality_check/
|-- figures/
|   |-- dataset_level/
|   |-- class_level/
|   |-- waveform_spectrogram/
```

### 6.3. Kết quả kiểm tra feature

- Metadata rows: 8,732.
- Feature rows: 8,732.
- Feature status ok: 8,732.
- Feature status error: 0.
- Unique classes: 10.
- Unique folds: 10.
- Existing audio files by metadata: 8,732/8,732.

### 6.4. Phân bố lớp từ file feature

| Class | Count |
|---|---:|
| dog_bark | 1000 |
| children_playing | 1000 |
| air_conditioner | 1000 |
| street_music | 1000 |
| jackhammer | 1000 |
| engine_idling | 1000 |
| drilling | 1000 |
| siren | 929 |
| car_horn | 429 |
| gun_shot | 374 |

### 6.5. Thống kê feature âm thanh chính

| Feature | Mean | Median | Std | Outliers IQR |
|---|---:|---:|---:|---:|
| duration_audio | 3.607522 | 4.000000 | 0.974394 | 1407 |
| rms_mean | 0.069614 | 0.051834 | 0.066487 | 447 |
| zcr_mean | 0.070410 | 0.050503 | 0.065309 | 806 |
| spectral_centroid_mean | 2839.990934 | 2348.069843 | 1669.250992 | 557 |
| spectral_bandwidth_mean | 3326.327284 | 3154.964867 | 1352.179034 | 123 |
| spectral_rolloff_mean | 5417.982096 | 4449.665272 | 3422.256138 | 245 |

### 6.6. Lưu ý khi diễn giải

- `duration_audio` có Q1 = Median = Q3 = 4 giây, nên IQR = 0.
- Vì vậy, các file có thời lượng khác 4 giây bị đánh dấu là outlier theo quy tắc IQR.
- Điều này không nhất thiết là lỗi dữ liệu; nó cho thấy phần lớn file audio đã được chuẩn hóa quanh 4 giây, còn một số file ngắn hơn hoặc dài hơn nhẹ.
- Các feature như `rms_mean`, `zcr_mean`, `spectral_centroid_mean` có CV cao, cho thấy dữ liệu âm thanh có mức biến động lớn giữa các lớp và giữa các file.

---

## 7. Bước tiếp theo

Bước tiếp theo là **Vòng 5 - Machine Learning**.

Vòng 5 sẽ gồm 3 phần lớn:

1. **A3 - Classification**
   - Thực hiện ít nhất 3 target khác nhau:
     - `Response`
     - `Education`
     - `Spending_Level`
   - So sánh tối thiểu 2 thuật toán.
   - Xuất confusion matrix, accuracy, precision, recall, F1-score.

2. **B3 - Regression / Forecasting**
   - Dự đoán `cnt` theo đúng time-series split.
   - Không dùng random split.
   - So sánh Linear Regression, Random Forest Regressor, Gradient Boosting/XGBoost nếu có.
   - Xuất MAE, RMSE, R2-score, actual vs prediction.

3. **A3/B3/C3 - Clustering**
   - Chạy K-Means, Hierarchical Clustering, DBSCAN.
   - Dùng Elbow Method/Silhouette Score.
   - Đọc vị cụm bằng thống kê mô tả.
   - Với C3 dùng feature audio đã trích xuất ở Vòng 4.
