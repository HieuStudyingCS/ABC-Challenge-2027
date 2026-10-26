# Báo cáo EDA dữ liệu UWB Radar

## 1. Phạm vi phân tích

Báo cáo này tổng hợp kết quả từ lần chạy EDA trên stream:

```text
/kaggle/input/datasets/hieu1lth/abc-2027-all-dataset/abc2027_uwb_train_v1/streams/train/stream_00.h5
```

Pipeline đã đọc dữ liệu từ dataset `cir` trong file HDF5 và sử dụng luồng
`radar_1` để thực hiện phân tích và trực quan hóa.

> Lưu ý: Kết quả hiện tại chỉ phản ánh `radar_1` của `stream_00.h5`, không
> phải thống kê tổng hợp của toàn bộ 13 stream train, 5 stream validation hoặc
> cả ba radar.

## 2. Kết quả tải dữ liệu

Log xác nhận stream đã được tải thành công:

```text
[INFO] Đã tải thành công stream từ:
/kaggle/input/datasets/hieu1lth/abc-2027-all-dataset/abc2027_uwb_train_v1/streams/train/stream_00.h5
```

Điều này xác nhận đường dẫn Kaggle hiện tại trỏ đúng đến file dữ liệu.

## 3. Kích thước và thời lượng dữ liệu

| Thông tin | Giá trị |
|---|---:|
| Số frames | 88.117 |
| Tần số lấy mẫu | 50 Hz |
| Thời lượng xấp xỉ | 1.762,3 giây |
| Thời lượng xấp xỉ theo phút | 29,4 phút |
| Số range bins | 368 |

Tính toán thời lượng:

```text
88.117 frames / 50 frames/giây = 1.762,34 giây
```

Số lượng range bins phù hợp với mô tả của dataset ABC2027, trong đó mỗi
frame CIR có 368 bins.

## 4. Thống kê biên độ

Kết quả in ra từ EDA:

```text
- Giá trị biên độ trung bình: 0.0000
- Độ lệch chuẩn (Std Dev): 0.0007
- Biên độ Min / Max: -0.0109 / 0.0115
```

### Diễn giải

- Biên độ trung bình rất gần 0.
- Độ lệch chuẩn là `0.0007`, cho thấy phần lớn giá trị tập trung trong khoảng
  nhỏ quanh mức trung tâm.
- Biên độ nhỏ nhất và lớn nhất lần lượt là `-0.0109` và `0.0115`.
- Dữ liệu có thể đã được chuẩn hóa hoặc đang biểu diễn phần biến thiên của
  tín hiệu quanh nền.
- Giá trị trung bình hiển thị là `0.0000` do pipeline chỉ định dạng bốn chữ
  số thập phân. Điều này không chứng minh giá trị trung bình chính xác bằng
  0.

Để xem giá trị trung bình với độ chính xác cao hơn, có thể sử dụng:

```python
print(f"{np.mean(r1):.10f}")
print(f"{np.std(r1):.10f}")
```

## 5. Phân tích heatmap

Pipeline tạo heatmap từ `radar_1` bằng cách hiển thị ma trận chuyển vị:

```python
axes[0].imshow(
    r1.T,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    extent=[0, frames / fs, 0, bins],
)
```

### Quan sát từ heatmap

- Phần lớn heatmap có màu xanh ngọc tương đối đồng đều.
- Có một số dải ngang rõ hơn ở một số range bin, đặc biệt quanh vùng
  khoảng 180–210 bins.
- Các dải kéo dài theo gần như toàn bộ trục thời gian có thể là thành phần
  nền hoặc phản xạ tĩnh của môi trường.
- Không thể quan sát rõ các biến đổi ngắn theo thời gian trên toàn bộ heatmap
  vì gần 30 phút dữ liệu được nén vào cùng một hình.
- Một sự kiện ngã kéo dài ngắn có thể bị làm mờ khi hiển thị trên heatmap
  toàn stream.

### Ý nghĩa của heatmap

Heatmap cho thấy dữ liệu có cấu trúc theo range bin và có nền tín hiệu tương
đối ổn định. Tuy nhiên, heatmap tổng thể chưa đủ để xác định vị trí hoặc thời
điểm của từng sự kiện ngã.

## 6. Phân tích PSD

PSD được tính tại range bin trung tâm của `radar_1`:

```python
f, Pxx = signal.welch(
    r1[:, bins // 2],
    fs=fs,
    nperseg=256,
)
```

Kết quả trực quan cho thấy:

- Năng lượng phổ lớn nhất tập trung gần `0 Hz`.
- PSD giảm nhanh khi tần số tăng.
- Năng lượng ở vùng tần số cao hơn nhỏ hơn đáng kể so với vùng gần 0 Hz.
- Trục tần số được hiển thị từ `0` đến `25 Hz`, tương ứng với tần số Nyquist
  của dữ liệu lấy mẫu tại `50 Hz`.

### Diễn giải

Đỉnh gần `0 Hz` thường liên quan đến:

- Thành phần nền hoặc phản xạ tĩnh.
- Biến thiên rất chậm của tín hiệu.
- Xu hướng chậm theo thời gian.

Đỉnh này không phải là bằng chứng cho một sự kiện ngã. PSD tại một range bin
duy nhất cũng không đủ để kết luận về toàn bộ hoạt động trong stream.

## 7. Những gì EDA xác nhận được

Từ lần chạy này có thể xác nhận:

1. Đường dẫn Kaggle đến `stream_00.h5` đang hoạt động.
2. File chứa dữ liệu radar hợp lệ để pipeline đọc.
3. Stream có 88.117 frames và 368 range bins.
4. Tín hiệu có biên độ nhỏ và tập trung quanh 0.
5. Nền tín hiệu nhìn tổng thể tương đối ổn định.
6. Năng lượng phổ tại range bin trung tâm chủ yếu nằm ở vùng tần số thấp.

## 8. Những gì EDA chưa thể xác nhận

Kết quả EDA này chưa thể xác nhận:

- Stream có bao nhiêu sự kiện ngã.
- Thời điểm bắt đầu và kết thúc của từng sự kiện.
- Stream có người ngã hay không.
- Model có dự đoán đúng hay không.
- Chất lượng dự đoán theo F1, tIoU hoặc các metric của ABC2027.
- Đặc điểm của hai radar còn lại.
- Đặc điểm thống kê của toàn bộ dataset.

Để xác định số lượng và thời điểm sự kiện ngã, cần đối chiếu stream với file:

```text
streams_events.csv
```

Trong file nhãn, stream này thường được tham chiếu bằng:

```text
train_00
```

## 9. Liên hệ với kết quả phát hiện `[]`

Pipeline đã in:

```text
Các sự kiện ngã phát hiện được: []
```

Kết quả này không chứng minh rằng dataset không có sự kiện ngã. Nguyên nhân
chính là model hiện đang được khởi tạo với:

```python
model = HybridXCiTResNet(num_classes=2, pretrained=False)
```

Model hiện tại:

- Chưa được huấn luyện trên dataset ABC2027.
- Chưa load checkpoint đã huấn luyện.
- Đang sử dụng trọng số khởi tạo ngẫu nhiên.
- Chưa sử dụng nhãn từ `streams_events.csv` để huấn luyện hoặc đánh giá.

Do đó, kết quả `[]` chỉ cho biết pipeline inference không trích xuất được
event theo chuỗi trạng thái hiện tại; nó chưa phải kết luận về nội dung nhãn
của dataset.

## 10. Kết luận tổng quát

`stream_00.h5` là một stream UWB radar hợp lệ, dài khoảng 29,4 phút, gồm
88.117 frames ở tần số 50 Hz và 368 range bins trên mỗi frame. Tín hiệu
`radar_1` có biên độ nhỏ, nền tương đối ổn định và năng lượng phổ tập trung
chủ yếu ở vùng tần số thấp.

Kết quả EDA phù hợp với dữ liệu radar đã được xử lý hoặc chuẩn hóa. Tuy
nhiên, để phát hiện fall thực tế cần thực hiện thêm các bước:

1. Đọc nhãn từ `streams_events.csv`.
2. Gán nhãn cho các sliding window.
3. Huấn luyện model hoặc load checkpoint đã huấn luyện.
4. Chạy inference trên từng stream.
5. Đánh giá bằng metric chính thức của ABC2027.

