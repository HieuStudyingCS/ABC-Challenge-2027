# Báo cáo chạy baseline ABC2027 UWB

Trạng thái: **hoàn thành**. Đã chạy lại energy baseline và learned baseline logistic regression trên dữ liệu phát hành, sau đó chấm bằng evaluator chính thức với --ignore-flagged. Phạm vi chấm chỉ gồm 5 stream validation.

## Kết quả và đối chiếu README

| Baseline | Mean F1 @ 0.1/0.3/0.5 | F1 @ 0.1 | F1 @ 0.3 | F1 @ 0.5 | F1 @ 0.7 | README |
|---|---:|---:|---:|---:|---:|---|
| Energy | 0.315091 | 0.358209 | 0.338308 | 0.248756 | 0.059701 | 0.315; 0.358 / 0.338 / 0.249 / 0.060 |
| Logistic | 0.257886 | 0.593103 | 0.166667 | 0.013889 | 0.000000 | 0.258; 0.593 / 0.167 / 0.014 / 0.000 |

Hai baseline khớp toàn bộ số README khi làm tròn 3 chữ số. Energy prediction CSV cũng khớp từng dòng với sample_submission.csv.

### Chỉ số tại theta 0.3

| Baseline | TP | FP | FN | False alarms/giờ | Sai số bắt đầu MAE / median (s) | Sai số kết thúc MAE / median (s) | Mean tIoU | Latency mean / median (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Energy | 34 | 100 | 33 | 48.808 | 0.516 / 0.320 | 0.398 / 0.350 | 0.567456 | 0.453 / 0.310 |
| Logistic | 12 | 65 | 55 | 31.725 | 2.152 / 2.190 | 2.272 / 2.550 | 0.363183 | 1.868 / 1.810 |

Evaluator đánh dấu 52 nhãn ground truth là vùng bỏ qua. Ở theta 0.3, có 6 dự đoán energy và 9 dự đoán logistic được bỏ qua theo vùng này. Cả hai tệp prediction có cột t_alarm; latency trong báo cáo là t_alarm trừ thời điểm bắt đầu nhãn, không phải thời gian tính toán.

## Dữ liệu và cấu hình

- Train: 13 stream, tổng 6.181861 giờ; 359 sự kiện train có ignore=false.
- Validation: 5 stream, tổng 2.048833 giờ; 119 hàng nhãn validation. Tạo bản CSV lọc riêng trong thư mục tạm; giữ nguyên giá trị và thứ tự các hàng validation từ streams_events.csv.
- Logistic train dùng 22,209 cửa sổ 200 frame, hop 50; 1,711 cửa sổ được gán dương theo IoU >= 0.1. Dùng StandardScaler và LogisticRegression(max_iter=200, class_weight=balanced), ngưỡng 0.85.
- Energy dùng percentile 98, thời lượng tối thiểu 0.6 giây, gộp khoảng cách 0.8 giây và EMA nền 3 giây.
- configs/streams.yaml có seed 2027 cho việc dựng streams; baseline dùng streams phát hành sẵn. Mã logistic không đặt seed (random_state=None, solver mặc định lbfgs); không tìm siêu tham số.
- Chỉ đọc streams/train để fit logistic và streams/val để dự đoán/chấm. Không dùng test streams hoặc nạp dữ liệu trong segments/.

## Môi trường và giới hạn tái lập

Dùng Python 3.12.0 có sẵn, với NumPy 2.4.6, SciPy 1.18.0, h5py 3.14.0, scikit-learn 1.9.0 và joblib 1.5.3. Các phiên bản ghim trong requirements.txt khác: NumPy 2.5.3, SciPy 1.18.1, h5py 3.16.0, scikit-learn 1.7.2. Thử cài đúng phiên bản trong virtualenv bên trong thư mục tạm nhưng DNS không truy cập được kho package; vì vậy chạy bằng môi trường có sẵn. Sai khác phiên bản là giới hạn tái lập đã xác minh; kết quả vẫn khớp README ở độ chính xác công bố.

src/abc2027/baseline.py định nghĩa các hàm baseline nhưng không có CLI/entrypoint. Runner tạm gọi trực tiếp các hàm chính thức, theo cách notebook phát hành dựng feature/label và theo configs/baseline.yaml; không sửa hay thay thế mã baseline.

## Lệnh đã chạy

Runner:
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:\Users\Khoa Bi\Downloads\abc2027_uwb_train_v1\abc2027_uwb_train_v1\src'
& 'C:\Users\Khoa Bi\AppData\Local\Programs\Python\Python312\python.exe' -B 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\run_official_baselines.py' 'C:\Users\Khoa Bi\Downloads\abc2027_uwb_train_v1\abc2027_uwb_train_v1' 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b'

Evaluator (lặp riêng cho từng prediction; --streams trỏ thẳng vào validation):
& 'C:\Users\Khoa Bi\AppData\Local\Programs\Python\Python312\python.exe' -B -m abc2027.evaluate --gt 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\validation_events.csv' --pred 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\energy_predictions.csv' --streams 'C:\Users\Khoa Bi\Downloads\abc2027_uwb_train_v1\abc2027_uwb_train_v1\streams\val' --ignore-flagged --out 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\energy_report.json'
& 'C:\Users\Khoa Bi\AppData\Local\Programs\Python\Python312\python.exe' -B -m abc2027.evaluate --gt 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\validation_events.csv' --pred 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\logistic_predictions.csv' --streams 'C:\Users\Khoa Bi\Downloads\abc2027_uwb_train_v1\abc2027_uwb_train_v1\streams\val' --ignore-flagged --out 'C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b\logistic_report.json'

## Thời gian

- Energy inference trên validation: 23.42 giây.
- Logistic trích feature/đọc 13 stream train: 67.77 giây.
- Fit logistic: 0.05 giây.
- Logistic dự đoán validation: 11.23 giây.
- Evaluator chính thức: khoảng 0.815 giây cho mỗi baseline.

## Kiểm tra bảo toàn file

- SHA-256 của 13 file source/config/nhãn tham chiếu trước và sau không đổi.
- Danh mục 673 file dự án trước/sau: không thêm, xóa, đổi kích thước hoặc thời gian sửa.
- Bản nhãn validation tạm có 119 hàng, đúng nguyên giá trị/thứ tự; không có hàng train/test.
- git status không khả dụng: thư mục gói phát hành không phải Git repository. Không có lệnh Git nào thay đổi trạng thái.

Tất cả file mới nằm dưới thư mục tạm:

C:\Users\Khoa Bi\.codex\visualizations\2026\10\25\01a172be-4bf3-74f1-918a-0f37a1cd97c6\abc2027-run-bdd393912ede4d98a0f8c824c5c71a8b

Xem integrity_check.json và baseline_run_metadata.json để biết chi tiết kiểm tra và cấu hình.
