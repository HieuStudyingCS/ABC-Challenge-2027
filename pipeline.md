### 1. Sơ đồ Kiến trúc Pipeline Tổng thể (ASCII Diagram)

```text
========================================================================================================
                                 INFERENCE PIPELINE FOR 9TH ABC CHALLENGE
========================================================================================================

 [Luồng dữ liệu Continuous Stream]
 ├── Radar 1: Raw Signals (368 bins, 50 Hz)
 ├── Radar 2: Raw Signals (368 bins, 50 Hz)
 └── Radar 3: Raw Signals (368 bins, 50 Hz)
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KHỐI 1: PREPROCESSING & BANDPASS FILTERING                                                           │
│  - Áp dụng Chebychev Type I / Butterworth Bandpass Filter (1–5 Hz) trên từng bin theo thời gian  │
│  - Cắt bỏ 50 frames dao động quá độ ban đầu (Filter Impulse Response)                           │
│  - Chuẩn hóa Z-score / Min-Max theo cột (Fast-time Bins)                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KHỐI 2: SLIDING WINDOW & 3-CHANNEL MATRIX CREATION                                                   │
│  - Quét cửa sổ trượt T = 4.0s (200 frames) với bước trượt Hop Size = 0.2s (10 frames, overlap 95%)  │
│  - Ghép tín hiệu 3 radar thành Tensor 3 kênh: (3, 200, 368) tại mỗi mốc thời gian t          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KHỐI 3: BACKBONE CLASSIFIER (ResNet-18 / Hybrid ViT)                                                 │
│  - Fast Feature Extractor: ResNet-18 (Global Features) + XCiT Transformer (Local Patches)   │
│  - Softmax Layer: Xuất xác suất ngã P_fall(t) liên tục theo từng mốc thời gian t                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │  Chuỗi xác suất liên tục: P_fall(t_1), P_fall(t_2), ..., P_fall(t_N)
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KHỐI 4: TEMPORAL LOCALIZER (HMM - VITERBI DECODING)                                                  │
│  - Trạng thái ẩn: S = {0: Normal/ADL, 1: Pre-Fall/Impact, 2: Lying-Still}                            │
│  - Viterbi Algorithm: Tìm đường đi trạng thái tối ưu, làm mịn nhiễu và định vị ranh giới      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │  Khoảng ứng viên thô: [t_start_raw, t_end_raw]
                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KHỐI 5: POST-PROCESSING & EVENT FILTERING (NMS)                                                      │
│  - Bỏ sự kiện có độ dài quá ngắn (< 0.8 giây - báo động giả từ va đập/ngồi nhanh)                   │
│  - Non-Maximum Suppression (NMS): Gộp các khoảng cảnh báo bị chồng lấp (< 1.5s)                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
 [File kết quả JSONL nộp bài]
  {"start": 120.4, "end": 123.8, "label": "fall"}
========================================================================================================
```

---

### 2. Mô tả Chi tiết từng Khối Kỹ thuật

#### **Khối 1: Preprocessing & Bandpass Filtering**
* **Mục tiêu:** Loại bỏ nhiễu môi trường, nhiễu tĩnh từ đồ đạc/tường và phản xạ từ các chuyển động không phải của con người.
* **Thực thi:** Dùng bộ lọc dải thông **Chebychev Type I dải 1–5 Hz** tác động lên trục Slow-time của từng bin thuộc 3 radar (mỗi frame gồm 368 bins ở tần số 50 Hz). Chuẩn hóa biên độ tín hiệu bằng Z-score \\($z = \frac{x - \mu}{\sigma}$\\).

#### **Khối 2: Sliding Window & 3-Channel Scatter Matrix**
* **Mục tiêu:** Chuyển đổi dữ liệu chuỗi continuous stream thành các tensor hình ảnh 3 kênh.
* **Thực thi:** Kích thước cửa sổ trượt \\($W = 200 \text{ frames}$\\) (4.0 giây) với stride \\($S = 10 \text{ frames}$\\) (0.2 giây). Mỗi window tạo ra một ma trận dạng `(3, 200, 368)` đóng vai trò tương tự ảnh RGB 3 kênh (chứa thông tin của Radar 1, Radar 2 và Radar 3).

#### **Khối 3: Backbone Classifier (ResNet-18 / Hybrid ViT)**
* **Mục tiêu:** Trích xuất đặc trưng hình học toàn cục (qua ResNet-18) và tương tác ma trận cục bộ (qua Cross-Covariance Attention của XCiT) để dự đoán xác suất ngã tại từng cửa sổ.
* **Đầu ra:** Một mảng xác suất \\($P = [p_1, p_2, \dots, p_N]$\\) biểu diễn nguy cơ ngã dọc theo trục thời gian của luồng stream continuous.

#### **Khối 4: Temporal Localizer (HMM - Viterbi Decoding)**
* **Mục tiêu:** Khắc phục hạn chế của thresholding cứng bằng cách mô hình hóa chuỗi chuyển động theo thời gian.
* **Thực thi:** Khai báo 3 trạng thái ẩn:
  * \\($S_0$\\): Activity of Daily Living (ADL)
  * \\($S_1$\\): Fall / Impact (Chuyển động rơi/va chạm sàn)
  * \\($S_2$\\): Lying Still (Nằm yên sau khi ngã)
  
  Mô hình dùng ma trận chuyển trạng thái \\(A\\) (State Transition Matrix) quy định rằng từ \\($S_0 \rightarrow S_1 \rightarrow S_2$\\). Algorithmic **Viterbi Decoding** sẽ giải chuỗi quan sát \\(P\\) để xuất ra đường đi trạng thái ẩn tối ưu, xác định mốc `start` khi chuyển sang \\(S_1\\) và mốc `end` khi kết thúc \\(S_2\\).

#### **Khối 5: Hậu xử lý NMS & Xuất kết quả**
* **Mục tiêu:** Lọc bỏ các dự đoán ngắn bất thường và gộp các khung thời gian bị đè nhau.
* **Thực thi:** Loại bỏ bất kỳ chuỗi ngã nào có thời lượng \\($t_{\text{end}} - t_{\text{start}} < 0.8\text{s}$\\). Sử dụng thuật toán NMS để gộp các sự kiện có khoảng cách nhỏ hơn 1.5s thành một sự kiện duy nhất.

---

### 3. Mã nguồn Python Minh họa (PyTorch & HMM Skeleton)

Dưới đây là đoạn mã Python mẫu triển khai các khối cốt lõi trong pipeline:

```python
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torchvision.models as models
import timm

# ==========================================
# KHỐI 1: FILTERING (Chebychev Type I 1-5 Hz)
# ==========================================
def apply_chebychev_filter(radar_data, fs=50.0, low=1.0, high=5.0):
    """
    radar_data: numpy array shape (Frames, Bins)
    """
    nyq = 0.5 * fs
    b, a = signal.cheby1(N=4, rp=0.5, Wn=[low/nyq, high/nyq], btype='band')
    # Lọc dải thông dọc theo trục Slow-time (axis 0)
    filtered_data = signal.filtfilt(b, a, radar_data, axis=0)
    # Cắt bỏ 50 frames nhiễu dao động ban đầu
    return filtered_data[50:]


# ==========================================
# KHỐI 3: HYBRID ViT BACKBONE (XCiT + ResNet-18)
# ==========================================
class HybridXCiTResNet(nn.Module):
    """
    Kiến trúc Hybrid ViT + ResNet-18 theo bài báo của Abudalfa & Bouchard (2025):
    - Nhánh 1: ResNet-18 trích xuất đặc trưng toàn cục (Global features) -> 2 logits
    - Nhánh 2: XCiT trích xuất đặc trưng cục bộ (Local features) -> 2 logits
    - Late Fusion: Concatenate (4) -> Linear(4, 1024) -> ReLU -> Linear(1024, 2)
    """
    def __init__(self, num_classes=2, pretrained=False):
        super(HybridXCiTResNet, self).__init__()
        
        # 1. Nhánh ResNet-18 (3 kênh đầu vào tương ứng 3 radar UWB)
        self.resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # 2. Nhánh XCiT (Sử dụng kiến trúc XCiT với patch_size=8, depth=24 như bài báo)
        self.xcit = timm.create_model(
            'xcit_small_24_p8_224',
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes
        )
        
        # 3. Nhánh Late Fusion Classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes * 2, 1024), # Nối 2 logits từ ResNet + 2 logits từ XCiT = 4
            nn.ReLU(),
            nn.Linear(1024, num_classes)      # Đầu ra phân loại nhị phân (0: Normal, 1: Fall)
        )

    def forward(self, x):
        # x shape: (Batch, 3, 200, 368)
        logits_resnet = self.resnet(x)        # Shape: (Batch, 2)
        logits_xcit = self.xcit(x)            # Shape: (Batch, 2)
        
        # Hợp nhất (Late Fusion)
        fused = torch.cat([logits_resnet, logits_xcit], dim=1) # Shape: (Batch, 4)
        out = self.fusion_head(fused)         # Shape: (Batch, 2)
        return out


# ==========================================
# KHỐI 4: HMM VITERBI DECODER FOR LOCALIZATION
# ==========================================
class ViterbiPostProcessor:
    def __init__(self, trans_mat, emit_factor=1.0):
        # States: 0: Normal, 1: Fall/Impact, 2: Lying Still
        self.A = np.array(trans_mat) # Ma trận chuyển trạng thái (3x3)
        self.num_states = 3

    def decode(self, prob_sequence):
        """
        prob_sequence: List các xác suất P_fall(t) từ Backbone Classifier
        """
        T = len(prob_sequence)
        if T == 0:
            return np.array([], dtype=int)
            
        V = np.zeros((T, self.num_states))
        path = np.zeros((T, self.num_states), dtype=int)

        # [ĐÃ SỬA LỖI]: Gán vào hàng đầu tiên V[0] thay vì ghi đè cả mảng V
        V[0] = [0.9, 0.1, 0.0]

        for t in range(1, T):
            p_fall = prob_sequence[t]
            p_norm = 1.0 - p_fall
            # Emission probabilities dựa trên đầu ra mô hình DL
            emissions = [p_norm, p_fall * 0.7, p_fall * 0.3]

            for s in range(self.num_states):
                trans_probs = V[t-1] * self.A[:, s] * emissions[s]
                path[t, s] = np.argmax(trans_probs)
                V[t, s] = np.max(trans_probs)

        # Truy vết đường đi tối ưu (Backtracking)
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(V[-1])
        for t in range(T-2, -1, -1):
            best_path[t] = path[t+1, best_path[t+1]]

        return best_path


# ==========================================
# KHỐI 5: EXTRACT EVENTS & NMS
# ==========================================
def extract_events_from_states(state_sequence, frame_time=0.2, min_dur=0.8):
    events = []
    in_event = False
    start_t = 0.0

    for idx, st in enumerate(state_sequence):
        curr_time = idx * frame_time
        # [ĐÃ SỬA LỖI]: Kiểm tra trạng thái thuộc [1, 2] (Fall hoặc Lying Still)
        if st in [1, 2] and not in_event:
            in_event = True
            start_t = curr_time
        elif st == 0 and in_event:
            in_event = False
            end_t = curr_time
            if (end_t - start_t) >= min_dur:
                events.append({"start": round(start_t, 2), "end": round(end_t, 2), "label": "fall"})

    # Xử lý trường hợp sự kiện ngã kéo dài đến hết chuỗi dữ liệu
    if in_event:
        end_t = len(state_sequence) * frame_time
        if (end_t - start_t) >= min_dur:
            events.append({"start": round(start_t, 2), "end": round(end_t, 2), "label": "fall"})

    return events


# ==========================================
# MAIN INFERENCE PIPELINE RUNNER
# ==========================================
def run_pipeline(continuous_stream_data, model, device='cpu'):
    # Step 1: Preprocessing & Filter
    f_r1 = apply_chebychev_filter(continuous_stream_data['radar_1'])
    f_r2 = apply_chebychev_filter(continuous_stream_data['radar_2'])
    f_r3 = apply_chebychev_filter(continuous_stream_data['radar_3'])

    # Step 2 & 3: Sliding Window & Inference
    window_len = 200 # 4s at 50Hz
    stride = 10      # 0.2s
    prob_sequence = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(f_r1) - window_len, stride):
            w1 = f_r1[i:i+window_len]
            w2 = f_r2[i:i+window_len]
            w3 = f_r3[i:i+window_len]
            
            # Đóng gói 3 kênh (1, 3, 200, 368)
            tensor_in = torch.tensor(np.stack([w1, w2, w3]), dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(tensor_in)
            
            # [ĐÃ SỬA LỖI]: Lấy xác suất của class 1 (Fall) tại vị trí [0, 1]
            p_fall = torch.softmax(outputs, dim=1)[0, 1].item()
            prob_sequence.append(p_fall)

    # Step 4: HMM Viterbi Decoding
    trans_matrix = [
        [0.95, 0.05, 0.00],  # Normal -> Normal / Pre-Fall
        [0.00, 0.80, 0.20],  # Pre-Fall -> Pre-Fall / Lying
        [0.05, 0.00, 0.95]   # Lying -> Normal / Lying
    ]
    decoder = ViterbiPostProcessor(trans_matrix)
    state_seq = decoder.decode(prob_sequence)

    # Step 5: Post-processing & JSONL Export
    detected_events = extract_events_from_states(state_seq, frame_time=0.2)
    return detected_events


# ==========================================
# KHỞI TẠO MÔ HÌNH VÀ CHẠY THỬ (DEMO)
# ==========================================
if __name__ == "__main__":
    # 1. Khởi tạo mô hình Hybrid ViT (XCiT + ResNet-18)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hybrid_model = HybridXCiTResNet(num_classes=2, pretrained=False)
    
    # Nếu bạn đã có file trọng số đã huấn luyện (checkpoint), nạp vào tại đây:
    # hybrid_model.load_state_dict(torch.load("hybrid_xcit_resnet_weights.pth", map_location=device))

    # 2. Tạo dữ liệu giả lập (1000 frames ~ 20 giây ở tần số 50Hz, 368 range bins)
    dummy_stream = {
        'radar_1': np.random.randn(1000, 368),
        'radar_2': np.random.randn(1000, 368),
        'radar_3': np.random.randn(1000, 368)
    }

    # 3. Chạy toàn bộ Pipeline
    events = run_pipeline(dummy_stream, hybrid_model, device=device)
    print("Các sự kiện ngã phát hiện được:", events)
```

Nếu áp dụng pipeline đề xuất (kết hợp **Pre-filtering + ResNet-18/Hybrid ViT Backbone + HMM-Viterbi Decoding + NMS**), dưới đây là phân tích chi tiết về **Ưu/Nhược điểm** và các **Điểm then chốt cần tinh chỉnh** khi mô hình hoạt động chưa như kỳ vọng:

---

### 1. Ưu điểm và Nhược điểm của Pipeline

#### **Ưu điểm (Pros):**
* **Giải quyết đúng bản chất bài toán Continuous Stream:** Chuyển đổi thành công từ bài toán phân loại đoạn tĩnh (binary classification) sang bài toán định vị ranh giới thời gian `[start, end]` thực tế bằng cách kết hợp trích xuất đặc trưng mức cửa sổ với mô hình hóa chuỗi thời gian.
* **Tận dụng các Backbone học sâu mạnh mẽ:** Tận dụng được các kiến trúc SOTA như ResNet-18 hay Hybrid ViT đã được chứng minh đạt độ chính xác ~98%–99% trong việc nhận diện ma trận tán xạ UWB Radar.
* **Khả năng lọc nhiễu chuẩn hóa:** Lọc dải thông Chebychev Type I (1–5 Hz) loại bỏ hiệu quả nhiễu môi trường và nhiễu tầng sóng tĩnh, giúp biên độ tín hiệu người chuyển động nổi bật rõ rệt.
* **Làm mịn chuỗi và giảm Cảnh báo giả (False Positives) nhờ HMM:** Khác với việc dùng ngưỡng cứng (thresholding) dễ bị rung lắc tín hiệu, Viterbi Decoding bắt buộc đường đi trạng thái phải tuân theo quy luật vật lý của cú ngã (\\(\text{ADL} \rightarrow \text{Impact} \rightarrow \text{Lying Still}\\)), loại bỏ các đỉnh xác suất vọt lên ngắn hạn do nhiễu.
* **Cấu trúc mô hình dạng mô-đun (Decoupled Architecture):** Các khối Preprocessing, Backbone, Localizer, và Post-processing độc lập với nhau, giúp bạn dễ dàng cô lập lỗi, kiểm thử và thay thế từng phần mà không cần huấn luyện lại toàn bộ hệ thống từ đầu.

#### **Nhược điểm (Cons):**
* **Chi phí tính toán và thời gian suy luận (Inference Time) lớn:** Việc chạy cửa sổ trượt quá dày (overlap 90–95%) bằng các mô hình nặng như Hybrid ViT (~103M tham số) trên luồng continuous kéo dài hơn 6 tiếng của tập Train sẽ tốn nhiều tài nguyên GPU và thời gian xử lý.
* **Phụ thuộc vào Ma trận chuyển trạng thái (Transition Matrix) của HMM:** Nếu ma trận chuyển trạng thái \\(A\\) được thiết lập bằng tay (heuristic), nó có thể không linh hoạt trước sự đa dạng của các kịch bản ngã (ví dụ: ngã chậm, ngã cuộn, ngã từ ghế).
* **Độ nhòe ranh giới thời gian (Temporal Boundary Blurring):** Việc trích xuất đặc trưng trên cửa sổ 4 giây có thể làm ranh giới `start` và `end` bị "nhòe" đi vài khung hình, làm giảm chỉ số Temporal IoU khi chấm điểm.

---

### 2. Các vị trí cần tinh chỉnh (Tuning) nếu Pipeline hoạt động chưa tốt

Nếu kết quả đánh giá (F1-score / Temporal IoU) trên tập Validation chưa cao, bạn nên tiến hành tinh chỉnh theo 4 nhóm mắt xích sau:

#### **A. Khối Tiền xử lý (Preprocessing & Filtering)**
* **Dải tần lọc Bandpass:** Mặc định dải tần là **1–5 Hz**. 
  * *Tinh chỉnh:* Nếu cú ngã diễn ra quá nhanh (chứa tần số cao) hoặc ngã rất chậm bị lọc mất, hãy thử mở rộng dải tần sang **0.5–8 Hz**. Bạn cũng có thể thử bộ lọc **Butterworth** thay cho Chebychev nếu gặp hiện tượng méo pha tín hiệu (phase distortion).
* **Số frames cắt bỏ dao động quá độ (Transient Response):**
  * *Tinh chỉnh:* Kiểm tra đáp ứng xung của filter để điều chỉnh số frames cắt bỏ từ **30 đến 70 frames** (thay vì cố định 50 frames), tránh trường hợp cắt phạm vào cú ngã xuất hiện ngay đầu file stream.

#### **B. Khối Quét cửa sổ (Sliding Window Configuration)**
* **Kích thước cửa sổ (\\(W\\) - Window Size):** 
  * *Tinh chỉnh:* Mặc định là 4.0s (200 frames). Nếu chỉ số Temporal IoU bị thấp do ranh giới thời gian không sắc nét, hãy **giảm kích thước cửa sổ xuống 2.0s hoặc 3.0s** (100–150 frames). Cửa sổ nhỏ hơn giúp mô hình định vị điểm bắt đầu/kết thúc chính xác hơn.
* **Bước trượt (\\(S\\) - Stride / Hop Size):**
  * *Tinh chỉnh:* Giảm bước trượt từ 0.2s xuống **0.1s (5 frames)** để tăng độ phân giải thời gian cho chuỗi xác suất \\($P_{\text{fall}}(t)$\\).

#### **C. Khối Backbone Classifier**
* **Chuyển sang Phân loại đa lớp (Multi-class Classification):**
  * *Tinh chỉnh:* Nếu mô hình bị báo động giả nhiều do các hành động "Ngồi xuống nhanh" hay "Nằm xuống giường", hãy đổi nhãn huấn luyện cho Backbone từ nhị phân thành **3 lớp: `[0: ADL/Di chuyển, 1: Ngồi/Nằm nhanh, 2: Ngã]`**. Việc này giúp Backbone học cách tách biệt rõ ràng tín hiệu của hành động Ngồi với Ngã.
* **Thay thế Backbone nhẹ hơn:**
  * *Tinh chỉnh:* Thay thế Hybrid ViT cồng kềnh bằng **ResNet-18** hoặc **ConvNeXt-Tiny**. Các mô hình này nhẹ hơn từ 5–10 lần, giúp tăng tốc độ suy luận gấp nhiều lần mà độ chính xác trích xuất đặc trưng vẫn giữ vững ở mức >95%.

#### **D. Khối HMM & Hậu xử lý (NMS)**
* **Học ma trận chuyển trạng thái \\(A\\) từ dữ liệu Train:**
  * *Tinh chỉnh:* Thay vì gán cứng ma trận \\(A\\), hãy dùng thuật toán **Baum-Welch (EM)** hoặc đếm tần suất chuyển dịch trạng thái trực tiếp từ 478 sự kiện ground-truth của tập Train để thu được ma trận xác suất chuyển trạng thái tối ưu nhất.
* **Tinh chỉnh tham số gộp NMS:**
  * *Tinh chỉnh:* Điều chỉnh ngưỡng thời gian gộp (`merge_gap`) từ **1.0s đến 2.0s** và ngưỡng thời lượng tối thiểu (`min_duration`) dựa trên biểu đồ phân bố độ dài sự kiện (Event Duration Distribution) đã thống kê ở bước EDA.
