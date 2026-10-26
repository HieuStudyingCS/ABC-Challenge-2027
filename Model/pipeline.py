from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
import h5py

# ==========================================
# KAGGLE PATHS
# ==========================================
# Dataset slug: hieu1lth/abc-2027-all-dataset
KAGGLE_DATASET_ROOT = Path("/kaggle/input/datasets/hieu1lth/abc-2027-all-dataset/abc2027_uwb_train_v1")
KAGGLE_TRAIN_STREAMS_DIR = KAGGLE_DATASET_ROOT / "streams" / "train"
KAGGLE_VAL_STREAMS_DIR = KAGGLE_DATASET_ROOT / "streams" / "val"
KAGGLE_SEGMENTS_DIR = KAGGLE_DATASET_ROOT / "segments"
KAGGLE_EVENTS_PATH = KAGGLE_DATASET_ROOT / "streams_events.csv"
KAGGLE_COMPOSITION_PATH = KAGGLE_DATASET_ROOT / "streams_composition.csv"
KAGGLE_SAMPLE_SUBMISSION_PATH = KAGGLE_DATASET_ROOT / "sample_submission.csv"
KAGGLE_TRAIN_STREAM_PATHS = sorted(KAGGLE_TRAIN_STREAMS_DIR.glob("stream_*.h5"))
KAGGLE_VAL_STREAM_PATHS = sorted(KAGGLE_VAL_STREAMS_DIR.glob("stream_*.h5"))
KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
KAGGLE_CHECKPOINT_DIR = KAGGLE_OUTPUT_DIR / "checkpoints"
KAGGLE_SUBMISSION_PATH = KAGGLE_OUTPUT_DIR / "submission.csv"
DEFAULT_DATASET_PATH = KAGGLE_TRAIN_STREAMS_DIR / "stream_00.h5"

# ==========================================
# KHỐI 0: DATASET LOADER & EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================
def load_uwb_radar_dataset(data_path=DEFAULT_DATASET_PATH):
    """
    Tải một stream UWB radar từ file HDF5 của ABC2027.

    Mỗi file stream lưu dữ liệu trong dataset ``cir`` với dạng
    ``(3, frames, 368)`` tương ứng với ba radar.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy stream tại '{data_path}'. "
            f"Các stream train khả dụng: {KAGGLE_TRAIN_STREAMS_DIR}"
        )

    if data_path.suffix.lower() == ".h5":
        with h5py.File(data_path, "r") as handle:
            if "cir" not in handle:
                raise KeyError(f"File HDF5 '{data_path}' không có dataset 'cir'.")
            cir = np.asarray(handle["cir"], dtype=np.float32)

        if cir.ndim != 3 or cir.shape[0] != 3:
            raise ValueError(
                f"Dataset 'cir' phải có dạng (3, frames, bins), nhận được {cir.shape}."
            )

        print(f"[INFO] Đã tải thành công stream từ: {data_path}")
        return {
            "radar_1": cir[0],
            "radar_2": cir[1],
            "radar_3": cir[2],
            "labels": np.zeros(cir.shape[1], dtype=np.int64),
        }

    if data_path.suffix.lower() == ".npz":
        data = np.load(data_path)
        required_keys = {"radar_1", "radar_2", "radar_3", "labels"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(f"File NPZ thiếu các key: {sorted(missing_keys)}")
        print(f"[INFO] Đã tải thành công dataset từ: {data_path}")
        return {
            "radar_1": data["radar_1"],
            "radar_2": data["radar_2"],
            "radar_3": data["radar_3"],
            "labels": data["labels"],
        }

    raise ValueError(
        f"Định dạng dataset chưa được hỗ trợ: '{data_path.suffix}'. "
        "Hãy sử dụng file .h5 hoặc .npz."
    )

def perform_eda(radar_data_dict, fs=50.0):
    """
    Thực hiện Exploratory Data Analysis (EDA) trên luồng dữ liệu Radar UWB:
    - Thống kê biên độ, phương sai theo thời gian (Slow-time).
    - Trực quan hóa bản đồ nhiệt (Heatmap) của biên độ tín hiệu thô.
    - Phân tích phổ mật độ công suất (PSD) để kiểm tra thành phần tần số chuyển động (1-5 Hz).
    """
    print("\n================ [EDA] THỰC KHẢO SÁT DỮ LIỆU RADAR ================")
    r1 = radar_data_dict['radar_1']
    frames, bins = r1.shape
    print(f"- Tổng số khung hình (Frames): {frames} (~{frames/fs:.1f} giây tại {fs}Hz)")
    print(f"- Số lượng khoảng cách (Range Bins): {bins}")
    print(f"- Giá trị biên độ trung bình: {np.mean(r1):.4f}")
    print(f"- Độ lệch chuẩn (Std Dev): {np.std(r1):.4f}")
    print(f"- Biên độ Min / Max: {np.min(r1):.4f} / {np.max(r1):.4f}")

    # Vẽ biểu đồ EDA cơ bản
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # 1. Heatmap tín hiệu thô theo thời gian và range bins
    im = axes[0].imshow(r1.T, aspect='auto', origin='lower', cmap='viridis', 
                        extent=[0, frames/fs, 0, bins])
    axes[0].set_title("EDA: Raw UWB Radar Heatmap (Slow-time vs Range Bins)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Range Bins")
    fig.colorbar(im, ax=axes[0], label="Amplitude")

    # 2. Phổ mật độ công suất trung bình (PSD) kiểm tra dải tần sinh học
    f, Pxx = signal.welch(r1[:, bins // 2], fs=fs, nperseg=256)
    axes[1].plot(f, Pxx, color='orange')
    axes[1].set_title("EDA: Power Spectral Density (PSD) at Center Range Bin")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_xlim(0, fs / 2)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    print("==================================================================\n")


# ==========================================
# KHỐI 1: FILTERING (Chebychev Type I 1-5 Hz)
# ==========================================
def apply_chebychev_filter(radar_data, fs=50.0, low=1.0, high=5.0):
    nyq = 0.5 * fs
    b, a = signal.cheby1(N=4, rp=0.5, Wn=[low/nyq, high/nyq], btype='band')
    filtered_data = signal.filtfilt(b, a, radar_data, axis=0)
    return filtered_data[50:]


# ==========================================
# KHỐI 3: HYBRID ViT BACKBONE (XCiT + ResNet-18)
# ==========================================
class HybridXCiTResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(HybridXCiTResNet, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        self.xcit = timm.create_model(
            'xcit_small_24_p8_224',
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        logits_resnet = self.resnet(x)
        logits_xcit = self.xcit(x)
        fused = torch.cat([logits_resnet, logits_xcit], dim=1)
        out = self.fusion_head(fused)
        return out


# ==========================================
# KHỐI 4: HMM VITERBI DECODER FOR LOCALIZATION
# ==========================================
class ViterbiPostProcessor:
    def __init__(self, trans_mat):
        self.A = np.array(trans_mat)
        self.num_states = 3

    def decode(self, prob_sequence):
        T = len(prob_sequence)
        if T == 0:
            return np.array([], dtype=int)
            
        V = np.zeros((T, self.num_states))
        path = np.zeros((T, self.num_states), dtype=int)

        V[0] = [0.9, 0.1, 0.0]

        for t in range(1, T):
            p_fall = prob_sequence[t]
            p_norm = 1.0 - p_fall
            emissions = [p_norm, p_fall * 0.7, p_fall * 0.3]

            for s in range(self.num_states):
                trans_probs = V[t-1] * self.A[:, s] * emissions[s]
                path[t, s] = np.argmax(trans_probs)
                V[t, s] = np.max(trans_probs)

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
        if st in [1, 2] and not in_event:
            in_event = True
            start_t = curr_time
        elif st == 0 and in_event:
            in_event = False
            end_t = curr_time
            if (end_t - start_t) >= min_dur:
                events.append({"start": round(start_t, 2), "end": round(end_t, 2), "label": "fall"})

    if in_event:
        end_t = len(state_sequence) * frame_time
        if (end_t - start_t) >= min_dur:
            events.append({"start": round(start_t, 2), "end": round(end_t, 2), "label": "fall"})

    return events


# ==========================================
# MAIN PIPELINE RUNNER (TÍCH HỢP EDA & PREPROCESSING)
# ==========================================
def run_pipeline(dataset_path=DEFAULT_DATASET_PATH, device='cpu'):
    # 1. Tải Dataset
    stream_data = load_uwb_radar_dataset(dataset_path)

    # 2. Chạy Khảo sát EDA (Phân tích phân bố & Phổ tần số)
    perform_eda(stream_data, fs=50.0)

    # 3. Tiền xử lý: Lọc dải thông Chebychev (1-5 Hz)
    f_r1 = apply_chebychev_filter(stream_data['radar_1'])
    f_r2 = apply_chebychev_filter(stream_data['radar_2'])
    f_r3 = apply_chebychev_filter(stream_data['radar_3'])

    # 4. Khởi tạo mô hình
    model = HybridXCiTResNet(num_classes=2, pretrained=False)
    model.to(device)
    model.eval()

    # 5. Sliding Window & Inference
    window_len = 200 # 4s at 50Hz
    stride = 10      # 0.2s
    prob_sequence = []

    with torch.no_grad():
        for i in range(0, len(f_r1) - window_len, stride):
            w1 = f_r1[i:i+window_len]
            w2 = f_r2[i:i+window_len]
            w3 = f_r3[i:i+window_len]
            
            tensor_in = torch.tensor(np.stack([w1, w2, w3]), dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(tensor_in)
            
            p_fall = torch.softmax(outputs, dim=1)[0, 1].item()
            prob_sequence.append(p_fall)

    # 6. HMM Viterbi Decoding
    trans_matrix = [
        [0.95, 0.05, 0.00],  # Normal
        [0.00, 0.80, 0.20],  # Pre-Fall
        [0.05, 0.00, 0.95]   # Lying
    ]
    decoder = ViterbiPostProcessor(trans_matrix)
    state_seq = decoder.decode(prob_sequence)

    # 7. Trích xuất sự kiện
    detected_events = extract_events_from_states(state_seq, frame_time=0.2)
    return detected_events


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    events = run_pipeline(dataset_path=DEFAULT_DATASET_PATH, device=device)
    print("Các sự kiện ngã phát hiện được:", events)