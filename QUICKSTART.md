# SAM2-MoE Quick Start Guide

빠르게 SAM2-MoE를 시작하는 방법을 안내합니다.

## 목차
1. [환경 설정](#환경-설정)
2. [모델 빌드 & 테스트](#모델-빌드--테스트)
3. [체크포인트 로드](#체크포인트-로드)
4. [Fine-tuning 시작](#fine-tuning-시작)
5. [Expert 분석](#expert-분석)

---

## 환경 설정

### 1. 저장소 클론 및 의존성 설치

```bash
# 이미 클론되어 있다면 스킵
cd /home/jinu/github.com/doldam0/samoe

# uv로 의존성 동기화
uv sync
```

### 2. SAM2 체크포인트 다운로드

```bash
# 체크포인트 다운로드 (이미 있다면 스킵)
bash checkpoints/download_ckpts.sh
```

현재 사용 가능한 체크포인트:
- ✓ `checkpoints/sam2.1_hiera_base_plus.pt` (309MB)
- ✓ `checkpoints/sam2.1_hiera_large.pt` (857MB)
- ✓ `checkpoints/sam2.1_hiera_small.pt` (176MB)
- ✓ `checkpoints/sam2.1_hiera_tiny.pt` (149MB)

---

## 모델 빌드 & 테스트

### 간단한 데모 실행

```bash
# MoE 모델 빌드 및 구조 확인
uv run python examples/sam2_moe_demo.py
```

**출력 예시:**
```
Building SAM2 with Mixture of Prompt Experts (MoPE)
✓ Model built successfully!

Model Architecture Summary
Total parameters: 89,123,456
Trainable parameters (MoE adapters): 4,567,890
Frozen parameters (base model): 84,555,566
Trainable ratio: 5.12%

MoE Structure Analysis
Found 8 MoE-enhanced attention modules:
  - Num experts: 10
  - LoRA rank: 4
  - Top-k: 2
```

---

## 체크포인트 로드

### Python 코드에서 사용

```python
from sam2.build_sam import build_sam2_video_predictor_moe

# MoE 모델 빌드 (base_plus 체크포인트 사용)
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
    device="cuda",
    mode="eval",  # 또는 "train"
)

# 파라미터 확인
total = sum(p.numel() for p in predictor.parameters())
trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

**중요 포인트:**
- ✅ Base model weights는 자동으로 frozen됨
- ✅ LoRA adapters와 gating networks만 학습 가능
- ✅ 약 5-10%의 파라미터만 trainable

---

## Fine-tuning 시작

### 방법 1: 간단한 예제 실행

```bash
# 간단한 training 데모 (dummy data 사용)
uv run python examples/simple_train_moe.py
```

이 스크립트는 다음을 보여줍니다:
1. ✓ MoE 모델 로드
2. ✓ Trainable parameter 확인
3. ✓ Optimizer 설정
4. ✓ Training loop 실행
5. ✓ MoE adapter 저장/로드

### 방법 2: 전체 Training 스크립트

```bash
# 실제 학습용 스크립트 (데이터셋 필요)
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_b+_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_base_plus.pt \
    --output_dir outputs/moe_training \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --use_dummy_data  # 테스트용 (실제 학습 시 제거)
```

### Training Configuration 커스터마이즈

`configs/training/train_moe_config.yaml` 파일을 수정하여 설정 변경:

```yaml
# MoE 하이퍼파라미터
moe:
  num_experts: 10      # Expert 개수
  lora_rank: 4         # LoRA rank
  top_k: 2             # 활성화할 expert 수
  lora_alpha: 1.0      # LoRA scaling

# Training 하이퍼파라미터
training:
  learning_rate: 1.0e-4
  batch_size: 2
  num_epochs: 10
```

---

## Expert 분석

### Expert Usage 모니터링

```python
from sam2.build_sam import build_sam2_video_predictor_moe
import torch

# 모델 로드
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
    device="cuda",
)

# MoE attention 모듈 찾기
moe_modules = []
for name, module in predictor.named_modules():
    if "MoERoPEAttention" in str(type(module).__name__):
        moe_modules.append((name, module))

print(f"Found {len(moe_modules)} MoE attention modules")

# Dummy input으로 expert usage 확인
dummy_q = torch.randn(1, 4096, 256).cuda()  # (B, N, D)
dummy_k = torch.randn(1, 4096, 256).cuda()
dummy_v = torch.randn(1, 4096, 256).cuda()

# 첫 번째 MoE 모듈의 expert statistics
name, moe_module = moe_modules[0]
with torch.no_grad():
    stats = moe_module.get_expert_statistics(dummy_q, dummy_k, dummy_v)

print(f"\nExpert usage for {name}:")
print("Q projection:")
for i, weight in enumerate(stats['q_proj']):
    print(f"  Expert {i}: {weight.item():.4f}")
```

### Visualization (optional)

```python
import matplotlib.pyplot as plt
import numpy as np

# Expert weights 시각화
weights = stats['q_proj'].cpu().numpy()

plt.figure(figsize=(10, 5))
plt.bar(range(len(weights)), weights)
plt.xlabel('Expert ID')
plt.ylabel('Average Weight')
plt.title('Expert Usage Distribution (Q Projection)')
plt.savefig('expert_usage.png')
print("Saved to expert_usage.png")
```

---

## 다음 단계

### 1. 데이터셋 준비

실제 학습을 위해 video object segmentation 데이터셋을 준비하세요:

```python
# 예시: Custom dataset 구현
class VideoSegmentationDataset:
    def __getitem__(self, idx):
        return {
            'frames': torch.Tensor,      # (T, C, H, W)
            'masks': torch.Tensor,       # (T, H, W)
            'points': torch.Tensor,      # (N, 2) 또는 None
        }
```

### 2. Training Loop 커스터마이즈

`train_moe.py`의 `train_step()` 메서드를 수정하여 실제 SAM2 inference API 사용:

```python
def train_step(self, batch):
    frames = batch['frames']
    masks = batch['masks']

    # SAM2 inference 사용
    inference_state = self.model.init_state(video_path=...)
    self.model.add_new_points(inference_state, points=...)
    predictions = self.model.propagate_in_video(inference_state)

    # Loss 계산
    loss = compute_segmentation_loss(predictions, masks)
    return loss
```

### 3. Expert Specialization 분석

학습 후 어떤 expert가 어떤 domain/object에 특화되었는지 분석:

```python
# 다양한 도메인에서 expert usage 비교
domains = ['medical', 'robotics', 'autonomous_driving']
for domain in domains:
    data = load_domain_data(domain)
    usage = analyze_expert_usage(model, data)
    print(f"{domain}: {usage}")
```

---

## 참고 자료

- **전체 문서**: [SAM2_MOE_README.md](SAM2_MOE_README.md)
- **Training 스크립트**: [train_moe.py](train_moe.py)
- **간단한 예제**: [examples/simple_train_moe.py](examples/simple_train_moe.py)
- **Configuration**: [configs/training/train_moe_config.yaml](configs/training/train_moe_config.yaml)

---

## 문제 해결

### CUDA Out of Memory

```bash
# Batch size 줄이기
uv run python train_moe.py --batch_size 1 --gradient_accumulation_steps 8

# 또는 작은 모델 사용
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_t_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_tiny.pt
```

### Import Error

```bash
# 의존성 재설치
uv sync --reinstall
```

### Checkpoint Loading Error

MoE adapter keys가 없는 것은 정상입니다 (처음 학습 시):
```
MoE adapter keys not loaded (expected): 1234 keys
```

이는 base weights만 로드되고 LoRA adapters는 random initialization되었다는 의미입니다.

---

## 요약

```bash
# 1. 데모 실행
uv run python examples/sam2_moe_demo.py

# 2. 간단한 training 테스트
uv run python examples/simple_train_moe.py

# 3. 전체 training (dummy data)
uv run python train_moe.py --use_dummy_data --num_epochs 2

# 4. 실제 training (데이터 준비 후)
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_b+_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_base_plus.pt \
    --num_epochs 10
```

Happy training! 🚀
