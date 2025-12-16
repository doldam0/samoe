# SAM2-MoE Training Guide

SAM2 base_plus 체크포인트를 initial parameter로 사용하여 MoE LoRA adapter를 fine-tuning하는 가이드입니다.

## 빠른 시작

### 1. 모델 로드 및 파라미터 확인

```python
from sam2.build_sam import build_sam2_video_predictor_moe
import torch

# SAM2-MoE 모델 빌드 (base_plus 체크포인트 사용)
predictor = build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_base_plus.pt",
    device="cuda",
    mode="train",
)

# 파라미터 통계
total = sum(p.numel() for p in predictor.parameters())
trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable ratio: {100*trainable/total:.2f}%")
```

**출력 예시:**
```
Total parameters: 81,510,978
Trainable parameters: 660,800
Trainable ratio: 0.81%
```

### 2. 간단한 Training 예제 실행

```bash
# Dummy data로 테스트
uv run python examples/simple_train_moe.py
```

이 스크립트는:
- ✅ Base weights 로드 (`checkpoints/sam2.1_hiera_base_plus.pt`)
- ✅ Base model freeze, MoE adapters만 trainable
- ✅ Optimizer 설정
- ✅ Training loop 실행
- ✅ MoE adapter checkpoint 저장/로드

## Training 스크립트 사용법

### 기본 사용

```bash
# Dummy data로 테스트
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_b+_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_base_plus.pt \
    --output_dir outputs/moe_training \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --use_dummy_data
```

### 실제 데이터셋 사용

```bash
# 실제 학습 (데이터셋 준비 후)
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_b+_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_base_plus.pt \
    --output_dir outputs/my_experiment \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4
```

## 주요 하이퍼파라미터

### MoE 설정 (config 파일에서)

```yaml
# configs/sam2.1/sam2.1_hiera_b+_moe.yaml

self_attention:
  num_experts: 10      # Expert 개수
  lora_rank: 4         # LoRA rank (bottleneck dimension)
  top_k: 2             # 활성화할 expert 수
  lora_alpha: 1.0      # LoRA scaling factor
  lora_dropout: 0.1    # LoRA layer dropout
```

### Training 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `learning_rate` | 1e-4 | 학습률 |
| `weight_decay` | 0.01 | Weight decay for AdamW |
| `batch_size` | 2 | Batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `num_epochs` | 10 | 전체 epoch 수 |

**Effective batch size** = `batch_size` × `gradient_accumulation_steps`

## Checkpoint 관리

### 자동 저장

Training 중 자동으로 checkpoint가 저장됩니다:

```
outputs/moe_training/
├── checkpoint-1000.pt      # 1000 step마다
├── checkpoint-2000.pt
├── checkpoint-epoch-1.pt   # Epoch마다
├── checkpoint-epoch-2.pt
└── ...
```

### Checkpoint 구조

```python
checkpoint = {
    'epoch': 현재 epoch,
    'global_step': 현재 step,
    'model_state_dict': MoE adapter weights만,
    'optimizer_state_dict': optimizer 상태,
    'scheduler_state_dict': scheduler 상태,
}
```

### MoE Adapter만 저장/로드

```python
# 저장
moe_state_dict = {
    name: param
    for name, param in model.state_dict().items()
    if any(k in name for k in ['lora', 'gate', 'experts'])
}
torch.save({'moe_state_dict': moe_state_dict}, 'moe_adapters.pt')

# 로드
checkpoint = torch.load('moe_adapters.pt')
model.load_state_dict(checkpoint['moe_state_dict'], strict=False)
```

## Fine-tuning 전략

### 1. Parameter-Efficient Fine-tuning

- **Base model**: Frozen (81M parameters)
- **LoRA adapters**: Trainable (660K parameters, 0.81%)
- **Gating networks**: Trainable

**장점:**
- 메모리 효율적 (gradient는 0.81%만)
- 빠른 학습
- Catastrophic forgetting 완화

### 2. Learning Rate 설정

```python
# Base model에서 fine-tuning하므로 낮은 LR 사용
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,  # 기본값
    weight_decay=0.01,
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)
```

### 3. Gradient Accumulation

GPU 메모리가 부족할 때:

```bash
# Batch size 줄이고 accumulation 늘리기
uv run python train_moe.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8  # effective batch = 8
```

## Expert 분석

### Expert Usage 모니터링

```python
# Training loop에서
if step % 500 == 0:
    # MoE attention 모듈 찾기
    for name, module in model.named_modules():
        if 'MoERoPEAttention' in str(type(module).__name__):
            # Dummy input으로 expert usage 확인
            with torch.no_grad():
                stats = module.get_expert_statistics(q, k, v)

            # Expert weights 로깅
            for i, weight in enumerate(stats['q_proj']):
                print(f"Expert {i}: {weight:.4f}")
```

### Expert Specialization 확인

Training 후 각 expert가 어떤 domain/object에 특화되었는지 분석:

```python
import numpy as np

# 여러 domain에서 expert usage 수집
domains = ['medical', 'robotics', 'outdoor']
expert_usage = {domain: [] for domain in domains}

for domain in domains:
    data = load_domain_data(domain)
    for batch in data:
        usage = get_expert_usage(model, batch)
        expert_usage[domain].append(usage)

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, domain in enumerate(domains):
    avg_usage = np.mean(expert_usage[domain], axis=0)
    axes[idx].bar(range(10), avg_usage)
    axes[idx].set_title(f'{domain} - Expert Usage')
    axes[idx].set_xlabel('Expert ID')
    axes[idx].set_ylabel('Usage')
plt.tight_layout()
plt.savefig('expert_specialization.png')
```

## 데이터셋 준비

### Custom Dataset 구현

```python
from torch.utils.data import Dataset

class VideoSegmentationDataset(Dataset):
    def __init__(self, video_dir, annotation_dir):
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        # ... 초기화

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # 비디오 프레임 로드
        frames = self.load_frames(idx)  # (T, C, H, W)

        # Annotation (masks, points) 로드
        masks = self.load_masks(idx)    # (T, H, W)
        points = self.load_points(idx)   # (N, 2) or None

        return {
            'frames': frames,
            'masks': masks,
            'points': points,
        }
```

### DataLoader 설정

```python
from torch.utils.data import DataLoader

dataset = VideoSegmentationDataset(
    video_dir='data/videos',
    annotation_dir='data/annotations',
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

## Loss Function

### Segmentation Loss

```python
def compute_segmentation_loss(predictions, targets):
    # Dice Loss
    dice_loss = dice_loss_fn(predictions, targets)

    # Focal Loss
    focal_loss = focal_loss_fn(predictions, targets)

    # Total Loss
    total_loss = dice_loss + focal_loss

    return total_loss
```

### IoU Prediction Loss (optional)

```python
# SAM2는 IoU도 예측하므로
iou_loss = F.mse_loss(pred_iou, target_iou)
total_loss = mask_loss + 0.1 * iou_loss
```

## 모니터링 & 디버깅

### Training 로그

```python
# train_moe.py에서 자동으로 로깅
if step % logging_steps == 0:
    print(f"Step {step}:")
    print(f"  Loss: {loss:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"  Trainable params: {trainable_params:,}")
```

### TensorBoard (optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('outputs/tensorboard')

# Training loop에서
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('LR', lr, step)
writer.add_histogram('Expert/usage', expert_usage, step)
```

### WandB (optional)

```python
import wandb

wandb.init(project='sam2-moe', name='experiment_1')

# Training loop에서
wandb.log({
    'loss': loss,
    'lr': lr,
    'expert_usage': expert_usage,
})
```

## 문제 해결

### CUDA Out of Memory

```bash
# 옵션 1: Batch size 줄이기
--batch_size 1 --gradient_accumulation_steps 8

# 옵션 2: 작은 모델 사용
--config_file configs/sam2.1/sam2.1_hiera_s_moe.yaml \
--ckpt_path checkpoints/sam2.1_hiera_small.pt
```

### Training이 너무 느림

```bash
# Mixed precision training 활성화
--use_amp

# DataLoader workers 늘리기
--num_workers 8
```

### Expert가 균등하게 사용되지 않음

Load balancing loss 추가:

```python
# Encourage uniform expert usage
load_balance_loss = torch.var(expert_weights.mean(dim=(0, 1)))
total_loss = total_loss + 0.01 * load_balance_loss
```

## 예제 Scripts

### 1. 간단한 테스트

```bash
uv run python examples/simple_train_moe.py
```

### 2. Dummy data로 full training

```bash
uv run python train_moe.py --use_dummy_data --num_epochs 2
```

### 3. 실제 학습

```bash
uv run python train_moe.py \
    --config_file configs/sam2.1/sam2.1_hiera_b+_moe.yaml \
    --ckpt_path checkpoints/sam2.1_hiera_base_plus.pt \
    --output_dir outputs/my_experiment \
    --num_epochs 20
```

## 참고 자료

- **전체 문서**: [SAM2_MOE_README.md](SAM2_MOE_README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Training Script**: [train_moe.py](train_moe.py)
- **Simple Example**: [examples/simple_train_moe.py](examples/simple_train_moe.py)

## 요약

✅ **Base weights 로드**: `checkpoints/sam2.1_hiera_base_plus.pt`
✅ **MoE adapters만 학습**: 0.81% of parameters
✅ **Parameter-efficient**: 81M frozen, 660K trainable
✅ **Catastrophic forgetting 완화**: Expert specialization
✅ **사용 가능한 체크포인트**: tiny, small, base_plus, large

Happy training! 🚀
