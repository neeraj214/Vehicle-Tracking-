import torch
import subprocess
import sys
import os
import time
import platform

def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_system():
    print_section("SYSTEM INFO")
    print(f"  OS            : {platform.system()} {platform.release()}")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA (torch)  : {torch.version.cuda}")
    print(f"  cuDNN         : {torch.backends.cudnn.version()}")


def check_gpu():
    print_section("GPU DETECTION")

    if not torch.cuda.is_available():
        print("  [ERROR] CUDA NOT available")
        print("\n  Fix options:")
        print("  1. Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision --index-url \\")
        print("     https://download.pytorch.org/whl/cu118")
        print("\n  2. Update NVIDIA drivers:")
        print("     https://www.nvidia.com/Download/index.aspx")
        print("\n  3. Check nvidia-smi in terminal:")
        print("     nvidia-smi")
        return False

    n = torch.cuda.device_count()
    print(f"  [OK] CUDA available")
    print(f"  GPU count     : {n}")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = round(props.total_memory / 1024**3, 2)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Total VRAM   : {mem_gb} GB")
        print(f"    CUDA cores   : {props.multi_processor_count * 128}")
        print(f"    Compute cap  : {props.major}.{props.minor}")
        print(f"    Clock speed  : {props.clock_rate / 1000:.0f} MHz")

    return True


def check_memory():
    print_section("MEMORY STATUS")

    if not torch.cuda.is_available():
        print("  [Skip] No GPU detected")
        return

    torch.cuda.empty_cache()
    total  = torch.cuda.get_device_properties(0).total_memory
    reserv = torch.cuda.memory_reserved(0)
    alloc  = torch.cuda.memory_allocated(0)
    free   = total - reserv

    print(f"  Total VRAM    : {total  / 1024**3:.2f} GB")
    print(f"  Reserved      : {reserv / 1024**3:.2f} GB")
    print(f"  Allocated     : {alloc  / 1024**3:.2f} GB")
    print(f"  Free          : {free   / 1024**3:.2f} GB")

    # Recommended batch sizes for RTX 2050 (4GB VRAM)
    print(f"\n  Recommended settings for your GPU:")
    if total < 4.5 * 1024**3:
        print(f"  batch_size    : 2  (4GB VRAM)")
        print(f"  Use mixed precision (FP16) — saves ~50% VRAM")
    elif total < 6 * 1024**3:
        print(f"  batch_size    : 4  (6GB VRAM)")
    elif total < 8 * 1024**3:
        print(f"  batch_size    : 8  (8GB VRAM)")
    else:
        print(f"  batch_size    : 16 (8GB+ VRAM)")


def run_benchmark():
    print_section("GPU SPEED BENCHMARK")

    if not torch.cuda.is_available():
        print("  [Skip] No GPU detected")
        return

    device = torch.device("cuda")
    sizes  = [512, 1024, 2048]

    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.time()
        for _ in range(10):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / 10 * 1000

        print(f"  MatMul {size}x{size}  : {elapsed:.2f} ms")

    # Test forward pass speed (simulates Re-ID head)
    print(f"\n  Re-ID Head Forward Pass Simulation:")
    from torch import nn
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)

    x = torch.randn(100, 256, device=device)

    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 100 * 1000

    print(f"  100 detections forward : {elapsed:.2f} ms")
    print(f"  Estimated Re-ID FPS    : {1000/elapsed:.0f} frames/sec")


def check_mixed_precision():
    print_section("MIXED PRECISION (FP16) SUPPORT")

    if not torch.cuda.is_available():
        print("  [Skip] No GPU detected")
        return

    try:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        device = torch.device("cuda")
        model  = torch.nn.Linear(256, 128).to(device)
        x      = torch.randn(32, 256, device=device)

        with autocast():
            out = model(x)

        print(f"  [OK] Mixed precision (FP16) supported")
        print(f"  autocast dtype : {out.dtype}")
        print(f"  GradScaler     : ready")
        print(f"\n  Benefit: ~50% VRAM reduction on RTX 2050")
        print(f"  Enable by passing: use_amp=True to train.py")

    except Exception as e:
        print(f"  [WARN] Mixed precision error: {e}")


def check_nvidia_smi():
    print_section("NVIDIA-SMI OUTPUT")
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            headers = [
                "GPU Name", "Driver", "VRAM Total",
                "VRAM Free", "Temp", "Utilization"
            ]
            for line in lines:
                vals = [v.strip() for v in line.split(",")]
                for h, v in zip(headers, vals):
                    print(f"  {h:<15}: {v}")
        else:
            print("  [WARN] nvidia-smi not found or failed")
            print("     Install NVIDIA drivers from:")
            print("     https://www.nvidia.com/Download/index.aspx")
    except FileNotFoundError:
        print("  [ERROR] nvidia-smi not found")
        print("     Make sure NVIDIA drivers are installed")
    except Exception as e:
        print(f"  [ERROR] Error: {e}")


def print_training_recommendations():
    print_section("TRAINING RECOMMENDATIONS FOR YOUR GPU")

    if not torch.cuda.is_available():
        print("  No GPU — will train on CPU (very slow)")
        print("  Consider using Google Colab free GPU instead")
        return

    total = torch.cuda.get_device_properties(0).total_memory
    name  = torch.cuda.get_device_name(0)

    print(f"  GPU: {name}")
    print(f"\n  Optimal train.py settings:")

    if total < 4.5 * 1024**3:
        # RTX 2050 — 4GB
        print(f"  python training/train.py \\")
        print(f"    --epochs 20 \\")
        print(f"    --batch  2  \\")
        print(f"    --lr     1e-4")
        print(f"\n  With mixed precision (recommended):")
        print(f"  python training/train.py \\")
        print(f"    --epochs 20 \\")
        print(f"    --batch  4  \\")
        print(f"    --lr     1e-4 \\")
        print(f"    --amp")
        print(f"\n  Expected training time: ~2-3 hours for 20 epochs")
    elif total < 8 * 1024**3:
        print(f"  python training/train.py \\")
        print(f"    --epochs 20 \\")
        print(f"    --batch  8  \\")
        print(f"    --lr     1e-4")
        print(f"\n  Expected training time: ~1-2 hours for 20 epochs")
    else:
        print(f"  python training/train.py \\")
        print(f"    --epochs 20 \\")
        print(f"    --batch  16 \\")
        print(f"    --lr     1e-4")
        print(f"\n  Expected training time: ~30-60 mins for 20 epochs")

    print(f"\n  Monitor GPU during training:")
    print(f"  watch -n 1 nvidia-smi")


if __name__ == "__main__":
    print("\n🔍 GPU DIAGNOSTIC TOOL — Vehicle Tracker Project")
    check_system()
    gpu_ok = check_gpu()
    check_nvidia_smi()
    check_memory()
    run_benchmark()
    check_mixed_precision()
    print_training_recommendations()
    print(f"\n{'='*55}")
    if gpu_ok:
        print("  [OK] GPU is ready for training!")
        print("  Next: python scripts/download_dataset.py")
    else:
        print("  [ERROR] Fix GPU issues above before training")
        print("  Alternative: Use Google Colab (free GPU)")
    print(f"{'='*55}\n")
