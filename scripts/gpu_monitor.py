import time
import subprocess
import sys
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_nvidia_smi_stats():
    """Get live GPU stats from nvidia-smi"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                         "temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return None

        vals = [v.strip() for v in result.stdout.strip().split(",")]
        if len(vals) < 7:
            return None

        return {
            "name":       vals[0],
            "util":       vals[1],
            "mem_used":   vals[2],
            "mem_total":  vals[3],
            "temp":       vals[4],
            "power":      vals[5],
            "clock":      vals[6],
        }
    except Exception:
        return None


def draw_bar(value, total, width=25, char="█"):
    """Draw an ASCII progress bar"""
    pct   = min(float(value) / max(float(total), 1), 1.0)
    filled = int(pct * width)
    bar   = char * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:.1f}%"


def get_torch_memory():
    """Get PyTorch allocated memory"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    return {
        "allocated": torch.cuda.memory_allocated(0) / 1024**2,
        "reserved":  torch.cuda.memory_reserved(0)  / 1024**2,
        "peak":      torch.cuda.max_memory_allocated(0) / 1024**2,
    }


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def monitor(interval=1.0):
    """
    Real-time GPU monitor.
    Run this in a second terminal while training.
    Press Ctrl+C to stop.
    """
    print("Starting GPU monitor... Press Ctrl+C to stop\n")
    time.sleep(1)

    start_time = time.time()

    try:
        while True:
            clear_screen()
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)

            print(f"┌{'─'*53}┐")
            print(f"│  🚗 Vehicle Tracker — GPU Monitor"
                  f"{'':>18}│")
            print(f"│  Runtime: {h:02d}:{m:02d}:{s:02d}"
                  f"{'':>38}│")
            print(f"├{'─'*53}┤")

            stats = get_nvidia_smi_stats()

            if stats:
                mem_used  = float(stats['mem_used'])
                mem_total = float(stats['mem_total'])
                util      = float(stats['util'])
                temp      = float(stats['temp'])

                mem_bar  = draw_bar(mem_used, mem_total)
                util_bar = draw_bar(util, 100)

                # Temp color indicator
                if temp < 70:
                    temp_icon = "🟢"
                elif temp < 85:
                    temp_icon = "🟡"
                else:
                    temp_icon = "🔴"

                print(f"│  GPU   : {stats['name'][:40]:<40}  │")
                print(f"├{'─'*53}┤")
                print(f"│  VRAM  : {mem_bar}  "
                      f"{mem_used:.0f}/{mem_total:.0f} MB  │")
                print(f"│  Usage : {util_bar}  "
                      f"{util:.0f}%{'':>12}│")
                print(f"├{'─'*53}┤")
                print(f"│  Temp  : {temp_icon} {temp:.0f}°C"
                      f"{'':>40}│")
                print(f"│  Power : ⚡ {stats['power']} W"
                      f"{'':>38}│")
                print(f"│  Clock : 🕐 {stats['clock']} MHz"
                      f"{'':>37}│")
                print(f"├{'─'*53}┤")

                # Warnings
                if temp >= 85:
                    print(f"│  [WARN]  HIGH TEMP — check laptop cooling!"
                          f"{'':>15}│")
                if mem_used / mem_total > 0.95:
                    print(f"│  [WARN]  VRAM NEAR FULL — reduce batch size!"
                          f"{'':>13}│")

            else:
                print(f"│  [ERROR] nvidia-smi not available"
                      f"{'':>24}│")

            # PyTorch memory
            tmem = get_torch_memory()
            if tmem:
                print(f"│  PyTorch allocated : {tmem['allocated']:.0f} MB"
                      f"{'':>27}│")
                print(f"│  PyTorch peak      : {tmem['peak']:.0f} MB"
                      f"{'':>27}│")
                print(f"├{'─'*53}┤")

            print(f"│  Refreshing every {interval}s — Ctrl+C to stop"
                  f"{'':>9}│")
            print(f"└{'─'*53}┘")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n[Monitor stopped]")


if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    monitor(interval)
