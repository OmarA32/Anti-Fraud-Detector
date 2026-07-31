import os
import subprocess
import sys

def check_intel_gpu():
    print("[System] Scanning hardware for Intel XPU (Arc/Graphics)...")
    try:
        # Run powershell command to get VideoController name
        result = subprocess.run(
            ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, check=True
        )
        gpus = result.stdout.lower()
        if "intel" in gpus and ("arc" in gpus or "graphics" in gpus):
            print(f"[OK] Intel GPU detected: {result.stdout.strip()}")
            return True
        print("[Info] No Intel Arc/Graphics GPU detected. Will default to standard CPU.")
        return False
    except Exception as e:
        print(f"[Warning] Failed to query GPU hardware: {e}. Defaulting to CPU.")
        return False

def install_pytorch(is_xpu):
    if is_xpu:
        print("[System] Installing Intel XPU PyTorch for hardware acceleration...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/xpu"]
    else:
        print("[System] Installing standard CPU PyTorch...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    is_xpu = check_intel_gpu()
    try:
        install_pytorch(is_xpu)
        print("[OK] PyTorch installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install PyTorch: {e}")
        sys.exit(1)
