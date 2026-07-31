import subprocess
import sys
import os

def get_gpu_vendor():
    try:
        # Use PowerShell Get-CimInstance to query the hardware GPU (wmic is deprecated in Windows 11)
        output = subprocess.check_output(
            ['powershell', '-Command', 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'], 
            text=True
        )
        output_lower = output.lower()
        
        if "nvidia" in output_lower:
            return "nvidia"
        elif "intel" in output_lower and "arc" in output_lower:
            return "intel_xpu"
        elif "amd" in output_lower or "radeon" in output_lower:
            return "amd"
        return "unknown"
    except Exception as e:
        print(f"[Warning] Could not detect hardware GPU: {e}")
        return "unknown"

def install_pytorch():
    vendor = get_gpu_vendor()
    
    # Base command
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if vendor == "nvidia":
        print("[System] NVIDIA GPU detected! Downloading CUDA-enabled Deep Learning pathways...")
        # The default PyPI index for Windows contains CUDA 11/12 support natively
        cmd.extend(["torch", "torchvision", "torchaudio"])
        
    elif vendor == "intel_xpu":
        print("[System] Intel Arc GPU detected! Downloading specialized Intel XPU pathways...")
        cmd.extend([
            "torch==2.13.0+xpu", 
            "torchvision==0.28.0+xpu", 
            "torchaudio==2.11.0+xpu", 
            "intel-extension-for-pytorch==2.1.30+xpu",
            "--index-url", "https://download.pytorch.org/whl/xpu"
        ])
        
    else:
        print(f"[System] Standard GPU detected ({vendor}). Downloading default CPU/CUDA pathways...")
        cmd.extend(["torch", "torchvision", "torchaudio"])
        
    # Execute the PIP command
    try:
        subprocess.check_call(cmd)
        print("[OK] Deep Learning architecture successfully embedded!")
    except subprocess.CalledProcessError as e:
        print(f"[!] FATAL ERROR: Failed to install PyTorch. {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_pytorch()
