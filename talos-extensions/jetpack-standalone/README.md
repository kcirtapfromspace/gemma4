# JetPack Standalone Dev Kit Setup

Flash a standalone NVIDIA dev kit with JetPack for maximum inference performance.
This dev kit runs outside the Talos cluster but on the same local network.

## Hardware

- NVIDIA Jetson Orin NX/Nano Dev Kit (any variant)
- USB-C cable for flashing
- microSD card or NVMe drive

## Flash Steps

### 1. Download NVIDIA SDK Manager

On an x86_64 Ubuntu 20.04/22.04 host:
```bash
# Download from https://developer.nvidia.com/sdk-manager
sudo dpkg -i sdkmanager_*.deb
sdkmanager --cli
```

### 2. Flash JetPack 6.1

```bash
# Connect dev kit in recovery mode (hold REC button, power on)
sdkmanager --cli install \
  --logintype devzone \
  --product Jetson \
  --target JETSON_ORIN_NX_8GB \
  --version JetPack_6.1 \
  --flash all
```

### 3. First Boot Setup

After flash, boot the dev kit and complete Ubuntu OOBE, then:

```bash
# Set 15W power mode (MAXN for even more power)
sudo nvpmodel -m 0  # MAXN (25W)
sudo jetson_clocks   # Pin all clocks to max

# Verify clocks
sudo tegrastats --interval 1000 | head -3
# Should show EMC_FREQ @3200, GR3D_FREQ @1020+
```

### 4. Install Docker + llama-server

```bash
# Docker is pre-installed on JetPack 6.x
sudo systemctl enable docker
sudo systemctl start docker

# Pull the llama-server image (or build locally)
# Option A: Use dustynv's pre-built container
sudo docker run -d \
  --name llama-server \
  --runtime nvidia \
  --gpus all \
  -p 8080:8080 \
  -v /models:/models \
  192.168.25.201:5050/llama-server:latest \
  /cuda-build/bin/llama-server \
    -m /models/gemma-4-E2B-it-Q3_K_M.gguf \
    --lora /models/cliniq-compact-lora.gguf \
    --port 8080 --host 0.0.0.0 \
    --ctx-size 1536 --n-gpu-layers 99 \
    --reasoning-budget 0 --parallel 1

# Option B: Use llama.cpp directly
# See apps/llama-server/Dockerfile.jetson for build instructions
```

### 5. Copy Models

```bash
# From the Talos cluster node (or any machine with the models)
scp user@model-host:/path/to/gemma-4-E2B-it-Q3_K_M.gguf /models/
scp user@model-host:/path/to/cliniq-compact-lora.gguf /models/
```

### 6. Network Setup

The dev kit should get a DHCP address on the local network.
Set a static IP if preferred:

```bash
sudo nmcli con mod "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses 192.168.150.50/24 \
  ipv4.gateway 192.168.150.1
sudo nmcli con up "Wired connection 1"
```

### 7. Test

```bash
# From any machine on the network
curl http://192.168.150.50:8080/health
# {"status":"ok"}

# Run the benchmark
python3 scripts/benchmark.py \
  --endpoint http://192.168.150.50:8080 \
  --experiment-name "jetpack-15w-standalone" \
  --runs 1 --warmup 0 --max-tokens 512 --no-stream \
  --system-prompt "Extract clinical entities from this eICR. Return minified JSON: {\"patient\":{...},\"conditions\":[...],\"labs\":[...],\"meds\":[...],\"vitals\":[...]}. All sections are arrays. Include SNOMED for conditions, LOINC for labs, RxNorm for meds. No summary. No markdown. JSON only."
```

## Expected Performance

With MAXN (25W) power mode + jetson_clocks:
- EMC: 3200 MHz (vs 2133 MHz on Talos 7W)
- GPU: 1020 MHz (vs 625 MHz on Talos 7W)
- Expected: **2-3 tok/s** (vs 1.4 on Talos) + compact prompt = **60-120s per extraction**
