# jetson-nvpmodel Talos Extension

Sets Jetson Orin power mode at boot via nvpmodel. Without this, Talos defaults
to the 7W power mode which severely limits memory bandwidth and GPU clock.

## What it does

- Runs nvpmodel as a Talos system container (before kubelet)
- Sets power mode 2 (15W) for Orin NX 8GB (p3767_0001)
- Unlocks EMC from 2133 MHz → 3200 MHz (+50% memory bandwidth)
- Unlocks GPU from 625 MHz → 1020 MHz (+63% compute)

## Board identification

The Turing Pi 2 Jetson slot identifies as:
- **Board**: p3767_0001 (Orin NX 8GB)
- **SOC**: tegra234 (T234)
- **L4T**: R36.4.7

## Building

This extension follows the standard Talos extensions pattern from
`talos-jetson-build/extensions/`. It downloads the nvpmodel binary
and config from NVIDIA's apt repos for the matching L4T version.

```bash
# From the talos-jetson-build directory
make jetson-nvpmodel
```

## Alternative: manual approach

If you can't build the extension, you can manually set the power mode
from any privileged container on the Jetson:

```bash
apt-get update && apt-get install -y nvidia-l4t-nvpmodel nvidia-l4t-init
nvpmodel -f /etc/nvpmodel/nvpmodel_p3767_0001.conf -m 2 --force
# Then reboot the node via talosctl
```

Note: the power mode change requires a reboot to take effect.

## Power modes (Orin NX 8GB / p3767_0001)

| Mode | Name | CPUs | GPU Max | EMC Max | Power |
|------|------|------|---------|---------|-------|
| 0    | MAXN | 6    | unlimited | unlimited | 25W |
| 1    | 10W  | 4    | limited | limited | 10W |
| 2    | 15W  | 6    | limited | 3200 MHz | 15W |
