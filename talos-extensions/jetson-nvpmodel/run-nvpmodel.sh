#!/bin/bash
# Jetson nvpmodel boot script for Talos
# Sets power mode 2 (15W) for Orin NX 8GB at early boot.

CONF="/etc/nvpmodel/nvpmodel_p3767_0001.conf"
STATE_DIR="/var/lib/nvpmodel"
MODE=2

export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

mkdir -p "$STATE_DIR"

echo "jetson-nvpmodel: setting power mode $MODE (15W)"

# Apply power mode. On first boot this may request reboot;
# on subsequent boots it applies directly since the GPU hasn't
# been re-initialized with different power gating settings.
/usr/sbin/nvpmodel -f "$CONF" -m "$MODE" -u "$STATE_DIR/status" --force 2>&1 || true

echo "jetson-nvpmodel: mode set, entering sleep"

# Stay alive so Talos doesn't restart the container in a loop
exec sleep infinity
