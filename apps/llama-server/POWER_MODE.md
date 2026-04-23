# Jetson Orin NX Power Mode on Talos

Team C1 investigation, 2026-04-23. Branch `team/c1-power-mode-2026-04-23`.

## TL;DR

The Jetson cannot be put into 15W (`nvpmodel -m 2`) mode **at runtime from any pod on Talos Linux**. The EMC memory controller is stuck at 2133 MHz and the GPU at ~625 MHz because:

1. The NVIDIA `nvpmodel` kernel driver is **not present** in the Talos kernel (no alias for `nvidia,nvpmodel` compatible; no module file; `module.sig_enforce=1` also blocks loading out-of-tree `.ko`).
2. The `nvpmodel` userspace binary **detects container/chroot and refuses to issue any sysfs writes** ("Running in chroot, ignoring request").
3. Even with a kernel driver, the **BPMP firmware's EMC timing tables only contain 3 rates** (204 / 665.6 / 2133 MHz). Higher-rate tables (2750 / 3199 / 3200 MHz) are not loaded by the current boot chain.
4. Direct writes to `/sys/kernel/debug/bpmp/debug/clk/{emc,gpc0clk}/{rate,max_rate}` are silently ignored by the BPMP — `mrq_rate_locked` toggling doesn't help.

**Only a custom Talos image with the `jetson-nvpmodel` extension (already scaffolded in `talos-extensions/jetson-nvpmodel/`) can unblock this.** Expect ~1.6x tok/s improvement if achieved, but this was NOT measured here because we could not change the state.

## Current state (measured 2026-04-23 from `power-probe` privileged pod, node `talos-jetson-3` / `192.168.150.41`)

| Resource | Current (locked) | p3767_0001 mode 2 (15W) target | MAXN target |
|---|---|---|---|
| `cpu0/cpufreq/cpuinfo_cur_freq` | **1510400 kHz** (1.51 GHz) | 1420800 | unlimited |
| `cpu0/cpufreq/scaling_max_freq` | 1510400 | 1420800 | -1 |
| `/sys/kernel/debug/bpmp/debug/clk/emc/rate` | **2133000000** (2.133 GHz) | 3200000000 | 3200000000 |
| `/sys/kernel/debug/bpmp/debug/clk/emc/max_rate` | **2133000000** | 3200000000 | -1 |
| `/sys/kernel/debug/bpmp/debug/emc/possible_rates` | `204000 665600 2133000 (kHz)` | also 2750 / 3199 / 3200 | same |
| `/sys/kernel/debug/bpmp/debug/clk/gpc0clk/rate` | **624750000** (624.75 MHz) | up to 612 MHz (cap) | 918 MHz |
| CPUs online | 0/1/2/3/4/5 (all 6) | 0/1/2/3 (four) | all 6 |
| `scaling_governor` (policy0, policy4) | `performance` | performance | performance |
| DRAM total | 7.4 GiB | — | — |
| DTB model | `NVIDIA Jetson Orin NX Engineering Reference Developer Kit` | — | — |
| CPU scaling max (cap enforced by BPMP firmware) | 1510400 | 1510400 | 1510400 |

Note: current CPU freq and CPUs-online actually match **MAXN**, not 7W. EMC and GPU are the real bottleneck — not the full 7W profile as described in the mission brief. Eval throughput at this mixed state (from active llama-server production traffic): **0.87 tok/s eval, 1.03 tok/s prompt**.

## What I tried

### (1) Fast path — redeploy the DaemonSet

Deployed `apps/llama-server/nvpmodel-daemonset.yaml`. Registry pull failed (`192.168.25.201:5050: connection refused`). Patched to `imagePullPolicy: IfNotPresent` to use cached image. Then:

```text
=== Jetson Power Mode Bootstrap ===
EMC: 2133000000 (max: 2133000000)
GPU: 624750000 (max: 624750000)
Installing nvpmodel...
... apt install OK from repo.download.nvidia.com ...
nvpmodel installed: /usr/sbin/nvpmodel
Setting power mode 2 (15W)...
Running in chroot, ignoring request.
NVPM WARN: Golden image context is already created
Automatically rebooting to reflect the mode change
NVPM WARN: rebooting..
Post-reboot run. Re-applying...
Running in chroot, ignoring request.   <-- same message
... EMC / GPU unchanged ...
```

Two specific failure walls, in order:

**(a) Registry unreachable** — `192.168.25.201:5050` returns connection refused. Workaround: add `imagePullPolicy: IfNotPresent`. Not a deal-breaker.

**(b) nvpmodel refuses sysfs writes in a container.** Strace of `nvpmodel -f /etc/nvpmodel/nvpmodel_p3767_0001.conf -m 2 --force` from inside the pod (verified via `strace -f -e trace=openat,write`):

- Opens only `/var/lib/nvpmodel/status` (writes `pmode:0002`) and `/var/lib/nvpmodel/conf_file_path`.
- **Does NOT open or write any of these expected sysfs paths**: `scaling_max_freq`, `scaling_min_freq`, `cpu4/online`, `cpu5/online`, `17000000.gpu/devfreq_dev/max_freq`, `nvpmodel_clk_cap/emc`, `nvpmodel_emc_cap/emc_iso_cap`, `fbp_pg_mask`, `tpc_pg_mask`, or `gpu.0/power/control`.
- Prints `"Running in chroot, ignoring request"` and exits 0.

`nvpmodel` detects the non-default mount namespace (pod mnt_ns = 4026533121 vs host PID 1 mnt_ns = 4026531832) and bails out before any hardware programming.

**(c) nsenter into host PID 1 — Talos exports nothing.** `nsenter --target 1 --mount --pid -- /bin/sh` fails with `No such file or directory`. Talos host filesystem has no `/bin/sh`, no `/usr/bin/ls`, no `/usr/sbin/nvpmodel`. Only `/usr/bin/init`. So even with hostPID and CAP_SYS_ADMIN, there is nothing to exec.

**(d) No `reboot` binary on host**, so the DaemonSet's `nsenter --target 1 -- reboot` line is non-functional.

### (2) Direct BPMP clock writes

Mounted debugfs (`mount -t debugfs none /sys/kernel/debug`), inspected all BPMP MRQ interfaces:

```bash
# Writable by a root-privileged pod? YES — open(2) succeeds.
# Sticks? NO — BPMP silently drops the request.

$ echo 3200000000 > /sys/kernel/debug/bpmp/debug/clk/emc/max_rate
$ cat /sys/kernel/debug/bpmp/debug/clk/emc/max_rate
2133000000   # unchanged

$ echo 3200000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
$ cat /sys/kernel/debug/bpmp/debug/clk/emc/rate
2133000000   # unchanged

$ echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked   # force lock
$ echo 3200000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
$ cat .../emc/rate
2133000000   # still unchanged, even with lock

$ cat /sys/kernel/debug/bpmp/debug/emc/possible_rates
204000 665600 2133000 (kHz)   # only 3 rates loaded
```

Same behavior for `gpc0clk` and `nafll_gpc0`. BPMP firmware rejects any rate that is not in its internal possible_rates table, and the 3200-MHz EMC / 918+ MHz GPU tables are not loaded.

Additionally, the nvpmodel kernel-side sysfs is entirely absent:

- `/sys/kernel/nvpmodel_clk_cap/emc` — no such file (driver not loaded)
- `/sys/kernel/nvpmodel_emc_cap/emc_iso_cap` — no such file
- `/sys/devices/platform/17000000.gpu/devfreq_dev/{min_freq,max_freq}` — no such file
- `/sys/devices/platform/nvpmodel/driver` → NULL (no driver bound to the devicetree node)
- `/sys/devices/platform/nvpmodel/modalias` = `of:NnvpmodelT(null)Cnvidia,nvpmodel`. **No module in `/usr/lib/modules/6.18.18-talos/modules.alias` matches `nvidia,nvpmodel`.**

### (3) Talos machineconfig (`machine.sysfs`) approach

Backed up via `talosctl -n 192.168.150.41 get machineconfig -o yaml > /tmp/machineconfig-backup-*.yaml`. The current machineconfig **already uses `machine.sysfs`** to set:

- `devices.system.cpu.cpufreq.policy0.scaling_governor: performance` (works)
- `devices.system.cpu.cpufreq.policy4.scaling_governor: performance` (works)
- `devices.platform.bus@0.17000000.gpu.{aelpg,blcg,elcg,elpg,slcg}_enable: "0"` — **these paths are WRONG** (real paths are `/sys/devices/platform/17000000.gpu/` without `bus@0` prefix). Talos logs `KernelParamSpecController` errors for these every minute in dmesg.

None of these buys us anything. The useful sysfs paths for lifting EMC/GPU caps are backed by the missing nvpmodel kernel driver — you can't write to paths that don't exist. `scaling_max_freq` is already at its CPU DVFS-imposed ceiling (1510400 > 15W target 1420800), so lowering it would slow us down.

I did NOT modify the machineconfig. Sysfs approach provides zero benefit without nvpmodel.ko.

### (4) Build the extension — scoping only (no rebuild)

The `jetson-nvpmodel` scaffold in `talos-extensions/jetson-nvpmodel/` downloads the stock L4T `nvidia-l4t-nvpmodel` and `nvidia-l4t-init` debs and ships them as a Talos system container. **This is insufficient on its own** — the kernel driver is also required. Steps to actually ship a working image:

1. **Kernel rebuild with NVIDIA L4T kernel-oot drivers.** Talos upstream kernel 6.18 lacks:
   - `drivers/platform/tegra/tegra23x-nvpmodel.c` (the driver creating `/sys/kernel/nvpmodel_*`)
   - `drivers/platform/tegra/bwmgr/*` (bandwidth manager)
   - Full `drivers/devfreq/tegra-nvgpu-devfreq.c` and `17000000.gpu/devfreq_dev/*`
   - Requires pulling the `kernel-oot` tree from [NVIDIA's L4T R36.4.7 sources](https://developer.nvidia.com/embedded/jetson-linux-r3647) and building against Talos's kernel config.
2. **Replace the DTB and BCT** so that boot loads the higher-rate EMC timing tables (2750/3199/3200 MHz). The current `p3767-0003` DTB only loads the 2133-MHz tables. Requires sourcing the 8GB-specific DTB + BCT from L4T BSP (`bootloader/t234ref/tegra234-p3767-0001-sdcard.dtb` and the matching `tegra234-bpmp-3767-0001.dtb`).
3. **Package as Talos extension** — the existing `pkg.yaml` just copies the userspace binary; it needs to also bundle the `.ko` files plus the DTB replacement, and register the system service to run `nvpmodel -f ... -m 2 --force` during `boot` phase (before kubelet).
4. **Rebuild Talos installer** via `talosctl gen config` + `imager` with the extension layer, then re-image all three Jetsons.
5. **Verify boot** — nsenter check for clock tables, confirm `/sys/kernel/nvpmodel_clk_cap/emc` exists.

**Estimated effort: 1-2 days** for someone familiar with the Talos build and NVIDIA L4T kernel source; longer if the out-of-tree drivers need porting to kernel 6.18 (stock L4T targets 5.15). **This is NOT a 3-hour task.**

Alternative: switch the Jetson node to NVIDIA's JetPack Ubuntu (abandon Talos for this node), which ships nvpmodel working out of the box.

## Reproducible commands to verify current state

From any machine with `talosctl` + `kubectl` configured for this cluster:

```bash
# Backup Talos machineconfig
talosctl -n 192.168.150.41 get machineconfig -o yaml > /tmp/machineconfig-backup.yaml

# Launch privileged probe pod
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: power-probe
  namespace: gemma4
spec:
  nodeSelector: {kubernetes.io/hostname: talos-jetson-3}
  hostPID: true
  restartPolicy: Never
  containers:
  - name: probe
    image: 192.168.25.201:5050/llama-server:latest
    imagePullPolicy: IfNotPresent
    command: ["sleep","3600"]
    securityContext: {privileged: true}
EOF

# Inspect current clocks
kubectl -n gemma4 exec power-probe -- bash -c '
  mount -t debugfs none /sys/kernel/debug 2>/dev/null
  echo "EMC: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/rate) / max $(cat /sys/kernel/debug/bpmp/debug/clk/emc/max_rate)"
  echo "GPU: $(cat /sys/kernel/debug/bpmp/debug/clk/gpc0clk/rate) / max $(cat /sys/kernel/debug/bpmp/debug/clk/gpc0clk/max_rate)"
  echo "EMC rates: $(cat /sys/kernel/debug/bpmp/debug/emc/possible_rates)"
  echo "CPU max: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)"
  echo "nvpmodel driver: $(readlink /sys/devices/platform/nvpmodel/driver || echo NONE)"
'

# Expected output today:
#   EMC: 2133000000 / max 2133000000
#   GPU: 624750000 / max 624750000
#   EMC rates: 204000 665600 2133000 (kHz)
#   CPU max: 1510400
#   nvpmodel driver: NONE
```

## Definitive proof an image rebuild is required

1. `/usr/lib/modules/6.18.18-talos/modules.alias` contains **zero** aliases matching `nvpmodel` or `nvidia,nvp`. The kernel driver is not distributed with Talos.
2. `/sys/devices/platform/nvpmodel/driver` is a NULL symlink — the platform device exists but no driver matches the `nvidia,nvpmodel` compatible string.
3. `/sys/kernel/nvpmodel_clk_cap/` and `/sys/kernel/nvpmodel_emc_cap/` do not exist — these are created by the absent nvpmodel driver's `__init` routine.
4. Direct BPMP debugfs writes to `emc/rate` and `gpu/rate` are silently ignored; `possible_rates` contains only 204 / 665.6 / 2133 MHz, confirming the 3200-MHz EMC timing tables are not loaded into BPMP firmware at boot — **a BCT/DTB-level issue, not kernel-driver-level**, and therefore untouchable from any pod.
5. nvpmodel userspace binary detects mount-namespace boundary and bails with "Running in chroot, ignoring request" before issuing any write (strace confirmed 0 sysfs writes beyond `/var/lib/nvpmodel/status`).
6. Talos host filesystem exports no userspace beyond `/usr/bin/init`, so nsenter-based workarounds cannot exec `nvpmodel`, `reboot`, or any helper.

## tok/s delta vs 7W

**Not measured — state could not be changed.** Current measured throughput (from llama-server production traffic at this "locked 7W-EMC" state): **0.87 tok/s eval, 1.03 tok/s prompt**. Consistent with the 0.9 tok/s reported in the mission brief.

The hypothesized ~1.6x improvement (back to 1.4 tok/s) is blocked on image rebuild and was not attainable in the 3-hour window.

## Recommendations

1. **Short term (hours)**: accept 0.87 tok/s. Any win from power-mode 2 requires a multi-day kernel/image build effort.
2. **Medium term (days)**: build a custom Talos image with the `jetson-nvpmodel` extension AND the L4T `kernel-oot` drivers bundled (see scoping above). Alternatively switch `talos-jetson-3` (only) to NVIDIA JetPack Ubuntu for inference workloads — accept the GitOps/Talos-consistency trade-off.
3. **Quick-win unrelated to power mode**: the Talos machineconfig has 5 broken `sysfs` entries for GPU clock-gating disables (wrong path). Fixing those paths to `devices.platform.17000000.gpu.*` (no `bus@0`) eliminates log spam and MAY give a small boost from disabled gating on active GPU workloads. Low risk. Worth doing in a separate PR.

## Files touched

- This doc: `apps/llama-server/POWER_MODE.md` (new).
- No changes to `apps/llama-server/nvpmodel-daemonset.yaml`, `apps/llama-server/deployment.yaml`, or any file outside my zone.
- No Talos machineconfig changes applied. Backup at `/tmp/machineconfig-backup-1776984030.yaml` on the laptop.
- No reboots of `talos-jetson-3`. llama-server and mlc-test pods untouched.
