"""Per-gate and joint pass probability vs roster size (read-only probe; thresholds from ahcl.experiment.CONFIG).

Usage: python3 scripts/synthetic_joint_power.py [SEEDS=48]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import synthetic_gate_sweep as sw  # noqa: E402

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(max(0, c-h), 3), round(min(1, c+h), 3))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
noise = 1.0
for fc in (12, 16, 20, 24, 32, 48, 64):
    per = {g: 0 for g in sw.GATE_NAMES}; joint = 0
    for n in range(N):
        am, sh, _ = sw._run(f"seed-{n}", noise, fc)
        m = sw._min_margin_per_gate(am, sh)
        for g, v in m.items(): per[g] += int(v >= 0)
        joint += int(min(m.values()) >= 0)
    print(json.dumps({"families": fc, "seeds": N, "joint_pass": joint/N, "joint_wilson95": wilson(joint, N),
                      "per_gate_pass": {g: round(c/N, 3) for g, c in per.items()}}), flush=True)
