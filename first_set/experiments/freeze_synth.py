"""Freeze the synthetic generator + shift-config (frozen-protocol Lock-in 5).

Before the first OOD evaluation, snapshot cwr/synth.py + the shift-config into results/frozen/ and
record content hashes. The project is not under git, so the sha256 in MANIFEST.json is the audit
anchor: any later change to the DGP changes the hash = a detectable design protocol violation.

Snapshotted: synth.py (the DGP), shift_config.json (frozen constants + ShiftSpec schema, extracted
live from the module so it cannot drift from the code), and validate_synth.py (so the "DGP does what
it claims" evidence is reproducible against exactly this generator).

Run:  python experiments/freeze_synth.py
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import scipy
import sklearn

from cwr import PROJECT_ROOT, RESULTS_DIR
from cwr import synth


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def shift_config():
    """Frozen shift-config, extracted live from cwr.synth (cannot drift from the code)."""
    return {
        "constants": {
            "MEAN_MAX": synth._MEAN_MAX,
            "REGION_MAX": synth._REGION_MAX,
            "CONCEPT_MAX": synth._CONCEPT_MAX,
            "TARGET_TILT": synth._TARGET_TILT,
            "DEFAULT_KNEE_Q": synth._DEFAULT_KNEE_Q,
        },
        "shift_kinds": list(synth.SHIFT_KINDS),
        "extrap_regimes": list(synth.EXTRAP_REGIMES),
        "covariate_modes": list(synth.COVARIATE_MODES),
        "shiftspec_defaults": {
            "kind": synth.ShiftSpec().kind,
            "severity": synth.ShiftSpec().severity,
            "covariate_mode": synth.ShiftSpec().covariate_mode,
            "extrap": synth.ShiftSpec().extrap,
            "extrap_amp": synth.ShiftSpec().extrap_amp,
        },
        "structural_frame": {
            "regime_partition_coords": [0, 1],
            "shift_extrap_axis_coord": 2,
            "observed_features": "X = S @ R (S ~ N(0,I) structural; R orthogonal, rotation-interpolated)",
        },
    }


def main():
    frozen = RESULTS_DIR / "frozen"
    os.makedirs(frozen, exist_ok=True)

    code = PROJECT_ROOT / "code"
    src_files = {
        "synth.py": code / "cwr" / "synth.py",
        "validate_synth.py": code / "experiments" / "validate_synth.py",
    }
    entries = {}
    for name, src in src_files.items():
        dst = frozen / name
        shutil.copyfile(src, dst)
        entries[name] = sha256(dst)

    cfg = shift_config()
    cfg_path = frozen / "shift_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    entries["shift_config.json"] = sha256(cfg_path)

    combined = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(entries.items())).encode()).hexdigest()

    manifest = {
        "purpose": "frozen-protocol Lock-in 5: synthetic DGP + shift-config snapshot",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "combined_sha256": combined,
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    with open(frozen / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("FROZEN (Lock-in 5) ->", frozen)
    for k, v in sorted(entries.items()):
        print(f"  {k:22s} sha256={v}")
    print(f"  {'COMBINED':22s} sha256={combined}")
    return combined


if __name__ == "__main__":
    main()
