"""Fast-training entry point.

Delegates entirely to train.py's main().  Replaces the runner class by
patching train's *own* module namespace — the only place that matters.

Why not patch my_on_policy_runner.MotionOnPolicyRunner?
  train.py line 68:
      from whole_body_tracking.utils.my_on_policy_runner import
          MotionOnPolicyRunner as OnPolicyRunner
  This creates a direct binding `OnPolicyRunner` in train's __dict__.
  Patching the source module after this binding is set has no effect.
  We must patch train.__dict__['OnPolicyRunner'] instead.
"""
import sys
from pathlib import Path

# Force current repo package path to highest priority, avoiding accidental
# import from another whole_body_tracking checkout in PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / "source/whole_body_tracking"
sys.path.insert(0, str(_PKG_ROOT))

import train as _train_mod
from whole_body_tracking.utils.fast_on_policy_runner import FastMotionOnPolicyRunner

# Replace the local binding in train.py's namespace.
# When main() executes `runner = OnPolicyRunner(...)`, Python looks up
# OnPolicyRunner in _train_mod.__dict__ at call time — so this works.
_train_mod.OnPolicyRunner = FastMotionOnPolicyRunner  # type: ignore

if __name__ == "__main__":
    _train_mod.main()
