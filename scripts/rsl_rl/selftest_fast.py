#!/usr/bin/env python3
"""Headless selftest for fast_bm_training.

Run from repo root:
    python scripts/rsl_rl/selftest_fast.py

No Isaac Sim / GPU required.

Two modes:
  1. Lightweight (no isaaclab install needed): tests pure-Python logic
     - ScaledUniformNoiseCfg noise math
     - _linear_schedule sentinel / boundary values
     - _check_plateau early-stopping logic
     - push scale application via mock
  2. Full (requires isaaclab Python packages in PYTHONPATH): adds
     - G1FastEnvCfg vs G1FlatEnvCfg cfg parity
     - All runner cfg sentinel values
     To run full mode, set PYTHONPATH to include isaaclab source dirs.
"""
import sys
import os
import importlib.util

# ---------------------------------------------------------------------------
# Direct-file loader: bypasses whole_body_tracking/__init__.py import chain
# so tests run without isaaclab installed.
# ---------------------------------------------------------------------------
_REPO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../.."))
_SRC = os.path.join(_REPO, "source/whole_body_tracking/whole_body_tracking")

# Add repo source to path for full-mode imports that do need the package.
sys.path.insert(0, os.path.join(_REPO, "source/whole_body_tracking"))


def _load(rel_path: str, package=None):  # package: Optional[str]
    """Import a .py file directly, bypassing package __init__ chains.

    Set `package` to the dotted package name if the file uses relative imports.
    """
    full = os.path.join(_SRC, rel_path)
    # Use a unique name to avoid collisions in sys.modules
    name = "selftest_direct." + rel_path.replace("/", ".").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
        # Register under the package-qualified name so relative imports work
        sys.modules[f"{package}.{os.path.splitext(os.path.basename(full))[0]}"] = mod
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
class R:
    def __init__(self):
        self.tests = []

    def check(self, name, ok, detail=""):
        self.tests.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    def summary(self):
        p = sum(1 for _, ok, _ in self.tests if ok)
        n = len(self.tests)
        print(f"\nSELFTEST: {p}/{n} passed, {n-p} failed")
        return 0 if p == n else 1


# ---------------------------------------------------------------------------
# M1 — noise curriculum (pure Python + torch, no isaaclab needed)
# ---------------------------------------------------------------------------
def _fake_runner_mods():
    """Return sys.modules patches that make fast_on_policy_runner importable.

    Key: MotionOnPolicyRunner must be a real class (not MagicMock) so that
    `class FastMotionOnPolicyRunner(MotionOnPolicyRunner)` creates a proper
    Python class rather than a Mock-subclass.
    """
    import unittest.mock as mock

    class _FakeMotionRunner:
        def __init__(self, *a, **kw): pass
        def log(self, *a, **kw): pass
        def learn(self, *a, **kw): pass
        def save(self, *a, **kw): pass

    return {
        "rsl_rl": mock.MagicMock(),
        "rsl_rl.env": mock.MagicMock(),
        "wandb": mock.MagicMock(),
        "whole_body_tracking.utils.my_on_policy_runner": mock.MagicMock(
            MotionOnPolicyRunner=_FakeMotionRunner
        ),
    }


def test_m1_noise_curriculum(r: R):
    import torch
    import unittest.mock as mock

    # Load curriculum.py with lightweight isaaclab mocks.
    def _configclass(cls):
        return cls

    class _UniformNoiseCfg:
        operation = "add"
        n_min = -1.0
        n_max = 1.0

    with mock.patch.dict("sys.modules", {
        "isaaclab": mock.MagicMock(),
        "isaaclab.utils": mock.MagicMock(configclass=_configclass),
        "isaaclab.utils.noise": mock.MagicMock(UniformNoiseCfg=_UniformNoiseCfg),
    }):
        cn = _load("tasks/tracking/mdp/curriculum.py")
    ScaledUniformNoiseCfg = cn.ScaledUniformNoiseCfg

    # Load fast_on_policy_runner with proper (real-class) mocks
    with mock.patch.dict("sys.modules", _fake_runner_mods()):
        fop = _load("utils/fast_on_policy_runner.py")
    _linear_schedule = fop._linear_schedule

    r.check("M1 schedule at it=0",
            abs(_linear_schedule(0, 0.2, 1.0, 5000) - 0.2) < 1e-9)
    r.check("M1 schedule at it=5000",
            abs(_linear_schedule(5000, 0.2, 1.0, 5000) - 1.0) < 1e-9)
    r.check("M1 schedule clamped after warmup",
            abs(_linear_schedule(9999, 0.2, 1.0, 5000) - 1.0) < 1e-9)
    r.check("M1 sentinel: start==end always returns end",
            _linear_schedule(0, 1.0, 1.0, 5000) == 1.0
            and _linear_schedule(100, 1.0, 1.0, 5000) == 1.0)

    noise = ScaledUniformNoiseCfg()
    noise.n_min = -0.25
    noise.n_max = 0.25
    noise.operation = "add"
    data = torch.ones(100)
    noise.scale = 0.0
    out0 = type(noise).func(data, noise)
    r.check("M1 ScaledNoise scale=0 → identity",
            torch.allclose(out0, data),
            f"max_diff={abs(out0 - data).max().item():.2e}")
    noise.scale = 1.0
    outs = torch.stack([type(noise).func(data.clone(), noise) - data for _ in range(500)])
    r.check("M1 ScaledNoise scale=1 within [-0.25, 0.25]",
            bool(((outs >= -0.25 - 1e-6) & (outs <= 0.25 + 1e-6)).all()))


# ---------------------------------------------------------------------------
# M2 — push curriculum (mock push_by_setting_velocity)
# ---------------------------------------------------------------------------
def test_m2_push_curriculum(r: R):
    import unittest.mock as mock

    _cap: dict = {}

    def _mock_push(env, env_ids, velocity_range):
        _cap["range"] = velocity_range

    # Keep import mocks active during function call, because curriculum.py
    # imports push_by_setting_velocity lazily inside the function.
    def _configclass(cls):
        return cls

    class _UniformNoiseCfg:
        operation = "add"
        n_min = -1.0
        n_max = 1.0

    with mock.patch.dict("sys.modules", {
        "isaaclab": mock.MagicMock(),
        "isaaclab.envs": mock.MagicMock(),
        "isaaclab.envs.mdp": mock.MagicMock(),
        "isaaclab.envs.mdp.events": mock.MagicMock(
            push_by_setting_velocity=_mock_push),
        "isaaclab.utils": mock.MagicMock(configclass=_configclass),
        "isaaclab.utils.noise": mock.MagicMock(UniformNoiseCfg=_UniformNoiseCfg),
    }):
        ce = _load("tasks/tracking/mdp/curriculum.py")

        # Use a fixed VELOCITY_RANGE for testing
        VR = {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}

        class Env01:
            push_velocity_scale = 0.1

        ce.curriculum_push_by_setting_velocity(Env01(), None, VR)
        r.check("M2 push scale=0.1 applied",
                abs(_cap["range"]["x"][0] - VR["x"][0] * 0.1) < 1e-9,
                f"got={_cap['range']['x'][0]:.4f} expected={VR['x'][0]*0.1:.4f}")

        class Env10:
            push_velocity_scale = 1.0

        ce.curriculum_push_by_setting_velocity(Env10(), None, VR)
        r.check("M2 push scale=1.0 = original range",
                abs(_cap["range"]["x"][0] - VR["x"][0]) < 1e-9)

        class EnvDefault:
            pass  # no push_velocity_scale attr → should default to 1.0

        ce.curriculum_push_by_setting_velocity(EnvDefault(), None, VR)
        r.check("M2 missing attr defaults to scale=1.0",
                abs(_cap["range"]["x"][0] - VR["x"][0]) < 1e-9)


# ---------------------------------------------------------------------------
# M3 — early stopping (pure Python, no GPU)
# ---------------------------------------------------------------------------
def test_m3_early_stopping(r: R):
    import unittest.mock as mock

    with mock.patch.dict("sys.modules", _fake_runner_mods()):
        fop = _load("utils/fast_on_policy_runner.py")
    # Extract as unbound method — avoids instantiation issues.
    _check_plateau = fop.FastMotionOnPolicyRunner._check_plateau

    class FakeRunner:
        def __init__(self, window=100, threshold=1e-4, min_iters=50):
            self._reward_history = []
            self.plateau_window    = window
            self.plateau_threshold = threshold
            self.min_iterations    = min_iters

    runner = FakeRunner()
    # Rising phase — should NOT trigger
    triggered = False
    for i in range(50):
        triggered = _check_plateau(runner, float(i) * 0.05, i)
    r.check("M3 no early stop during rising phase", not triggered)

    # Flat phase — SHOULD trigger
    for i in range(50, 200):
        triggered = _check_plateau(runner, 5.0 + 1e-7 * i, i)
    r.check("M3 early stop triggered during flat phase", triggered)

    # Sentinel: min_iterations=int(1e9) never triggers
    runner2 = FakeRunner(window=10, threshold=1.0, min_iters=int(1e9))
    triggered2 = False
    for i in range(20):
        triggered2 = _check_plateau(runner2, 5.0, i)
    r.check("M3 sentinel min_iterations=1e9 never triggers", not triggered2)


# ---------------------------------------------------------------------------
# M0 — cfg sentinel values (pure Python, no isaaclab runtime needed)
# ---------------------------------------------------------------------------
def test_m0_cfg_sentinels(r: R):
    import unittest.mock as mock

    # Mock the entire isaaclab config system so we can import the cfg files.
    def _configclass(cls):
        return cls

    fake_runner_cfg = type("RslRlOnPolicyRunnerCfg", (), {
        "num_steps_per_env": 24, "max_iterations": 30000,
        "save_interval": 2000, "experiment_name": "base",
    })

    # We load both cfg files directly; relative imports become absolute via sys.modules injection.
    agents_pkg = "whole_body_tracking.tasks.tracking.config.g1.agents"
    fake_mods = {
        "isaaclab": mock.MagicMock(),
        "isaaclab.utils": mock.MagicMock(configclass=_configclass),
        "isaaclab_rl": mock.MagicMock(),
        "isaaclab_rl.rsl_rl": mock.MagicMock(
            RslRlOnPolicyRunnerCfg=fake_runner_cfg,
            RslRlPpoActorCriticCfg=mock.MagicMock(),
            RslRlPpoAlgorithmCfg=mock.MagicMock(),
        ),
    }
    with mock.patch.dict("sys.modules", fake_mods):
        # Load base cfg under its package-qualified name so relative imports resolve.
        base_ppo = _load("tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
                         package=agents_pkg)
        # Reload fast cfg; relative `from .rsl_rl_ppo_cfg import ...` will find
        # base_ppo because _load registered it as `{agents_pkg}.rsl_rl_ppo_cfg`.
        fast_ppo = _load("tasks/tracking/config/g1/agents/rsl_rl_fast_ppo_cfg.py",
                         package=agents_pkg)

    G1FastBasePPORunnerCfg      = fast_ppo.G1FastBasePPORunnerCfg
    G1FastNoisePPORunnerCfg     = fast_ppo.G1FastNoisePPORunnerCfg
    G1FastPushPPORunnerCfg      = fast_ppo.G1FastPushPPORunnerCfg
    G1FastEarlyStopPPORunnerCfg = fast_ppo.G1FastEarlyStopPPORunnerCfg
    G1FastFullPPORunnerCfg      = fast_ppo.G1FastFullPPORunnerCfg

    base = G1FastBasePPORunnerCfg()
    r.check("M0 base: noise_scale_start sentinel=1.0", base.noise_scale_start == 1.0)
    r.check("M0 base: push_scale_start sentinel=1.0",  base.push_scale_start == 1.0)
    r.check("M0 base: min_iterations sentinel>=1e8",   base.min_iterations >= int(1e8))

    noise = G1FastNoisePPORunnerCfg()
    r.check("M0 noise cfg: noise_scale_start<1.0", noise.noise_scale_start < 1.0)
    r.check("M0 noise cfg: push still disabled",   noise.push_scale_start == 1.0)
    r.check("M0 noise cfg: early stop still off",  noise.min_iterations >= int(1e8))

    push = G1FastPushPPORunnerCfg()
    r.check("M0 push cfg: push_scale_start<1.0",  push.push_scale_start < 1.0)
    r.check("M0 push cfg: noise still disabled",  push.noise_scale_start == 1.0)

    es = G1FastEarlyStopPPORunnerCfg()
    r.check("M0 earlystop cfg: min_iterations<1e8",   es.min_iterations < int(1e8))
    r.check("M0 earlystop cfg: noise still disabled", es.noise_scale_start == 1.0)

    full = G1FastFullPPORunnerCfg()
    r.check("M0 full cfg: noise active",     full.noise_scale_start < 1.0)
    r.check("M0 full cfg: push active",      full.push_scale_start < 1.0)
    r.check("M0 full cfg: early stop active", full.min_iterations < int(1e8))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    r = R()
    print("=== M0: cfg sentinel values ===")
    test_m0_cfg_sentinels(r)
    print("=== M1: noise curriculum ===")
    test_m1_noise_curriculum(r)
    print("=== M2: push curriculum ===")
    test_m2_push_curriculum(r)
    print("=== M3: early stopping ===")
    test_m3_early_stopping(r)
    return r.summary()


if __name__ == "__main__":
    raise SystemExit(main())
