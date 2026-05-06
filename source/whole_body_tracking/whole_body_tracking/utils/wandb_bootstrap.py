from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_first_nonempty_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _version_tuple(raw: str) -> tuple[int, int, int]:
    nums: list[int] = []
    cur = ""
    for ch in raw:
        if ch.isdigit():
            cur += ch
        elif cur:
            nums.append(int(cur))
            cur = ""
    if cur:
        nums.append(int(cur))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _pip_install(spec: str, verbose: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-U", spec]
    if verbose:
        print(f"[wandb_bootstrap] pip install: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _ensure_versions(auto_fix: bool, verbose: bool) -> tuple[str | None, str | None]:
    wandb_ver = _pkg_version("wandb")
    protobuf_ver = _pkg_version("protobuf")

    wandb_ok = wandb_ver is not None and _version_tuple(wandb_ver) >= (0, 19, 0)
    protobuf_ok = protobuf_ver is not None and _version_tuple(protobuf_ver) < (5, 0, 0)

    if auto_fix:
        if not wandb_ok:
            _pip_install("wandb>=0.19", verbose)
        if not protobuf_ok:
            _pip_install("protobuf>=3.20.2,<5.0.0", verbose)
        wandb_ver = _pkg_version("wandb")
        protobuf_ver = _pkg_version("protobuf")

    return wandb_ver, protobuf_ver


def _ensure_wandb_api_key(verbose: bool) -> str | None:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key

    candidates = [
        _repo_root() / "wandb_api_key.txt",
        Path.home() / ".wandb_api_key",
    ]
    for path in candidates:
        key = _read_first_nonempty_line(path)
        if key:
            os.environ["WANDB_API_KEY"] = key
            if verbose:
                print(f"[wandb_bootstrap] WANDB_API_KEY loaded from {path}")
            return key
    return None


def _login_wandb(key: str, verbose: bool) -> None:
    import wandb

    wandb.login(key=key, relogin=False)
    if verbose:
        print("[wandb_bootstrap] wandb login ensured.")


def ensure_wandb_runtime(
    *,
    auto_fix: bool = True,
    require_auth: bool = False,
    verbose: bool = True,
) -> dict[str, str | bool | None]:
    """Ensure wandb/protobuf compatibility and optional non-interactive login."""
    wandb_ver, protobuf_ver = _ensure_versions(auto_fix=auto_fix, verbose=verbose)
    wandb_ok = wandb_ver is not None and _version_tuple(wandb_ver) >= (0, 19, 0)
    protobuf_ok = protobuf_ver is not None and _version_tuple(protobuf_ver) < (5, 0, 0)

    if not auto_fix and (not wandb_ok or not protobuf_ok):
        raise RuntimeError(
            "Incompatible runtime dependencies. Expected wandb>=0.19 and protobuf<5. "
            f"Current wandb={wandb_ver}, protobuf={protobuf_ver}. "
            "Please run: python -m pip install -U 'wandb>=0.19' 'protobuf>=3.20.2,<5.0.0'"
        )

    key = _ensure_wandb_api_key(verbose=verbose)

    if require_auth:
        if not key:
            raise RuntimeError(
                "WANDB_API_KEY not found. Set env WANDB_API_KEY or provide key file "
                "at repo root 'wandb_api_key.txt' or '~/.wandb_api_key'."
            )
        _login_wandb(key, verbose=verbose)

    return {
        "wandb_version": wandb_ver,
        "protobuf_version": protobuf_ver,
        "has_api_key": bool(key),
    }
