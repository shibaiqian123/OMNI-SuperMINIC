"""Evaluate G1 policy variants under one deterministic benchmark protocol.

This command does not train policies. It verifies that every checkpoint uses
the same motion horizon and robot variant, runs the existing closed-loop
benchmark, and writes checkpoint/schema hashes beside the results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, summarize
from supermimic.policy.checkpoint import checkpoint_metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(schema: dict) -> dict:
    return {
        "future_frames": schema["goal"]["future_frames"],
        "robot": schema["robot"],
        "observation": schema["observation"],
    }


def main(
    checkpoints: list[Path],
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    manifest: Path | None = None,
    num_envs: int = 1024,
    device: str = "cuda:0",
    out: Path = Path("omnimimic_ablation.json"),
) -> None:
    """Compare trained checkpoints; all checkpoints and clips must be explicit inputs."""
    if len(checkpoints) < 2:
        raise ValueError("ablation requires at least two checkpoints")
    files = [str(path) for path in discover_npz(npz_dir, names=clips, manifest=manifest)]
    if not files:
        raise FileNotFoundError(f"no evaluation clips under {npz_dir}")

    rows = []
    expected_contract = None
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        schema = checkpoint_metadata(payload)
        contract = _contract(schema)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError(f"checkpoint contract mismatch: {path}: {contract} != {expected_contract}")

        policy, cfg = load_policy_from_checkpoint(path, device)
        results = run_benchmark(
            policy,
            files,
            device=device,
            num_envs=num_envs,
            future_frames=cfg["env"]["future_frames"],
            anchor_obs=cfg["env"].get("anchor_obs", False),
            lock_waist=cfg["env"].get("lock_waist", False),
            lock_waist_refs=cfg["env"].get("lock_waist_refs", "feasible"),
        )
        rows.append(
            {
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": _sha256(path),
                "schema": schema,
                "summary": summarize(results),
                "failed_clips": [r.name for r in results if r.failed],
                "worst_clips": [
                    {"name": r.name, "norm_return": r.norm_return, "fail_step": r.fail_step}
                    for r in sorted(results, key=lambda item: item.norm_return)[:10]
                ],
            }
        )

    report = {
        "protocol": {
            "npz_dir": str(npz_dir.resolve()),
            "manifest": None if manifest is None else str(manifest.resolve()),
            "clips": [Path(path).stem for path in files],
            "num_envs": num_envs,
            "device": device,
            "deterministic_seed": 0,
        },
        "results": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out} ({len(rows)} checkpoints, {len(files)} clips)")


if __name__ == "__main__":
    tyro.cli(main)
