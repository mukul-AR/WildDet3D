# JENGA — project instructions

Prompt-free, single-view **9-DoF 3D box detector** for warehouse unloading.
Frozen SAM3 + LingBot-Depth encoders → **Stage 1** (visible box) → **Stage 2**
(dimension-conditioned: inherits rotation, picks the scene SKU, places the full
actual box). Canonical branch: `nearface-anchor` (== `main`).

**`HANDOFF.md` has the full design, run history, and current status — read it first.**

## Environment
- Python: `.venv/bin/python`, always with `PYTHONPATH=.`.
- Core code: `wilddet3d/dense/` (`model.py`, `head.py`, `stage2.py`, `loss.py`,
  `metrics.py`, `sim_jenga_dataset.py`). Trainer: `scripts/train_jenga.py`.
  End-to-end eval: `scripts/jenga_eval_e2e.py`.
- Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests/dense/`.
- Sim data (local): `/storage/3dl_sim_data/` (13.2k-scene dump).

## Conventions
- **W&B entity is `anyware-robotics`** — the script default (`mukul-ganwal`) fails.
  Always pass `--wandb-entity anyware-robotics`.
- **Headline metric = the graspable subset (`vis_frac ≥ 0.6`)** — heavily-occluded
  boxes are picked later, so their IoU isn't actionable now.
- 3D IoU is **Monte-Carlo** (pure-torch; no pytorch3d/CUDA ops on this stack).
- Stage 2 **inherits rotation** from the visible box — do not add a rotation head.
- The SKU catalog is a per-scene **input**, never baked into weights.
- Input is **locked at 1008²** (SAM3 RoPE).
- Eval/export scripts rebuild the model from the checkpoint's saved `args`
  (`fpn_level`, `head_width`, `head_convs`) — preserve those in the ckpt.

## H100 training box
- `ssh ubuntu@209.20.157.13` (1× H100 80GB, key-based, ephemeral disk).
- Code is synced by **rsync from this dev box** (the H100 has no GitHub auth):
  `rsync -azR <files> ubuntu@209.20.157.13:WildDet3D/`.
- Run there with `.venv/bin/python`, `PYTHONPATH=.`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, inside a **tmux** session.
  Checkpoints: `~/WildDet3D/ckpt/<run>/jenga_last.pt`. W&B key: `~/.wandb_env`.

## Gotchas
- **Never `pkill -f "scripts/train_jenga.py"`** — the pattern self-matches the ssh
  shell running it AND kills the tmux server. Use `pkill -f "train_jenga[.]py"`
  (the `[.]` regex form) or kill by PID.
- **Backgrounded local commands don't survive the turn** (sandbox teardown). Long
  jobs — training, large transfers, viz servers — run in a **remote tmux** or
  `nohup`, never as a backgrounded local Bash command.
