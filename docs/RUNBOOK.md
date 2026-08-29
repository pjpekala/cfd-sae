# Training Runbook — Colab-First

> **Use this as the ordered checklist.** Deep flag tables, hardware YAML, and troubleshooting live in [HOWTO.md](HOWTO.md) §§5-8 and [README.md](../README.md). Same pipeline works locally with `uv run` — only the `hardware` preset and `--base-dir` change.

Four stages chain on one `--run-name` (e.g. `myrun`). Artifacts survive VM teardown on **Drive** at `/content/drive/MyDrive/cfd-sae/` and can be pulled locally on demand.

```
train_mgn (MGN next-frame, hidden 128, 9 message passes, colab epochs 25)
  → extract_embeddings (frozen h_i, paper uses split test)
    → train_sae (Linear→ReLU encoder, unit-L2 decoder, recon+lambda*L1, early-stop on val)
      → analyze (Top-K Variance/MeanAbs/Entropy) → 05_analysis.ipynb (interactive)
```

For every step this runbook shows three equivalent ways to run it:

- **(A) Wrapper** — `bash scripts/colab.sh ...` convenience (calls `colab exec` via Python `subprocess`)
- **(B) Interactive shell** — `colab console -s cfd` → run raw `uv` commands manually (recommended manual path)
- **(C) Non-interactive `colab exec`** — `printf '...' | colab exec -s cfd` alternative for scripting (collapsed)

Pick one path and stick with one auth strategy (oauth2 or `--auth=adc`) — mixing is unreliable. `(B)` is the preferred manual option: you `console` into the VM and type the `uv run python ...` commands directly, with history, tab-complete, and `Ctrl-C` support.

---

## 0. Prerequisites (once)

```bash
uv tool install google-colab-cli
colab whoami   # prints active Google account / scopes / expiry
# Drive mounted later belongs to THIS account — check before downloading 16 GB.
```

Auth — whichever you use must prefix **every** `colab` command:

- **oauth2 (default):** browser consent, token cached `~/.config/colab-cli/token.json`. Switch account: `rm ~/.config/colab-cli/token.json && colab whoami`.
- **adc (student/alt account):**
  ```bash
  gcloud auth application-default login \
    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
  colab --auth=adc whoami
  # then use colab --auth=adc ... everywhere; scripts/colab.sh does NOT pass --auth
  ```

VM name is `cfd` throughout. `colab.sh` hardcodes `SESSION=cfd`, `VM_DIR=/content/cfd-sae`, `DRIVE_BASE=/content/drive/MyDrive/cfd-sae`. `--dry-run` previews any wrapper command without executing.

```bash
bash scripts/colab.sh --dry-run run train_mgn --run-name myrun --epochs 25
```

Repo URL for `sync` is taken from `git remote get-url origin` (ssh `git@host:org/repo.git` auto-converted to https). Override with `bash scripts/colab.sh --repo-url https://github.com/you/cfd-sae sync`.

---

## 1. Provision + Mount Drive (once per VM)

**Wrapper**

```bash
bash scripts/colab.sh new          # T4 free-tier; --gpu L4|A100|CPU to change
bash scripts/colab.sh drive        # approve in browser, press Enter
bash scripts/colab.sh status       # optional: confirm VM + Drive
```

**Manual `colab` CLI**

```bash
colab new -s cfd --gpu T4
colab drivemount -s cfd            # same browser consent
colab status -s cfd
# ADC variant: colab --auth=adc new -s cfd --gpu T4 ; colab --auth=adc drivemount -s cfd
```

---

## 2. Sync Code + Deps (re-run after every `git push`)

**Wrapper**

```bash
bash scripts/colab.sh sync
```

**Interactive shell (preferred manual)**

```bash
colab console -s cfd              # or: bash scripts/colab.sh console
# inside VM shell:
if [ ! -d /content/cfd-sae ]; then
  git clone --depth 1 https://github.com/pjpekala/cfd-sae.git /content/cfd-sae
else
  git -C /content/cfd-sae pull --ff-only
fi
if ! which uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd /content/cfd-sae && uv sync
exit
# ADC: colab --auth=adc console -s cfd
```

<details><summary>Non-interactive <code>colab exec</code> alternative</summary>

```bash
colab exec -s cfd --timeout 600 <<'PY'
import os, shutil, subprocess
vm = '/content/cfd-sae'
repo = 'https://github.com/pjpekala/cfd-sae.git'
if not os.path.isdir(vm):
    subprocess.run(['git', 'clone', '--depth', '1', repo, vm], check=True)
else:
    subprocess.run(['git', '-C', vm, 'pull', '--ff-only'], check=True)
if shutil.which('uv') is None:
    subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
    os.environ['PATH'] = os.path.expanduser('~/.local/bin') + os.pathsep + os.environ.get('PATH','')
subprocess.run(['bash','-lc','cd /content/cfd-sae && uv sync'], check=True)
PY
```

</details>

---

## 3. Download Data (once per Drive, ~16 GB)

**Wrapper**

```bash
bash scripts/colab.sh data
```

**Interactive shell**

```bash
colab console -s cfd
# inside VM:
cd /content/cfd-sae && uv run python scripts/download_data.py --data-dir /content/drive/MyDrive/cfd-sae/data --skip-existing
ls -lh /content/drive/MyDrive/cfd-sae/data  # should show train/valid/test.tfrecord
exit
```

<details><summary>Non-interactive alternative</summary>

```bash
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/download_data.py --data-dir /content/drive/MyDrive/cfd-sae/data --skip-existing", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
```

</details>

Shell alternative `bash scripts/download_data.sh ./data` exists but `uv run python scripts/download_data.py --skip-existing` is preferred.

---

## 4. Train MGN

Preset `configs/hardware/colab.yaml`: `hidden_dim 128`, `message_passing_steps 9`, `epochs 25`, `batch_size 1`, SAE `batch_size 128` (see `HOWTO.md §10`).

**Wrapper**

```bash
bash scripts/colab.sh run train_mgn --run-name myrun --epochs 25
# resume after disconnect:
bash scripts/colab.sh run train_mgn --run-name myrun --resume --epochs 25
# what it previews under --dry-run: colab exec -s cfd --timeout 3600 <<'PY' with guards
# on /content/cfd-sae, /content/drive, and data/train.tfrecord
```

**Interactive shell (preferred manual)**

```bash
colab console -s cfd
# inside VM:
cd /content/cfd-sae && uv run python scripts/train_mgn.py --hardware colab --run-name myrun --epochs 25
# resume after disconnect:
cd /content/cfd-sae && uv run python scripts/train_mgn.py --hardware colab --run-name myrun --resume --epochs 25
exit
```

<details><summary>Non-interactive <code>colab exec</code> alternative</summary>

```bash
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/train_mgn.py --hardware colab --run-name myrun --epochs 25", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/train_mgn.py --hardware colab --run-name myrun --resume --epochs 25", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
```

</details>

**Live TensorBoard (explicit sidecar, Drive-persistent)**

```bash
# Terminal 1 — start TB sidecar for this run (polls Drive every 5s, ok with delay):
bash scripts/colab.sh tensorboard myrun --port 6006
# or interactive:
colab console -s cfd
# inside VM:
nohup tensorboard --logdir /content/drive/MyDrive/cfd-sae/runs/myrun/tb --host 0.0.0.0 --port 6006 --reload_interval 5 > /tmp/tb-myrun.log 2>&1 &
cat /tmp/tb-myrun.log   # if Colab prints a https://*.colab.googleusercontent.com proxy URL, open it
exit
# fallback local polling (same Drive path after colab download or Drive mount):
uv run tensorboard --logdir runs/myrun/tb --port 6006   # → http://localhost:6006, 5-10s delay

# Terminal 2 — then start/resume training (logs to runs/myrun/tb on Drive):
bash scripts/colab.sh run train_mgn --run-name myrun --epochs 25
# add --no-tb to disable
```

<details><summary>Non-interactive TB alternative</summary>

```bash
printf 'import subprocess; subprocess.run("nohup tensorboard --logdir /content/drive/MyDrive/cfd-sae/runs/myrun/tb --host 0.0.0.0 --port 6006 --reload_interval 5 > /tmp/tb-myrun.log 2>&1 &", shell=True, check=True)\n' | colab exec -s cfd --timeout 60
```

</details>

**Smoke (quick wiring check, local or VM)**

```bash
uv run python scripts/train_mgn.py --hardware colab --run-name smoke-colab --epochs 1 --max-steps 100
# logs to runs/smoke-colab/tb; view: uv run tensorboard --logdir runs/smoke-colab/tb
```

**Verify:** `checkpoints/myrun/best.pt` + `checkpoints/myrun/epoch_*.pt` (last 3 kept) + `runs/myrun/resolved_config.yaml` + `runs/myrun/run_metadata.json` + `runs/myrun/tb/events.*` (on Drive: `/content/drive/MyDrive/cfd-sae/runs/myrun/tb`). Resume validates `mgn/sae/batch_size` against snapshot — mismatch → new `--run-name`.

**Local equivalent:** `uv run python scripts/train_mgn.py --hardware desktop --run-name myrun --epochs 50` (or `--hardware macbook`).

---

## 5. Extract Embeddings

Paper trains SAE on **test** embeddings (default). One `.npy` `[N,128]` per `(example, frame)`.

**Wrapper**

```bash
bash scripts/colab.sh run extract_embeddings --run-name myrun --split test
```

**Interactive shell**

```bash
colab console -s cfd
# inside VM:
cd /content/cfd-sae && uv run python scripts/extract_embeddings.py --hardware colab --run-name myrun --split test
# cross-run MGN: --mgn-run mgn-exp-1  (use another run's best.pt)
# smoke cap: --max-examples 10
ls /content/drive/MyDrive/cfd-sae/embeddings/myrun/test | head
exit
```

<details><summary>Non-interactive alternative</summary>

```bash
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/extract_embeddings.py --hardware colab --run-name myrun --split test", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
```

</details>

**Verify:** `embeddings/myrun/test/ex00000_fr0000.npy` exists.

---

## 6. Train SAE

Decoder rows re-normalized to unit L2 each step. Z-score stats computed over **train subset only** → `embeddings/myrun/embedding_stats.npz`. Early-stop on held-out recon MSE (`--val-frac 0.1 --patience 5 --min-epochs 1`); `--no-val` disables.

**Wrapper**

```bash
bash scripts/colab.sh run train_sae --run-name myrun --epochs 25
# resume: bash scripts/colab.sh run train_sae --run-name myrun --resume --epochs 25
```

**Interactive shell**

```bash
colab console -s cfd
# inside VM (TB sidecar optional, same as §4):
nohup tensorboard --logdir /content/drive/MyDrive/cfd-sae/runs/myrun/tb --host 0.0.0.0 --port 6006 --reload_interval 5 > /tmp/tb-myrun.log 2>&1 &
cd /content/cfd-sae && uv run python scripts/train_sae.py --hardware colab --run-name myrun --epochs 25
# fixed epochs without early-stop:
cd /content/cfd-sae && uv run python scripts/train_sae.py --hardware colab --run-name myrun --epochs 25 --no-val
exit
# TB logs: train/loss + train/l1 every step, val/mse per epoch → runs/myrun/tb
```

<details><summary>Non-interactive alternative</summary>

```bash
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/train_sae.py --hardware colab --run-name myrun --epochs 25", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
```

</details>

**Smoke**

```bash
uv run python scripts/train_sae.py --hardware colab --run-name myrun --epochs 1 --max-steps 100 --no-val
```

**Verify:** `checkpoints/myrun/sae/best.pt` + `checkpoints/myrun/sae/best_val.pt` + `embeddings/myrun/embedding_stats.npz` + `runs/myrun/tb/events.*` (on Drive).

---

## 7. Analyze (CLI)

Computes `recon_mse`, `mean_l1_codes`, `frac_inactive_codes`, Top-K per Variance/MeanAbs/Entropy → `runs/myrun/analysis.json`.

**Wrapper**

```bash
bash scripts/colab.sh run analyze --run-name myrun --split test
```

**Interactive shell**

```bash
colab console -s cfd
# inside VM:
cd /content/cfd-sae && uv run python scripts/analyze.py --hardware colab --run-name myrun --split test --top-k 20
cat /content/drive/MyDrive/cfd-sae/runs/myrun/analysis.json | head -n 40
exit
```

<details><summary>Non-interactive alternative</summary>

```bash
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/analyze.py --hardware colab --run-name myrun --split test --top-k 20", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 3600
```

</details>

**Local:** `uv run python scripts/analyze.py --hardware auto --run-name myrun --split test --top-k 20`.

---

## 8. Notebook — `notebooks/05_analysis.ipynb` (interactive, no training)

Loads an **existing** run's SAE checkpoint + embeddings from `embeddings/myrun/test` and `checkpoints/myrun/sae`.

**Open on Colab (choose one):**

- **From Drive VM:** after `sync`, file is at `/content/cfd-sae/notebooks/05_analysis.ipynb` — open via Colab File browser or `colab console -s cfd` then serve.
- **Upload:** upload `05_analysis.ipynb` directly in Colab UI, then `!git clone https://github.com/pjpekala/cfd-sae` + `!uv sync` + set paths.

**Run cells:**

1. Cell 1 Setup — imports `src/visualize`, `scripts.analyze`, `src.data.cylinder_flow`; prints `setup OK`.
2. Cell 2 Load a run — set widgets `run_name='myrun'`, `split='test'` → `load_all_codes` → `salient_scores(bins=50)` → prints `variance/mean_abs/entropy max/mean`.
3. Cell 3 Top-K bars — `top_latents_bar` for 3 scores (Top-20).
4. Cells 4-5 Interactive latent — slider `0..hidden-1` → `latent_histogram` + `spatial_scatter` of most-activated frame (node color = latent activation). Reuses `src/utils/checkpoint.py:load_latest` and `scripts/analyze.py:load_normalizer`.

Needs `ipywidgets` (dev dep via `uv sync`). `get_env(hardware='auto')` resolves to `colab` on VM, `macbook` on Darwin, `desktop` elsewhere.

**Local:**

```bash
uv sync
jupyter lab  # open notebooks/05_analysis.ipynb, set run_name/split widgets, Run All
```

---

## 9. Pull Artifacts Back (optional — Drive is source of truth)

**Wrapper**

```bash
bash scripts/colab.sh download myrun   # unpacks into ./checkpoints ./embeddings ./runs
```

**Interactive shell + manual download**

```bash
colab console -s cfd
# inside VM:
tar czf /content/cfd-sae-myrun.tgz -C /content/drive/MyDrive/cfd-sae checkpoints/myrun embeddings/myrun runs/myrun
ls -lh /content/cfd-sae-myrun.tgz
exit
# back on local:
colab download -s cfd /content/cfd-sae-myrun.tgz ./cfd-sae-myrun.tgz
tar xzf cfd-sae-myrun.tgz -C .
# ADC: colab --auth=adc download -s cfd /content/cfd-sae-myrun.tgz ./...
```

<details><summary>Non-interactive alternative</summary>

```bash
printf 'import subprocess; subprocess.run("tar czf /content/cfd-sae-myrun.tgz -C /content/drive/MyDrive/cfd-sae checkpoints/myrun embeddings/myrun runs/myrun", shell=True, check=True)\n' \
  | colab exec -s cfd --timeout 300
colab download -s cfd /content/cfd-sae-myrun.tgz ./cfd-sae-myrun.tgz
tar xzf cfd-sae-myrun.tgz -C .
```

</details>

---

## 10. Stop VM & Debug

```bash
bash scripts/colab.sh stop        # wrapper
colab stop -s cfd                 # manual (or colab --auth=adc stop -s cfd)
colab status -s cfd               # manual status check
colab log -s cfd -o colab_run_log.md   # optional log export
```

Interactive shell is also your debug shell: `colab console -s cfd` → `nvidia-smi`, `ls /content/drive/MyDrive/cfd-sae/checkpoints/myrun`, `cat /content/drive/MyDrive/cfd-sae/runs/myrun/run_metadata.json`, `exit` to leave.

---

## Quick Smoke (end-to-end in minutes)

```bash
uv run python scripts/download_data.py --data-dir data --skip-existing
uv run python scripts/train_mgn.py --hardware colab --run-name verify-smoke --epochs 1 --max-steps 50
uv run python scripts/extract_embeddings.py --hardware colab --run-name verify-smoke --split test --max-examples 10
uv run python scripts/train_sae.py --hardware colab --run-name verify-smoke --epochs 1 --max-steps 100 --no-val
uv run python scripts/analyze.py --hardware colab --run-name verify-smoke --split test --top-k 5
# or via wrapper: bash scripts/colab.sh run train_mgn --run-name verify-smoke --epochs 1 --max-steps 50 && ...
```

Expected: no `FileNotFoundError`/`Resume mismatch`, and `runs/verify-smoke/analysis.json` with `recon_mse`, `top_k_latents`.

---

## Artifact Layout (run `myrun`)

```
checkpoints/myrun/best.pt              # MGN best
checkpoints/myrun/epoch_*.pt           # last 3 periodic
checkpoints/myrun/sae/best.pt          # SAE best train
checkpoints/myrun/sae/best_val.pt      # SAE best val (restored at end)
embeddings/myrun/test/ex*.npy          # [N,128] per frame
embeddings/myrun/embedding_stats.npz   # train-only z-score mean/std
runs/myrun/resolved_config.yaml
runs/myrun/run_metadata.json
runs/myrun/analysis.json
runs/myrun/tb/events.*                 # TensorBoard logs (persistent on Drive)
```
`cat runs/myrun/resolved_config.yaml` / `run_metadata.json` for reproducibility. `ls runs/myrun/tb` / `ls /content/drive/MyDrive/cfd-sae/runs/myrun/tb` for TB.

---

## Also Local (for reference)

```bash
uv sync
uv run python scripts/download_data.py --data-dir data --skip-existing
uv run python scripts/train_mgn.py --hardware macbook --run-name myrun --epochs 5
uv run python scripts/extract_embeddings.py --hardware macbook --run-name myrun --split test
uv run python scripts/train_sae.py --hardware macbook --run-name myrun --epochs 50
uv run python scripts/analyze.py --hardware macbook --run-name myrun --split test
```

Hardware auto-resolves `colab` in Colab, `macbook` on Darwin, `desktop` otherwise. Override storage: `--base-dir /mnt/fast-ssd/cfd-sae`.

---

## Troubleshooting

See [HOWTO.md §12](HOWTO.md#12-troubleshooting) for full table. Common runbook hits: `No MGN checkpoint`→run `train_mgn` first with same `--run-name`; `No embeddings at .../test`→run `extract_embeddings --split test`; `Resume config mismatch`→new `--run-name` or restore preset; `CUDA OOM`→ smaller `sae.batch_size` or smoke caps; `torch_scatter` missing→ `uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html`.
