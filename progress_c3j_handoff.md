# c3j eğitim işini UHeM'e gönderme — devralma notu

Bu dosya, başka bir oturumda devam edecek ajan için **eksiksiz** bir devralma
prompt'udur. Yapılması gereken son adımı atmayı amaçlar: c3j_720_polish
eğitimini UHeM Altay'da A100×4 üzerinde başlatmak.

---

## Bağlam (5 satır)

- `sat010_e699` modelini değerlendirdik; 256/256 başarılı, 9.4 s, ama
  thruster saturation duty %50, pitch 30° peak, yaw error 43°.
- sat010'dan **c3j_720_polish** stage'iyle warm-start eğitim planlandı: daha
  sıkı pitch/yaw/depth ve saturation cezaları, settle 0.25 s, XY settle 0.7 m.
- Local'de 3 commit hazır, push edilmedi.
- A100×4 sbatch + submit script hazır, çalıştırılmadı.
- UHeM'e SSH bağlanılmadı, iş gönderilmedi.

---

## Local repo durumu (taze çekildi `git log` ile)

Branch: `main`
HEAD: `d7b2717d16e10770f5031cedbaa7fe0a1b23d409`
`origin/main`: `16ab57c` (3 commit gerisinde)

Push edilmemiş 3 commit:

```text
d7b2717 feat(eval): TensorBoard-driven roll evaluation suite
46925f6 feat(roll): multi-GPU train driver and A100x4 Slurm runner
85436bf feat(roll): add settle XY drift limit and c3j_720_polish curriculum
```

Bu commit'ler içeriği itibariyle güvenli, lokal testler **46 passed**.

Working tree'de **dokunulmaması gereken** uncommitted dosyalar var (başka iş
parçacığından kalma, c3j ile alakasız):

- `src/auvrl/asset_zoo/vehicles/taluy/taluy_body_mjcf.xml`
- `src/auvrl/scripts/demo/_odometry_panel.py`
- `src/auvrl/scripts/demo/model_record.py`
- `src/auvrl/scripts/demo/taluy_velocity_play.py`
- `src/auvrl/scripts/demo/taluy_velocity_viewer.py`
- `src/auvrl/tasks/velocity/mdp/velocity_command.py`

Bunlara **dokunma**, git add etme. Bir sonraki ajan istemediği sürece bunlar
local kalsın.

---

## Tamamlanan iş (yapılmasına gerek yok)

### Kod değişiklikleri

1. **`src/auvrl/tasks/roll/curriculum.py`**
   - `RollCurriculumStage` dataclass'a `settle_xy_drift_limit_m: float | None = None`
     alanı eklendi.
   - `roll_env_kwargs()` çıktısına yeni alan dahil.
   - **`c3j_720_polish`** stage'i eklendi (parametreleri aşağıda).
   - `c3h_720_hold_0p05_xy_tight`, `c3i_sat005/sat010/sat020` için
     `settle_xy_drift_limit_m=1.25` set edildi.

   c3j parametreleri (yapılan değişiklikler **bu** değerler — başka değer
   göndermeyin):
   ```python
   target_roll_deg=720.0, episode_length_s=20.0, settle_window_s=0.25,
   k_prog=5.0, k_xy=0.25, k_pitch=2.0, k_yaw=0.8, k_depth=0.8, k_smooth=0.030,
   excess_pitch_deg=70.0, excess_depth_error_m=2.0, excess_xy_drift_m=2.5,
   settle_pitch_limit_deg=30.0, settle_yaw_limit_deg=60.0,
   settle_ang_vel_limit_rad_s=1.5, settle_depth_error_limit_m=0.8,
   settle_xy_drift_limit_m=0.7,
   terminal_success_weight=200.0, terminal_failure_weight=-40.0,
   k_action_effort=0.005, k_thruster_saturation=0.30,
   thruster_saturation_threshold=0.75,
   ```

2. **`src/auvrl/tasks/roll/roll_env_cfg.py`**
   - `settle_xy_drift_limit_m` argümanı kabul ediliyor, validasyon var,
     task_success terminasyonuna geçiriliyor.

3. **`src/auvrl/tasks/roll/mdp/terminations.py`**
   - `task_success` fonksiyonu `settle_xy_drift_limit_m` parametresini
     `settle_condition_mask`'a iletir.

4. **`src/auvrl/tasks/roll/runtime.py`**
   - `settle_condition_mask` artık opsiyonel `xy_drift_m` ve
     `xy_drift_limit_m` parametrelerini destekler.

5. **`src/auvrl/scripts/train/taluy_roll.py`** — multi-GPU desteği
   - `_resolve_device`: `LOCAL_RANK` env var'ı varsa `cuda:{local_rank}`
     döner, `MUJOCO_EGL_DEVICE_ID` set eder.
   - `_make_log_dir`: `AUVRL_LOG_DIR` (tam yol) veya `AUVRL_LOG_STAMP`
     (sadece stamp) env var override'larını destekler — tüm rank'ların aynı
     dizine yazması için.
   - `main`: `RANK`/`WORLD_SIZE` okur, seed'i rank başına ofsetler, sadece
     rank-0 yaml dump'lar ve banner basar.

6. **`jobs/slurm/taluy_roll_a100x4.sbatch`** (YENİ)
   - `--gres=gpu:a100:4`, `-c 32`, 12 saat.
   - `AUVRL_LOG_DIR` önceden hesaplanıp export edilir.
   - `MASTER_PORT` `SLURM_JOB_ID`'den deterministik üretilir.
   - `uv run python -m torch.distributed.run --standalone --nproc-per-node=4 -m auvrl.scripts.train.taluy_roll …`
     ile launch eder. rsl_rl `WORLD_SIZE > 1`'i görünce `init_process_group`
     ile NCCL'i kendi başlatır.

7. **`jobs/slurm/submit_c3j_from_sat010.sh`** (YENİ)
   - sat010 checkpoint path'i `BASE_CKPT` ile, varsayılan:
     `logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt`
   - Default exports: `STAGE=c3j_720_polish`, `NUM_GPUS=4`,
     `NUM_ENVS_PER_GPU=1024` (total 4096), `NUM_STEPS_PER_ENV=256`,
     `ITERS=700`, `SAVE_INTERVAL=50`, `LEARNING_RATE=3e-4`,
     `ENTROPY_COEF=0.003`, `DESIRED_KL=0.006`, `RESUME_MODE=weights-only`.
   - `sbatch jobs/slurm/taluy_roll_a100x4.sbatch` çağırır.

8. **Yeni eval suite** (c3j ile alakasız ama aynı PR'da: bir önceki konuşmada
   yapılan iş): `src/auvrl/scripts/eval/taluy_roll_tensorboard.py` ve
   `tests/tasks/roll/test_roll_tensorboard_eval.py`.

### Doğrulanmış testler

```bash
uv run --with pytest python -m pytest tests/tasks/roll/ -q
# 46 passed, 44 warnings in 18.84s
```

### Local sat010 eval çıktısı

`logs/tensorboard_eval/roll/c3i_compare_20260521T183527/sat010_e699/` —
özellikle `sat010_analysis.html` (2.7 MB, tek dosya). Bu c3j gerekçesini
detaylı açıklar.

---

## Eksik kalan iş (yapılacak)

Tek hedef: **UHeM Altay'da c3j eğitimini A100×4'le başlatmak.**

### 1. Push (ZORUNLU)

```bash
cd /home/emin/code/auvrl
git push origin main
```

Uyarı: working tree'de uncommitted dosyalar var ama `git push` HEAD'i gönderir,
sadece commit'lenmiş 3 commit ulaşır. **Bunlara dokunma.**

### 2. UHeM'e SSH ve hazırlık

```bash
ssh makine          # ~/auv/auv-rl içine düşer
~/auv/bin/auv-env   # interactif AUV ortamı
cd ~/auv/auv-rl

# Local commit'leri çek
git fetch origin
git log --oneline origin/main -5
# Bekleyen: d7b2717, 46925f6, 85436bf
git pull --ff-only origin main
```

### 3. sat010 checkpoint UHeM'de var mı kontrol

```bash
CKPT="logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt"
ls -lh "$CKPT" 2>/dev/null && echo "ckpt OK" || echo "ckpt MISSING"
```

- **VARSA**: doğrudan submit'e geç.
- **YOKSA**: local makineden rsync gerek. Local'den çalıştır (UHeM'de değil):
  ```bash
  rsync -avP /home/emin/code/auvrl/logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt \
    makine:/ari/users/btutak/auv/auv-rl/logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt
  ```
  Not: `makine` host alias'ı SSH config'de tanımlı; `ssh makine` çalışıyor demek rsync'in de çalışacağı anlamına gelir (`scp/rsync` aynı config'i kullanır).

### 4. UV bağımlıları taze mi kontrol

```bash
uv sync 2>&1 | tail -5
```

Eğer yeni paket gerekiyorsa (bu commit'ler hiç yeni dep eklemedi) yine de
emin olmak için bir çalıştır.

### 5. Quick smoke test (öneri — tek adımda atlanabilir)

İstenirse 50 iterasyonluk hızlı smoke test (~5-10 dk) ile pipeline'ı
doğrula:

```bash
ITERS=50 NUM_ENVS_PER_GPU=256 SAVE_INTERVAL=25 \
  JOB_NAME=roll-c3j-smoke \
  RUN_NAME=c3j_smoke_$(date +%H%M%S) \
  bash jobs/slurm/submit_c3j_from_sat010.sh
```

Bekle, log'a bak:

```bash
squeue -u btutak
tail -F ~/auv/logs/slurm/roll-c3j-smoke-*.out
```

Görmek istediğin bayraklar:

- `host=...` (compute node ismi)
- `distributed=true world_size=4`
- 4 rank için ayrı `rank=N/4 device=cuda:N` satırı
- nvidia-smi'de 4 GPU görünüyor
- İlk iterasyonun tamamlanması (~30-60 s içinde)

**Sorun çıkarsa** olası şüpheliler:
- `torch.distributed.run` import path'i: `python -m torch.distributed.run` mı yoksa `torchrun` mı? Sbatch'te `python -m torch.distributed.run` kullanılıyor (uv run + module ile). Eğer çalışmazsa: `uv run torchrun --standalone --nproc-per-node=4 -m auvrl.scripts.train.taluy_roll ...`
- `MASTER_PORT` çakışması: aynı node'da başka bir job 25xxx port'unu kullanıyor olabilir; deterministik 25000 + job_id%10000 ama yine çakışırsa `MASTER_PORT=29500` set et.
- EGL device: 4 farklı rank, 4 farklı GPU. `MUJOCO_EGL_DEVICE_ID` setdefault ile rank'a göre set edilir; `setdefault` çağrısı önemli (override etme).
- mjlab/rsl_rl multi-GPU yolu hiç bizim repo'da test edilmedi. mjlab kendi train.py'sinde aynı pattern'i kullanıyor → çalışmalı, ama ilk seferde sürpriz olabilir.

### 6. Asıl iş (smoke OK ise)

```bash
bash jobs/slurm/submit_c3j_from_sat010.sh
```

Default ITERS=700 ile A100×4'te tahmini 6-8 saat.

Çıktı dosyaları:
- Slurm out/err: `~/auv/logs/slurm/roll-c3j-polish-<JOBID>.{out,err}`
- Training stdout: `~/auv/auv-rl/logs/remote_runs/c3j_720_polish_from_sat010_a100x4_<STAMP>.log`
- Checkpoint'ler + TB events: `~/auv/auv-rl/logs/rsl_rl/taluy_roll_v1/<STAMP>_c3j_720_polish_from_sat010_a100x4_<STAMP>/`

### 7. İzleme

```bash
# Job durumu
squeue -u btutak
# Logu canlı izle
tail -F ~/auv/logs/slurm/roll-c3j-polish-*.out
# Training stdout
tail -F ~/auv/auv-rl/logs/remote_runs/c3j_720_polish_from_sat010_a100x4_*.log
# Checkpoint'leri listele
ls -lh ~/auv/auv-rl/logs/rsl_rl/taluy_roll_v1/*c3j*polish*/model_*.pt
```

### 8. Eğitim bitince — değerlendirme

c3j tamamlandığında, local'de aynı eval suite ile karşılaştırma:

```bash
# Local çalıştır:
uv run python -m auvrl.scripts.eval.taluy_roll_tensorboard \
  --checkpoint <path/to/c3j/model_699.pt> \
  --label c3j_e699 \
  --suite c3j_vs_sat010 \
  --curriculum-stage c3j_720_polish \
  --num-envs 256 --device cuda --overwrite

uv run python -m auvrl.scripts.eval.taluy_roll_tensorboard \
  --checkpoint logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt \
  --label sat010_e699 \
  --suite c3j_vs_sat010 \
  --curriculum-stage c3i_720_hold_0p10_sat010 \
  --num-envs 256 --device cuda
```

İzlenecek başarı göstergeleri:

- **Bekleniyor**: thruster duty cycle %50 → %25, pitch peak 30° → 15°,
  yaw error 43° → 20°, action_rate peak 14 → 5.
- **Acceptable trade-off**: ilk-done süresi 9.4 s → ~12 s (uzayabilir,
  policy daha disiplinli oluyor).
- **Kötü işaret**: success_rate %100'den %85'in altına düşerse stage çok
  agresif demektir; o zaman k_pitch / k_yaw'u biraz geri al.

---

## Yapma'lar (DO NOT)

- Working tree'deki diğer uncommitted dosyalara dokunma (mjcf.xml, demo
  scripts, velocity_command.py).
- `progress_c3j_handoff.md`'yi commit etme — bu sadece devralma notu.
- UHeM'de `~/auv` dışına yazma. Tüm log/cache `~/auv` altında.
- Manuel olarak `torch.distributed.init_process_group` çağırma — rsl_rl
  zaten yapıyor.
- `--num-envs` toplam değil **rank başına** olduğunu unutma; 4×1024=4096
  effective batch.

---

## Tek satır görev (TLDR)

> "Local'de `d7b2717` HEAD'i origin/main'e push et. UHeM'e `ssh makine` ile
> bağlan, `cd ~/auv/auv-rl && git pull --ff-only`. sat010 checkpoint orada
> mı kontrol et. `bash jobs/slurm/submit_c3j_from_sat010.sh` ile sbatch'i
> gönder. `squeue -u btutak` ve `tail -F ~/auv/logs/slurm/roll-c3j-polish-*.out`
> ile izle. İlk iterasyonun tamamlandığını gör, 'distributed=true
> world_size=4' bayrağını doğrula. Sorun çıkarsa MASTER_PORT veya
> torchrun çağırma şekli (`python -m torch.distributed.run` vs `torchrun`)
> şüpheli."

---

## Bilinen riskler

| Risk | Olasılık | Çözüm |
|---|---|---|
| `python -m torch.distributed.run` uv ortamında bulunmaz | Düşük | `uv run torchrun` ile değiştir |
| EGL device per-rank doğru atanmaz, MuJoCo render hatası | Orta | mjlab pattern'i aynısı; sorun çıkarsa `MUJOCO_GL=egl` ve `MUJOCO_EGL_DEVICE_ID` log'da görünmeli |
| `MjlabOnPolicyRunner` distributed mode'da checkpoint load'u rank'lara dağıtmıyor | Düşük | mjlab tüm rank'larda load() çağırıyor; bizim de aynı. map_location=device olduğu için her rank kendi GPU'suna yüklüyor |
| Slurm partition `a100q` 4-GPU job'a hazır değil | Düşük | UHeM'de A100 node'ları 4×A100; `--gres=gpu:a100:4` standart |
| sat010 checkpoint UHeM'de yok | Orta | rsync komutu hazır (yukarıda Adım 3) |
| WANDB veya TB logger init rank-0'da sorun | Düşük | rsl_rl bunu zaten yapıyor |

---

## Faydalı referanslar

- mjlab'in multi-GPU pattern referansı:
  `/home/emin/code/auvrl/.venv/lib/python3.12/site-packages/mjlab/scripts/train.py`
  (özellikle `run_train` ve `launch_training`).
- rsl_rl multi-GPU init:
  `.venv/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py`
  satır 207-249 (`_configure_multi_gpu`).
- UHeM Altay sbatch örneği:
  `/home/emin/code/auvrl/jobs/slurm/taluy_roll.sbatch` (eski 1×A100 versiyon).
- sat010 detaylı analizi:
  `/home/emin/code/auvrl/logs/tensorboard_eval/roll/c3i_compare_20260521T183527/sat010_e699/sat010_analysis.html`
