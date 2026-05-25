# Otomatik Curriculum Önerisi (Taluy Roll)

> Hazırlayan: Cascade ile sohbet, 2026-05-23.
> Amaç: Elle `c0 → c1 → c2 → c3j → c3k` stage zinciri yazıp ayrı ayrı submit etme yükünü kaldırmak. Tek runda otomatik ilerleyen bir curriculum'a geçmek.

---

## 1. Mevcut Durum

- **Statik stage**: `RollCurriculumStage` donmuş bir dataclass (`src/auvrl/tasks/roll/curriculum.py`). `target_roll_deg`, `excess_xy_drift_m`, `settle_window_s`, reward katsayıları vs. hepsi sabit.
- **Stage seçimi env yaratımında**: `make_taluy_roll_env_cfg(curriculum_stage="c3j_720_polish")` çağrısı stage parametrelerini tek seferlik enjekte ediyor (`src/auvrl/tasks/roll/config/taluy/env_cfgs.py:27-32`).
- **Stage geçişi runlar arası**: Her stage ayrı sbatch, ayrı checkpoint init (`--resume`), ayrı eval. Bir stage tasarlama-koşma-değerlendirme döngüsü yarım gün – 1 gün sürüyor.
- **mjlab `CurriculumManager` kullanılmıyor**: `cfg.curriculum` dict boş, env `NullCurriculumManager`'a düşüyor (`mjlab/envs/manager_based_rl_env.py:328-332`).

Sonuç: Curriculum tasarımı insan-kapasitesi ile sınırlı; her parametre değişikliği yeni bir Slurm jobu.

---

## 2. Hedef

- **Tek runda c0→c3 progresyonu.** İnsan stage zinciri tasarlamasın.
- **Performansa bağlı ilerleme.** `success_rate` yüksekse zorlaş, düşerse geri çekil.
- **Mevcut statik akışı bozmadan** opt-in bir mod (`auto_curriculum=True`).

---

## 3. mjlab CurriculumManager Mimarisi (kısa özet)

`mjlab/managers/curriculum_manager.py` ve `manager_based_rl_env.py:543-570`:

- Her env reset'inde `_reset_idx` → `curriculum_manager.compute(env_ids)` çağrılır.
- `compute` her kayıtlı term için `func(env, env_ids, **params) -> state` çalıştırır.
- `func` env'in herhangi bir parametresini mutate edebilir (reward weight, termination eşiği, command range).
- Dönen `state` (skaler/dict) TB'ye `Curriculum/<term_name>` olarak otomatik loglanır.

Kayıt:

```python
from mjlab.managers.curriculum_manager import CurriculumTermCfg
cfg.curriculum["xy_settle"] = CurriculumTermCfg(
    func=ratchet_param,
    params={"source": "terminations", "term": "xy_drift_excess",
            "key": "limit_m", "goal": 0.5, "step": 0.1,
            "sr_up": 0.7, "sr_down": 0.3},
)
```

Önemli kısıt: `CurriculumManager` bir **scheduler**'dır, **algoritma değildir**. "Ne zaman zorlaşacağını" yine kod yazıyor (success-rate gating, ALP-GMM, ADR vs.).

---

## 4. Geçiş Planı

### Adım 1 — Stage'i başlangıç + hedef olarak yeniden çerçevele
- `c0_90_discovery` → curriculum'un **başlangıç** değerleri.
- `c3j_720_polish` (veya bir sonraki "c3k") → curriculum'un **hedef** değerleri.
- `RollCurriculumStage` dataclass aynen kalır; sadece artık iki stage çekilip aralarında ratchet edilecek.

### Adım 2 — Generic ratchet term'i yaz
Tek dosya: `src/auvrl/tasks/roll/auto_curriculum.py`.

```python
from __future__ import annotations
import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg

def _episode_success_rate(env) -> float:
    """Son N episode üzerinde rolling success rate. MetricsManager'a
    bağlanmalı; ilk iterasyonda 0 döner."""
    ...

def ratchet_param(env, env_ids, *, source: str, term: str, key: str,
                   goal: float, step: float, sr_up: float, sr_down: float,
                   floor: float | None = None, ceil: float | None = None) -> float:
    sr = _episode_success_rate(env)
    manager = getattr(env, f"{source}_manager")
    cfg = manager.get_term_cfg(term)
    cur = cfg.params[key] if key in cfg.params else getattr(cfg, key)
    direction = 1.0 if goal > cur else -1.0
    if sr > sr_up:
        cur = cur + direction * step
    elif sr < sr_down:
        cur = cur - direction * step
    if direction > 0: cur = min(cur, goal)
    else:             cur = max(cur, goal)
    if floor is not None: cur = max(cur, floor)
    if ceil  is not None: cur = min(cur, ceil)
    if key in cfg.params: cfg.params[key] = cur
    else:                 setattr(cfg, key, cur)
    return cur
```

Ek olarak reward weight için ayrı bir helper (mjlab `reward_manager.get_term_cfg(...).weight` direkt set edilebilir).

### Adım 3 — `make_taluy_roll_env_cfg`'ye `auto_curriculum` flag'i
```python
def make_taluy_roll_env_cfg(
    *,
    auto_curriculum: bool = False,
    auto_start: str = "c0_90_discovery",
    auto_goal:  str = "c3j_720_polish",
    ...,
):
    if auto_curriculum:
        start = get_roll_curriculum_stage(auto_start)
        goal  = get_roll_curriculum_stage(auto_goal)
        roll_kwargs.update(start.roll_env_kwargs())
        cfg.curriculum = build_auto_curriculum(start, goal)
```

`build_auto_curriculum(start, goal)` tabloyu üretir. Eksenler (öneri):

| Eksen | Source | Term | Key | Step | sr_up | sr_down |
|---|---|---|---|---|---|---|
| `target_roll_deg` | command | `roll_target` | `goal_deg` | 90 | 0.7 | 0.3 |
| `excess_xy_drift_m` | termination | `xy_excess` | `limit_m` | ×0.9 | 0.7 | 0.3 |
| `settle_window_s` | (env attr) | — | `settle_window_s` | +0.1 | 0.7 | 0.3 |
| `excess_pitch_deg` | termination | `pitch_excess` | `limit_deg` | −5 | 0.7 | 0.3 |
| `k_xy` | reward | `xy_drift` | `weight` | +0.1 | 0.6 | 0.2 |
| `k_thruster_saturation` | reward | `thruster_sat` | `weight` | +0.005 | 0.6 | 0.2 |

(Gerçek term/key isimleri `roll_env_cfg.py`'den doğrulanmalı; tablo şu an placeholder.)

### Adım 4 — Train script entegrasyonu
`scripts/train/taluy_roll.py`:
- `--auto-curriculum` flag'i.
- Run name'e `auto_<start>_to_<goal>` eklensin.
- TB'de `Curriculum/*` zaten otomatik akacak.

### Adım 5 — Success rate ölçümü
`MetricsManager`'a "last_N_episode_success_rate" term'i ekle. Curriculum term'leri buradan okusun. N≈64 (num_envs ≈ 4096 düşününce 1-2 reset cycle kadar).

---

## 5. Beklenen Faydalar

- **Slurm job sayısı 5-6×'dan 1'e düşer.** Bir long run tüm zinciri yer.
- **Manuel stage tasarımı sona erer.** Yeni bir parametre eklemek = tabloya bir satır.
- **Geri çekilme otomatik.** Policy çökerse curriculum gevşer, rejim ölmez.
- **TB'de live görünürlük.** `Curriculum/xy_settle` plotunu izleyerek curriculum'un nerede tıkandığını görüyorsun.

---

## 6. Riskler ve Uyarılar

- **Şok ayarlama.** Reward weight'i adım başına %5'ten fazla değiştirme; PPO trust region kırılır.
- **Success rate tanımı kritik.** Eğer `task_success_rate` mevcut tanımıyla curriculum gate'i tetiklerse, gate erken açılıp curriculum patlayabilir. Önce mevcut başarı tanımının c3j'da neden 0 olduğunu (bkz. `progress_c3j_handoff.md`) anla.
- **Reward shaping bug'larını çözmez.** Otomatik curriculum kötü reward'la da uzun bir kötü training koşar. Yani önce shaping doğrulanmalı.
- **mjlab sürüm bağımlılığı.** `_reset_idx` içindeki `curriculum_manager.compute()` çağrısı mevcut `.venv` sürümünde var (`manager_based_rl_env.py:544`). Pin'lendiğinden emin olunmalı.
- **Termination param erişimi.** mjlab `TerminationManager` bazı term'leri `params`'la, bazılarını dataclass field'ıyla tutar; ratchet helper her ikisini de desteklemeli (yukarıdaki kodda var).

---

## 7. İleri Aşama (opsiyonel)

CurriculumManager scheduler-katmanı. Üzerine algoritma katmanı koymak istersek:

- **ADR (Automatic Domain Randomization).** Her parametre için `[low, high]` aralığı; performans iyiyse aralık genişler. ~200 satır.
- **ALP-GMM (TeachMyAgent).** Tüm parametre uzayında learning-progress maksimize eden GMM sampler. Curriculum term'i sadece `sampler.sample()` çağırır, parametreyi env'e uygular. ~1 hafta entegrasyon.
- **PBT.** Birden fazla policy paralel; curriculum dahil hyperparametreler popülasyon seçimiyle evrilir. UHeM kotası izin veriyorsa en güçlü çözüm, ama altyapı ağır.

Önce Adım 1-5 ile success-rate-gated ratchet'ı oturt; gerçek bottleneck görüldükten sonra ADR/ALP-GMM'e geç.

---

## 8. Açık Sorular

- [ ] `roll_env_cfg.py` içinde reward/termination term isimleri tam olarak ne?
- [ ] `task_success_rate` şu an `MetricsManager`'da mı yoksa loglarda hesaplanan post-hoc bir metrik mi?
- [ ] `settle_window_s` runtime'da değiştirilebilir mi yoksa env yapısına gömülü mü?
- [ ] c3j'daki `success_rate=0` reward shaping bug'ı mı, yoksa gerçek zorluk artışı mı? (Önce bu ayırt edilmeli, yoksa otomatik curriculum bunu maskeler.)

---

## 9. Önerilen İlk PR Kapsamı

1. `src/auvrl/tasks/roll/auto_curriculum.py` (yeni, ~150 satır).
2. `src/auvrl/tasks/roll/config/taluy/env_cfgs.py` — `auto_curriculum` flag.
3. `src/auvrl/scripts/train/taluy_roll.py` — `--auto-curriculum` CLI.
4. `MetricsManager`'a rolling success-rate term'i.
5. Tek smoke run (50 iter, num_envs küçük) — TB'de `Curriculum/*` plotları görünüyor mu doğrulama.

İnsan-saat tahmini: ~1 gün kod + yarım gün doğrulama. Sonra ilk full run.
