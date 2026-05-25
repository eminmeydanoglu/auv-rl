# c3j Sonrası 3 Deney Müfredatı Önerisi

## Ana teşhis

`c3j` tek seferde fazla şeyi sıkılaştırdı:

- **Success şartları çok sıkılaştı**
  - `settle_xy`: `1.25m → 0.7m`
  - `settle_pitch`: `45° → 30°`
  - `settle_yaw`: `90° → 60°`
  - `settle_depth`: `1.5m → 0.8m`
  - `settle_window`: `0.1s → 0.25s`

- **Reward dengesi aynı anda çok değişti**
  - `k_prog`: `6.0 → 5.0`
  - `k_xy`: `0.15 → 0.25`
  - `k_pitch`: `1.0 → 2.0`
  - `k_yaw`: `0.4 → 0.8`
  - `k_smooth`: `0.01 → 0.03`
  - `k_thruster_saturation`: `0.10 → 0.30`
  - `thruster_saturation_threshold`: `0.85 → 0.75`

Sonuç:

- **Pitch iyileşti:** `29.9° → 19.2°`
- **Yaw az iyileşti:** `42.6° → 39.2°`
- **Ama XY tamamen bozuldu:** `0.535m → 2.498m`
- **Success:** `100% → 0%`
- **Saturation düşmedi:** `0.487 → 0.502`

Bu yüzden yeni müfredatların ana prensibi şu olmalı:

> **C3j gibi her şeyi aynı anda sıkılaştırmayalım. Success’i koruyup tek tek baskı ekleyelim.**

Aşağıdaki üç müfredatı bu mantıkla öneriyorum.

---

# Deney 1: `c3k_720_soft_polish`

## Amaç

`c3j`’nin hedefini daha güvenli yapmak:

- Pitch/yaw/depth biraz iyileşsin.
- XY success korunmaya devam etsin.
- Saturation’a hafif baskı gelsin.
- Roll progress motivasyonu düşmesin.

Bu, üç deney içinde **en güvenli genel polish** adayı.

## Parametreler

| Parametre | Base sat010 | c3j | Öneri `c3k_soft_polish` |
|---|---:|---:|---:|
| `settle_window_s` | `0.10` | `0.25` | `0.15` |
| `k_prog` | `6.0` | `5.0` | `6.0` |
| `k_xy` | `0.15` | `0.25` | `0.18` |
| `k_pitch` | `1.0` | `2.0` | `1.35` |
| `k_yaw` | `0.4` | `0.8` | `0.55` |
| `k_depth` | `0.5` | `0.8` | `0.60` |
| `k_smooth` | `0.010` | `0.030` | `0.014` |
| `excess_pitch_deg` | `80` | `70` | `80` |
| `excess_depth_error_m` | `2.0` | `2.0` | `2.0` |
| `excess_xy_drift_m` | `2.5` | `2.5` | `2.5` |
| `settle_pitch_limit_deg` | `45` | `30` | `40` |
| `settle_yaw_limit_deg` | `90` | `60` | `80` |
| `settle_ang_vel_limit_rad_s` | `2.0` | `1.5` | `1.8` |
| `settle_depth_error_limit_m` | `1.5` | `0.8` | `1.2` |
| `settle_xy_drift_limit_m` | `1.25` | `0.7` | `1.05` |
| `terminal_success_weight` | `180` | `200` | `200` |
| `terminal_failure_weight` | `-40` | `-40` | `-45` |
| `k_action_effort` | `0.003` | `0.005` | `0.0035` |
| `k_thruster_saturation` | `0.10` | `0.30` | `0.12` |
| `thruster_saturation_threshold` | `0.85` | `0.75` | `0.85` |

## Neden yüksek başarı şansı var?

- `k_prog` base’teki gibi `6.0` kalıyor; roll hedefi terk edilmiyor.
- XY settle `0.7m` gibi agresif değil, `1.05m`.
- Pitch/yaw baskısı var ama c3j kadar sert değil.
- Saturation threshold düşürülmüyor; `0.75` hamlesi c3j’de fayda vermedi.

## Beklenen sonuç

- **Success:** yüksek ihtimalle `%90-100`
- **XY peak:** `0.55-0.8m`
- **Pitch peak:** `24-27°` civarına inebilir
- **Yaw:** küçük iyileşme
- **Saturation:** küçük iyileşme veya aynı seviye

## Bu deneyin görevi

Bu bizim **ana adayımız** olmalı. Eğer sadece birini seçsek, bunu seçerdim.

---

# Deney 2: `c3l_720_xy_guard`

## Amaç

`c3j`’nin ana çöküş sebebi XY idi. Bu deneyin amacı:

- Base’in success davranışını korumak
- XY drift’i daha kontrollü öğretmek
- Attitude/saturation tarafını fazla kurcalamamak

Bu, **XY drift problemini izole eden** deney.

## Parametreler

| Parametre | Base sat010 | c3j | Öneri `c3l_xy_guard` |
|---|---:|---:|---:|
| `settle_window_s` | `0.10` | `0.25` | `0.10` |
| `k_prog` | `6.0` | `5.0` | `6.0` |
| `k_xy` | `0.15` | `0.25` | `0.28` |
| `k_pitch` | `1.0` | `2.0` | `1.0` |
| `k_yaw` | `0.4` | `0.8` | `0.4` |
| `k_depth` | `0.5` | `0.8` | `0.55` |
| `k_smooth` | `0.010` | `0.030` | `0.010` |
| `excess_pitch_deg` | `80` | `70` | `80` |
| `excess_depth_error_m` | `2.0` | `2.0` | `2.0` |
| `excess_xy_drift_m` | `2.5` | `2.5` | `2.0` |
| `settle_pitch_limit_deg` | `45` | `30` | `45` |
| `settle_yaw_limit_deg` | `90` | `60` | `90` |
| `settle_ang_vel_limit_rad_s` | `2.0` | `1.5` | `2.0` |
| `settle_depth_error_limit_m` | `1.5` | `0.8` | `1.5` |
| `settle_xy_drift_limit_m` | `1.25` | `0.7` | `0.9` |
| `terminal_success_weight` | `180` | `200` | `220` |
| `terminal_failure_weight` | `-40` | `-40` | `-55` |
| `k_action_effort` | `0.003` | `0.005` | `0.003` |
| `k_thruster_saturation` | `0.10` | `0.30` | `0.10` |
| `thruster_saturation_threshold` | `0.85` | `0.75` | `0.85` |

## Neden yüksek başarı şansı var?

- Base’ten çok uzaklaşmıyor.
- Pitch/yaw/saturation gibi eksenler sabit kalıyor.
- Sadece XY öğrenme sinyali güçleniyor.
- `excess_xy_drift_m=2.0`, base’in `0.535m` peak’ine göre hâlâ çok rahat ama c3j’deki `2.5m fail duvarı`na kadar drift etmeyi daha erken cezalandırır.

## Beklenen sonuç

- **Success:** `%80-100`
- **XY peak:** `0.5-0.75m`
- **Pitch/yaw:** base’e yakın kalır
- **Saturation:** base’e yakın kalır

## Bu deneyin görevi

Eğer bu başarılı olursa, c3j’nin probleminin “sıkı attitude/saturation” değil, **XY shaping’in yanlış birleşimi** olduğunu doğrularız.

---

# Deney 3: `c3m_720_smooth_sat_guard`

## Amaç

Saturation’ı azaltmayı denemek ama c3j’deki gibi sert threshold/ceza ile policy’yi bozmamak.

C3j’de saturation tarafında yapılan şeyler:

- `k_thruster_saturation=0.30`
- `threshold=0.75`
- `k_smooth=0.030`
- `k_action_effort=0.005`

Bunlar birlikte çok agresif oldu ve saturation düşmedi. Bu deneyde saturation baskısını daha yumuşak kuruyoruz:

- Threshold aynı kalıyor: `0.85`
- Saturation weight az artıyor: `0.10 → 0.15`
- Smoothness artıyor ama c3j kadar değil
- Success şartları base’e yakın kalıyor

## Parametreler

| Parametre | Base sat010 | c3j | Öneri `c3m_smooth_sat_guard` |
|---|---:|---:|---:|
| `settle_window_s` | `0.10` | `0.25` | `0.10` |
| `k_prog` | `6.0` | `5.0` | `6.0` |
| `k_xy` | `0.15` | `0.25` | `0.16` |
| `k_pitch` | `1.0` | `2.0` | `1.10` |
| `k_yaw` | `0.4` | `0.8` | `0.45` |
| `k_depth` | `0.5` | `0.8` | `0.55` |
| `k_smooth` | `0.010` | `0.030` | `0.020` |
| `excess_pitch_deg` | `80` | `70` | `80` |
| `excess_depth_error_m` | `2.0` | `2.0` | `2.0` |
| `excess_xy_drift_m` | `2.5` | `2.5` | `2.5` |
| `settle_pitch_limit_deg` | `45` | `30` | `45` |
| `settle_yaw_limit_deg` | `90` | `60` | `90` |
| `settle_ang_vel_limit_rad_s` | `2.0` | `1.5` | `2.0` |
| `settle_depth_error_limit_m` | `1.5` | `0.8` | `1.5` |
| `settle_xy_drift_limit_m` | `1.25` | `0.7` | `1.25` |
| `terminal_success_weight` | `180` | `200` | `200` |
| `terminal_failure_weight` | `-40` | `-40` | `-45` |
| `k_action_effort` | `0.003` | `0.005` | `0.006` |
| `k_thruster_saturation` | `0.10` | `0.30` | `0.15` |
| `thruster_saturation_threshold` | `0.85` | `0.75` | `0.85` |

## Neden yüksek başarı şansı var?

- Success şartları neredeyse base ile aynı.
- Ana değişiklik action/smoothness/saturation tarafında.
- `threshold=0.75` yapmıyoruz çünkü c3j’de fayda vermedi.
- `k_thruster_saturation=0.20` sat020’de success’i biraz düşürmüştü; burada `0.15` daha güvenli orta nokta.

## Beklenen sonuç

- **Success:** `%90-100`
- **Saturation time mean:** `0.487 → 0.45-0.48` arası olabilir
- **Action L2 / action rate:** düşme beklerim
- **XY:** base’e yakın kalmalı
- **Pitch/yaw:** küçük iyileşme olabilir ama ana hedef değil

## Bu deneyin görevi

Bu deney, “saturation’ı success’i bozmadan azaltabilir miyiz?” sorusuna cevap verir.

---

# Üç deneyin rolleri

| Deney | Ana hedef | Risk | Başarı şansı |
|---|---|---|---|
| `c3k_720_soft_polish` | Genel kalite polish | Düşük-orta | En yüksek |
| `c3l_720_xy_guard` | XY drift’i kontrol etmek | Orta | Yüksek |
| `c3m_720_smooth_sat_guard` | Saturation/action smoothness azaltmak | Düşük | Yüksek |

Benim sıralamam:

1. **`c3k_720_soft_polish`**
2. **`c3m_720_smooth_sat_guard`**
3. **`c3l_720_xy_guard`**

Ama üçünü paralel koşturacaksan çok iyi bir deney seti olur; çünkü her biri farklı hipotezi test ediyor.

---

# Beklenen karar kriteri

Her model için eval’den sonra şunlara bakmanı öneririm:

## Minimum geçiş kriteri

- **Success rate:** `>= 0.95`
- **Excess XY drift rate:** `<= 0.05`
- **XY peak mean:** `< 0.8m`
- **Depth peak mean:** `< 0.5m`
- **Saturation time mean:** `< 0.49`
- **Pitch peak mean:** `< 0.45 rad` yaklaşık `< 25.8°`

## Kazanan nasıl seçilir?

- Eğer `c3k` success’i koruyup pitch’i düşürürse:
  - **Ana devam checkpoint’i o olsun.**

- Eğer `c3m` saturation’ı düşürüp success’i korursa:
  - **c3k ile birleştirilecek ikinci adım olabilir.**

- Eğer `c3l` XY’yi ciddi iyileştirirse:
  - **c3k sonrası daha sıkı settle curriculum için temel olur.**

---

# En önemli tasarım kararı

Bence şu anda **c3j tarzı tek büyük sıçramadan kaçınmalıyız**.

Özellikle aynı anda şunları yapmamalıyız:

- `settle_xy=0.7`
- `settle_window=0.25`
- `k_pitch=2.0`
- `k_yaw=0.8`
- `k_sat=0.30`
- `threshold=0.75`
- `k_prog=5.0`

Bu kombinasyon, c3i’den gelen iyi policy’yi “daha iyi stabilize ol” yönünde değil, **XY fail local optimum’una** taşımış gibi görünüyor.

Benim önerim:

- **Success’i koru**
- **Roll progress’i düşürme**
- **Sıkılaştırmayı tek eksende yap**
- **C3j’deki sert saturation threshold hamlesini şimdilik kullanma**
- **XY settle’ı bir anda `0.7m` yapma; önce `1.05m` veya `0.9m` dene**

Bu üç deney bu mantığı kapsıyor.
