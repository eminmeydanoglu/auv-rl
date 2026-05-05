## 2026-05-05 720 Wave Uygulama Raporu

### Kısa karar

Bu turda 540 ara durağını ana yol olmaktan çıkarıp iyi B baseline üzerinden doğrudan 720 varyantlarını deniyoruz. Seçilen başlangıç checkpoint'i:

```text
logs/rsl_rl/taluy_roll_v1/2026-05-04_17-09-30_c2c_360_hold_0p10_from_c2b_best_env1024_steps256_20260504_170830/model_650.pt
```

Bu seçimde `B-from-A model_650`, `B-direct model_650` kadar yüksek reward/depth kalitesi vermedi; fakat 720 için asıl darboğaz olan `xy_drift`, attitude hatası ve saturation tarafında daha iyi göründü. Önceki 720 denemesinde araç hedefe ulaşmıştı, depth de iyiydi; sorun 720 sonrası yan drift ve yerleşme kalitesiydi. Bu yüzden 720 wave için B-from-A daha doğru başlangıç kabul edildi.

### Eğitilecek varyantlar

Bu wave dört A100 işi olarak tasarlandı:

```text
c3c_720_reach_loose_control
c3d_720_reach_xy_moderate
c3e_720_hold_0p05_xy_light
c3f_720_hold_0p10_soft
```

`c3c_720_reach_loose_control`, kontrol koşusu. Eski `c3b_720_reach_0p02` ayarlarına yakın kalır; tek temel fark daha iyi B checkpoint'ten başlamasıdır. Beklenti: `target_reached_last = 1.0` korunmalı ve eski 720 drift'i olan yaklaşık `0.96 m` değerinin altına inmelidir.

`c3d_720_reach_xy_moderate`, aynı kısa 720 reach görevini XY cezasını artırarak dener. Beklenti: hedefe ulaşma bozulmadan `xy_drift_m` ciddi şekilde düşmelidir. Eğer success belirgin düşmeden drift azalırsa sonraki refinement için en değerli aday budur.

`c3e_720_hold_0p05_xy_light`, 720 sonrası 0.05 saniyelik kısa hold ister. Beklenti: `settle_counter_s_last` yaklaşık `0.056 s` seviyesine oturmalı, failure termination'lar sıfır kalmalı ve drift kontrol run'ından çok kötüleşmemelidir.

`c3f_720_hold_0p10_soft`, 0.10 saniyelik daha zor ama hâlâ yumuşak hold denemesidir. Beklenti: 720 reach davranışı hayatta kalırsa bu stage bizi deployment-quality settle aşamasına hazırlayan en iyi köprü olabilir. Eğer success çok düşerse bunu refine değil, sadece teşhis koşusu olarak kullanacağız.

### İnceleme planı

Final checkpoint tek başına karar verdirmeyecek. Her run için en az şu checkpoint'ler karşılaştırılacak:

```text
model_200.pt
model_300.pt
model_450.pt
model_600.pt
model_699.pt
```

Önce eleme kriterleri:

```text
target_reached_last == 1.0
nan_detected == 0
excess_pitch == 0
excess_depth_error == 0
excess_xy_drift == 0
terminal_failure == 0
```

Sonra seçim kriterleri:

```text
xy_drift_m düşük olmalı, tercihen <= 0.65-0.75 m
settle_counter_s_last stage hedefini karşılamalı
task_success son pencere ortalaması rakiplerden kopmamalı
depth_abs_error_m <= 0.10-0.15 m bandında kalmalı
pitch_abs_rad ve yaw_abs_error_rad düşük kalmalı
root_ang_speed_rad_s düşme eğilimi göstermeli
body_wrench_saturation_fraction artmamalı
```

Beklenen iyi sonuç, 720 hedefe ulaşıp eski D koşusundaki `xy_drift ~= 0.96 m` problemini düşüren bir checkpoint bulmaktır. Eğer `c3d` drift'i düşürür ve success'i korursa sonraki aşama düşük learning rate ile XY/settle refinement olur. Eğer `c3e` veya `c3f` hold'u koruyarak makul drift verirse, onları daha iyi temel kabul ederiz. Eğer tüm hold varyantları kötüleşirse `c3c` veya `c3d` üzerinden önce reach+drift çözülür, sonra hold tekrar denenir.

### Çalıştırma komutu

Remote repo güncel commit'i çektikten sonra wave şu helper ile gönderilir:

```bash
BASE_CKPT="logs/rsl_rl/taluy_roll_v1/2026-05-04_17-09-30_c2c_360_hold_0p10_from_c2b_best_env1024_steps256_20260504_170830/model_650.pt" \
BASE_LABEL="b_from_a650" \
jobs/slurm/submit_roll_720_wave.sh
```

Submission kayıtları:

```text
logs/remote_runs/roll_720_wave_submissions_<timestamp>.tsv
```

---

Bence bu fikir güçlü: **540’ta durmak şart değil.** Eldeki veri aslında şunu söylüyor: 720’ye çıkmak ana problem değil, 720’den sonra “temiz kalmak” problem. O yüzden rotayı 540 köprüsünden ziyade **720 reach → 720 drift kontrolü → 720 kısa settle → 720 deploy-quality settle** diye kurmak daha mantıklı olabilir.

Önce geçmiş tecrübeyi tek cümleyle bağlayayım: `D / c3b_720_reach_0p02` zaten 720 hedefe ulaştı, `target_reached_last = 1.0`, `roll_progress_ratio ≈ 1.005`, `depth_abs_error ≈ 0.027 m`, failure termination’lar `0`. Bu müthiş önemli. Ama `xy_drift ≈ 0.96 m`, `task_success ≈ 2.8`, `pitch/yaw` da 360’a göre daha kötü. Yani “720 yapamıyoruz” değil, “720 yaparken alanı dağıtıyoruz ve yeterince yerleşemiyoruz.”

Ben bundan sonra rotayı şöyle çizerdim.

**Ana Karar**
İyi bir B baseline varsayıyorsak, yani `B-from-A-best` bize temiz bir `c2c_360_hold_0p10` policy verirse, sonraki ana hedef doğrudan:

```text
720 dereceye çıkmak
```

540’ı şu an ana yol yapmazdım. 540’ın değeri ancak debug/ara teşhis için var: “720 patlarsa 540’a dönüp problemi izole edelim.” Ama 720 zaten patlamadı; sadece driftli ve settle zayıf. Bu durumda 540’ta fazla vakit geçirmek policy’yi gereksiz ara hedefe optimize edebilir.

**Neden Direkt 720 Mantıklı**
Birincisi, mevcut 720 run’da hedefe ulaşma sinyali var. `phi_total_rad_last ≈ 12.63`, yani 720 dereceye denk gelen `4π ≈ 12.57` civarına geliyor. Roll progress doğru yönde.

İkincisi, depth kontrolü 720’de şaşırtıcı derecede iyi. D run finalde `depth_abs_error ≈ 0.027 m`; bu B-direct kadar iyi. Demek ki 720’ye çıkarken araç derinlikte tamamen dağılmıyor.

Üçüncüsü, failure termination’lar yok. `nan`, `excess_pitch`, `excess_depth`, `excess_xy` final pencerede `0`. Bu da ortamın/ödülün “çok sert duvara çarpma” halinde olmadığını gösteriyor.

Dördüncüsü, asıl kötü metrik tek bir yere yoğunlaşıyor: `xy_drift`. Bu çok güzel bir teşhis, çünkü hedefi büyütmekten ziyade yan maliyeti ayarlayabileceğimiz anlamına geliyor.

**Benim Önerdiğim Sonraki Wave**
B baseline geldikten sonra aynı anda 3-4 tane 720 varyantı çalıştıralım. Hepsi B baseline’dan `weights-only` resume olsun. Learning rate ilk wave’de yine `3e-4` kalsın; çünkü önce 720 davranışını keşfetmesini istiyoruz. Daha sonra seçilen checkpoint’i `1e-4` ile refine ederiz.

1. `720_reach_loose_control`
Bu bizim kontrol run’ımız olur. Mevcut `c3b_720_reach_0p02` gibi düşük settle ister, hafif XY cezası kullanır. Ama bunu eski c2a baseline’dan değil, yeni iyi B baseline’dan başlatırız.

Amaç: “Sadece daha iyi B başlangıcı, 720 drift’i düşürüyor mu?” sorusunu cevaplamak.

Beklenti: Hedefe hızlı ulaşmalı. Eğer XY drift eski D’deki `0.96m` yerine `0.65-0.75m` bandına düşerse bile B baseline’ın işe yaradığını anlarız.

2. `720_reach_xy_moderate`
Aynı 720 hedef, yine kısa settle, ama XY cezasını artırıyoruz. Mevcut D’de `k_xy = 0.02` çok zayıf kalmış olabilir. Ben burada `k_xy = 0.05` veya `0.08` denerdim. `excess_xy_drift_m` limitini hemen çok sıkılaştırmazdım; limit hâlâ geniş kalsın, ama reward policy’ye “yanlara gitme” desin.

Amaç: 720’ye ulaşmayı bozmadan drift’i aşağı çekmek.

Başarı eşiği: `target_reached = 1.0`, `xy_drift <= 0.60-0.70m`, `task_success` kontrol run’dan çok düşmesin.

3. `720_hold_0p05_xy_light`
Burada settle penceresini `0.02s`’den `0.05s`’ye çıkarırız. XY cezası hafif-orta olsun, mesela `k_xy = 0.04-0.05`. Bunu çok sert yapmayalım; önce 720 sonrası küçük bir durma davranışı gelsin.

Amaç: “720’den sonra en azından kısa süre stabil kalabiliyor mu?” sorusu.

Başarı eşiği: `settle_counter_s_last ≈ 0.05`, `target_reached = 1.0`, failure yok, `xy_drift <= 0.75m`.

4. `720_hold_0p10_soft`
Bu biraz daha ileri varyant. `settle_window_s = 0.10`, ama final toleransları hâlâ yumuşak. Bunu hemen `c4_720_settle` gibi 1 saniye ve çok sıkı ang-vel/depth koşullarıyla yapmayalım. Çünkü c4 şu an çok büyük sıçrama olur.

Amaç: 720 reach davranışı bozulmadan settle süresini uzatabilir miyiz?

Başarı eşiği: `settle_counter_s_last ≈ 0.10`, `target_reached = 1.0`, `xy_drift <= 0.8m`, `task_success` kontrol run’dan fazla kötüleşmesin.

**Neyi Henüz Yapmazdım**
Ben şu an direkt `c4_720_settle` çalıştırmazdım. Çünkü mevcut `c4_720_settle` mantığı çok daha deployment kalitesinde: `settle_window_s = 1.0`, `settle_ang_vel_limit = 0.25`, `settle_depth_error_limit = 0.15`, daha sıkı final koşulları. Elimizdeki 720 policy henüz `root_ang_speed ≈ 4.06 rad/s` civarında; yani 1 saniye sakin kalma hedefine zıplamak fazla sert olur.

Ayrıca sadece final `model_699.pt`’ye bakmazdım. Geçmiş D run’da en iyi `task_success` finalde değil, ara bölgede görülmüş olabilir; TensorBoard’da D için `task_success best ≈ 3.01 @ step 476`, final `2.79`. Yani 720’de final checkpoint her zaman en iyi checkpoint olmayabilir. Bundan sonra her run için `200 / 300 / 450 / 600 / 699` evaluate etmek şart.

**Karar Mekanizması**
B baseline’dan çıkan bu 720 wave bittikten sonra seçim şöyle olmalı:

```text
Önce hedefe ulaşıyor mu?
Sonra failure var mı?
Sonra xy drift ne kadar?
Sonra settle süresi korunmuş mu?
En son reward ve episode length.
```

Benim pratik skorlamam:

```text
reject if target_reached < 1.0
reject if terminal_failure > 0
prefer xy_drift <= 0.65m
prefer depth <= 0.10m
prefer pitch <= 0.03rad
prefer yaw <= 0.04rad
prefer root_ang_speed decreasing
prefer task_success increasing
```

Eğer `720_reach_xy_moderate` hedefe ulaşıp drift’i ciddi düşürürse onu seçerim. Eğer drift düşüyor ama success çok düşüyorsa önce `720_reach_loose_control` üzerinden devam edip ikinci aşamada XY cezasını artırırım. Eğer `720_hold_0p05` başarıyı bozmadan çalışırsa onu ana yol yaparım; çünkü settle davranışını erken kazandırmak daha sonra c4’e geçişi kolaylaştırır.

**Benim Net Rota**
1. `B-from-A-best` bitsin, en iyi checkpoint’i selector/eval ile seçelim.
2. Bu checkpoint’ten aynı anda 4 tane 720 varyantı başlatalım:
   `720_reach_loose_control`, `720_reach_xy_moderate`, `720_hold_0p05_xy_light`, `720_hold_0p10_soft`.
   3. Her varyantta final değil, aday checkpoint’leri evaluate edelim.
   4. En iyi 720 checkpoint’i seçelim.
   5. Seçilen checkpoint’ten düşük LR ile refine başlatalım:
      `lr = 1e-4`, daha yüksek XY cezası, biraz daha uzun settle.
      6. Ancak bundan sonra `c4_720_settle` benzeri deployment-quality stage’e geçelim.

      Benim tercih ettiğim yol bu: **B baseline → doğrudan 720 multi-probe → en iyi 720 checkpoint seçimi → düşük LR ile drift/settle refinement.** 540’ı ana hat dışına alırız; sadece 720 varyantları tamamen saçılırsa geri dönüp diagnostik köprü olarak kullanırız. Bu daha cesur ama eldeki veriyle savunulabilir bir rota.
