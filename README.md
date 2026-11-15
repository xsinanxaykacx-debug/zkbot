# ZK 2.0

Bağımsız, kronolojik ve sızıntı kontrollü futbol tahmin sistemi.

## İlk kapsam
- 1X2 (H/D/A)
- Alt/Üst 2.5
- KG Var/Yok

## Veri kaynakları
İlk denetim kaynakları: `180k.xlsx`, `DATA.csv`, `EPL.csv`. Ham dosyalar yerel ortamda salt okunur tutulur; bu depoya yüklenmez. Kaynaklar ayrı ayrı şema denetiminden ve kapsam analizinden geçmeden birleştirilmez.

## Veri güvenliği kuralları
- Eksik oranlar sıfıra çevrilmez; eksik kalır.
- Eksik skorlar hedef üretmez; hedef `NA` kalır.
- Takım adları yalnızca onaylı kanonik eşleştirme tablosuyla eşlenir.
- Tam tarih + ev sahibi + deplasman eşleşmeleri otomatik eşleşme adayıdır.
- ±1 gün ve fuzzy eşleşmeler yalnızca manuel inceleme adayıdır.
- Takım adı benzerliği tek başına kulüp kimliğini kanıtlamaz.
- Eğitim / doğrulama / test kronolojik ayrılır; test dönemi kilitli tutulur.
- Maç öncesi olmayan hiçbir bilgi feature olarak kullanılamaz.

## Geliştirme sırası
1. Kaynak şeması ve veri sözlüğü denetimi
2. Takım kimliği ve eşleşme raporları
3. Hedef üretimi ve veri kalite testleri
4. Kronolojik baseline modeller
5. Ayrı market modelleri, kalibrasyon ve kilitli test

Model eğitimi, gerçek kaynak dosyaları doğrulanmadan başlatılmaz.

## Kurulum
```bash
python -m pip install -e ".[dev]"
pytest
```


## Locked dataset v1

The normalized, leakage-conscious combined dataset is built by GitHub Actions from the pinned source-file SHA-256 values recorded in [data/locks/zk2-locked-v1.json](data/locks/zk2-locked-v1.json). The compressed CSV and manifest are published as the `zk2-locked-dataset-v1` workflow artifact; raw source files are not duplicated into the repository. The lock contains 11,272 matches from DATA.csv and EPL.xlsm, with zero cross-source team overlap, zero exact-match overlap, zero duplicate match keys, and no missing targets.

Rebuild with the **Build locked ZK 2.0 dataset** workflow. Do not overwrite the v1 manifest or dataset hash; create a new lock version if either source file changes.
