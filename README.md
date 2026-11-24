# 高たんぱく食材の「たんぱく質あたり栄養密度」クラスタリング

食材 i について、たんぱく質を基準にした各栄養素 α の充足比

$$C_{i,\alpha}=\frac12\left(\frac{M_{i,\alpha}R_{m,p}}{M_{i,p}R_{m,\alpha}}+\frac{M_{i,\alpha}R_{f,p}}{M_{i,p}R_{f,\alpha}}\right)$$

をベクトル化し、k-means++ でクラスタリングする。理想は全成分 1
（その食材でたんぱく質の推奨量を満たすと、他の栄養素もちょうど推奨量を満たす）。

## 実行

```bash
pip install pandas openpyxl scikit-learn matplotlib
python3 src/analysis.py              # log2 空間（既定）→ results/log/
python3 src/analysis.py --space raw  # C をそのまま → results/raw/
python3 src/analysis.py --cap 1.0    # C を min(1, C) で頭打ち → results/log_cap1/
```

主なオプション: `--cap`（C の上限。1.0 で過剰分を評価しない）、`--k`（k を固定）、`--min-protein`（既定 10 g）、`--min-protein-egg-dairy`（既定 3 g）、
`--floor`（log 化時の C=0 の下限、既定 1/16）、`--with-biotin`。

## データ

| 用途 | 出典 | ファイル |
|---|---|---|
| M（100 g あたり成分値） | 文部科学省「日本食品標準成分表（八訂）増補2023年」第2章データ（2026-03-27 更新版） https://www.mext.go.jp/a_menu/syokuhinseibun/mext_00001.html | `data/raw/food_composition_2023.xlsx` |
| R（18〜29 歳の推奨量） | 厚生労働省「日本人の食事摂取基準（2025年版）」 https://www.mhlw.go.jp/stf/newpage_44138.html | `data/dri_2025_18-29.csv` |

- 推奨量がない栄養素（B12・パントテン酸・ビオチン）は目安量を使っている。
- 正誤表（令和7年3月25日）を確認済み。上の値に関わる訂正はない。

## 前処理と設計上の判断

- **対象食品**: 豆類・種実類・魚介類・肉類はたんぱく質 10 g/100 g 以上。卵類・乳類は牛乳とヨーグルトを含めるため 3 g/100 g 以上。乳児用調製粉乳（栄養強化品）は除外。
- **栄養素**: B1, B2, ナイアシン当量, B6, B12, 葉酸, パントテン酸, 亜鉛。ビオチンは対象食品の半数以上で未測定（`-`）なので、既定では除外。
- **成分表の記号**: `Tr` と `(0)` は 0、`(数値)` は推定値として数値、`-` は欠測。欠測がある食品は除外。
- **クラスタリング空間**: 既定は log2 C。C は B12 で数百倍まで振れるため、そのままだと少数の外れ値（貝類など）が支配する。log2 にすると「2 倍過剰」と「1/2 不足」が対称になり、理想点は原点になる。C=0（植物性食品の B12 など）は 1/16 で下限処理する。
- **k**: 2〜12 のシルエット係数が最大の k を採用。k-means++、n_init=100、seed=0。
- クラスタ番号は、中心が理想点に近い順に 0, 1, … と振っている。

## 出力（`results/<space>/`）

- `foods_clustered.csv`: 食品ごとの C、クラスタ、理想点との距離 `dist_to_ideal`、不足のみの指標 `shortfall_rms_log2`、C≥1 の栄養素数 `n_satisfied`
- `cluster_summary.csv`: クラスタ中心（C 値）、件数、食品群の内訳、代表例
- `k_selection.csv` / `k_selection.png`、`cluster_centers.png`（ヒートマップ）、`pca_clusters.png`（PCA 射影と理想点）

## 解析一覧

| ディレクトリ | 定義 | 空間 | 採用 k |
|---|---|---|---|
| `results/log/` | C（上限なし） | log2 | 11 |
| `results/raw/` | C（上限なし） | そのまま | 2（外れ値 1 件が単独クラスタ。参考用） |
| `results/log_cap1/` | min(1, C) | log2 | 12（k=13〜20 もシルエットは約 0.28〜0.29 で横ばい） |

`log_cap1` では過剰分を評価しないため、理想点（原点）からの距離は不足量だけで決まる。
