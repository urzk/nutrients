"""高たんぱく食材の「たんぱく質あたり栄養密度」ベクトル C_i を k-means++ でクラスタリングする。

C_{i,α} = 1/2 * ( M_{i,α} R_{m,p} / (M_{i,p} R_{m,α}) + M_{i,α} R_{f,p} / (M_{i,p} R_{f,α}) )

C_{i,α} = 1 は「その食材でたんぱく質の推奨量をちょうど満たすと、栄養αの推奨量もちょうど満たす」ことを意味する。
理想ベクトルは全成分 1。

使い方:
    python3 src/analysis.py                # 既定: log2 空間, 自動で k 選択
    python3 src/analysis.py --space raw    # C をそのまま使う
    python3 src/analysis.py --k 6          # k を固定
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
FOOD_XLSX = ROOT / "data/raw/food_composition_2023.xlsx"
DRI_CSV = ROOT / "data/dri_2025_18-29.csv"

from matplotlib import font_manager

_JP_FONTS = ["Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "WenQuanYi Zen Hei"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in _JP_FONTS if f in _installed] + ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 食品群コード -> 区分
GROUPS = {
    "04": "豆類", "05": "種実類", "10": "魚介類", "11": "肉類", "12": "卵類", "13": "乳類",
}
PLANT = {"04", "05"}
EGG_DAIRY = {"12", "13"}
# 栄養強化された乳児用食品は成人の「食材」として不適当なので除外
EXCLUDE_NAME_PATTERNS = ["乳児用"]

# 解析に使う栄養素（成分識別子 -> 表示名）。ビオチンは欠測が過半のため既定では除外。
NUTRIENTS = {
    "THIA": "B1", "RIBF": "B2", "NE": "ナイアシン当量", "VITB6A": "B6",
    "VITB12": "B12", "FOL": "葉酸", "PANTAC": "パントテン酸", "ZN": "亜鉛",
}
OPTIONAL = {"BIOT": "ビオチン"}


def parse_value(s) -> float:
    """成分表の記号を数値化する。Tr/(Tr)/(0) は 0、括弧付き推定値は数値、'-'(未測定) は NaN。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().strip("()").strip()
    if s in ("Tr", ""):
        return 0.0 if s == "Tr" else np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_foods(min_protein: float, min_protein_egg_dairy: float, groups: set[str]) -> pd.DataFrame:
    raw = pd.read_excel(FOOD_XLSX, sheet_name="表全体", header=None, dtype=str)
    ids = raw.iloc[11]
    body = raw.iloc[12:].copy()
    body.columns = [f"c{i}" if (pd.isna(v) or i == 3) else str(v).strip() for i, v in enumerate(ids)]
    body = body.rename(columns={"c0": "group", "c1": "code", "c3": "name"})
    body = body[body["group"].isin(groups)].copy()
    cols = ["PROT-", *NUTRIENTS, *OPTIONAL]
    for c in cols:
        body[c] = body[c].map(parse_value)
    body["name"] = body["name"].str.replace("　", " ").str.strip()
    body["category"] = body["group"].map(GROUPS)
    body["origin"] = np.where(body["group"].isin(PLANT), "植物性", "動物性")
    thr = np.where(body["group"].isin(EGG_DAIRY), min_protein_egg_dairy, min_protein)
    body = body[body["PROT-"] >= thr]
    for pat in EXCLUDE_NAME_PATTERNS:
        body = body[~body["name"].str.contains(pat)]
    return body[["code", "group", "category", "origin", "name", *cols]].reset_index(drop=True)


def compute_C(foods: pd.DataFrame, dri: pd.DataFrame, nutrients: list[str]) -> pd.DataFrame:
    r = dri.set_index("id")
    Rmp, Rfp = r.loc["PROT-", "male"], r.loc["PROT-", "female"]
    out = {}
    for a in nutrients:
        ratio = foods[a] / foods["PROT-"]
        out[a] = 0.5 * (ratio * Rmp / r.loc[a, "male"] + ratio * Rfp / r.loc[a, "female"])
    return pd.DataFrame(out)


def to_space(C: pd.DataFrame, space: str, floor: float) -> np.ndarray:
    if space == "raw":
        return C.to_numpy()
    return np.log2(np.maximum(C.to_numpy(), floor))


def choose_k(X: np.ndarray, kmin: int, kmax: int, seed: int) -> pd.DataFrame:
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["log", "raw"], default="log")
    ap.add_argument("--floor", type=float, default=1 / 16, help="log 空間で C=0 を置き換える下限")
    ap.add_argument("--min-protein", type=float, default=10.0, help="対象とする 100 g あたりたんぱく質の下限 [g]")
    ap.add_argument("--min-protein-egg-dairy", type=float, default=3.0,
                    help="卵類・乳類のたんぱく質下限 [g]（牛乳 3.3 g, ヨーグルト 3.6 g を含めるため低めに設定）")
    ap.add_argument("--groups", default="04,05,10,11,12,13")
    ap.add_argument("--with-biotin", action="store_true")
    ap.add_argument("--cap", type=float, default=None,
                    help="C を min(cap, C) で頭打ちにする（例: 1.0 で過剰分を評価しない）")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_out = f"results/{args.space}" + (f"_cap{args.cap:g}" if args.cap is not None else "")
    out = ROOT / (args.out or default_out)
    out.mkdir(parents=True, exist_ok=True)

    nut = dict(NUTRIENTS, **(OPTIONAL if args.with_biotin else {}))
    dri = pd.read_csv(DRI_CSV)
    foods = load_foods(args.min_protein, args.min_protein_egg_dairy, set(args.groups.split(",")))
    n_all = len(foods)
    foods = foods.dropna(subset=list(nut)).reset_index(drop=True)
    print(f"対象食品: {len(foods)} 件 (たんぱく質 >= {args.min_protein} g, 卵・乳は >= {args.min_protein_egg_dairy} g, 欠測除外 {n_all - len(foods)} 件)")

    C = compute_C(foods, dri, list(nut))
    if args.cap is not None:
        C = C.clip(upper=args.cap)
    X = to_space(C, args.space, args.floor)
    ideal = to_space(pd.DataFrame([np.ones(len(nut))], columns=list(nut)), args.space, args.floor)[0]

    # ---- k の選択 ----
    ksel = choose_k(X, 2, args.kmax, args.seed)
    ksel.to_csv(out / "k_selection.csv", index=False)
    k = args.k or int(ksel.loc[ksel["silhouette"].idxmax(), "k"])
    print(ksel.round(3).to_string(index=False))
    print(f"採用 k = {k}")

    km = KMeans(n_clusters=k, init="k-means++", n_init=100, random_state=args.seed).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_

    # 理想点 (全成分1) からの距離でクラスタ番号を並べ替える（0 が最も理想に近い）
    d_center = np.linalg.norm(centers - ideal, axis=1)
    order = np.argsort(d_center)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    centers, d_center = centers[order], d_center[order]

    # ---- 食品ごとの結果 ----
    res = foods[["code", "category", "origin", "name", "PROT-"]].copy()
    for a, lab in nut.items():
        res[f"C_{lab}"] = C[a]
    res["cluster"] = labels
    res["dist_to_ideal"] = np.linalg.norm(X - ideal, axis=1)
    # 不足だけを見る指標: log2 C の負の部分の RMS（0 なら全成分が推奨比を満たす）
    logC = np.log2(np.maximum(C.to_numpy(), args.floor))
    res["shortfall_rms_log2"] = np.sqrt((np.minimum(logC, 0) ** 2).mean(axis=1))
    res["n_satisfied"] = (C.to_numpy() >= 1).sum(axis=1)
    res = res.sort_values(["cluster", "dist_to_ideal"]).reset_index(drop=True)
    res.to_csv(out / "foods_clustered.csv", index=False, float_format="%.4g")

    # ---- クラスタ要約 ----
    center_C = 2 ** centers if args.space == "log" else centers
    summ = pd.DataFrame(center_C, columns=[f"C_{v}" for v in nut.values()])
    summ.insert(0, "cluster", range(k))
    summ.insert(1, "n", [int((labels == c).sum()) for c in range(k)])
    summ.insert(2, "center_dist_to_ideal", d_center)
    comp = res.groupby("cluster")["category"].value_counts().unstack(fill_value=0)
    summ = summ.join(comp, on="cluster")
    summ["examples"] = [
        " / ".join(res[res.cluster == c].nsmallest(5, "dist_to_ideal")["name"].str.slice(0, 18))
        for c in range(k)
    ]
    summ.to_csv(out / "cluster_summary.csv", index=False, float_format="%.4g")
    print(summ.drop(columns="examples").round(2).to_string(index=False))

    plot_all(out, ksel, k, X, labels, centers, ideal, nut, foods["name"].to_numpy(), args.space)


def short_name(name: str) -> str:
    """図のラベル用に食品名を短くする（＜畜肉類＞ などの分類見出しと括弧を落とす）。"""
    import re
    name = re.sub(r"＜[^＞]*＞", "", name)
    name = re.sub(r"[［］（）]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def plot_all(out, ksel, k, X, labels, centers, ideal, nut, names, space):
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

    # k 選択
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ksel.k, ksel.inertia, "o-", color="#4a6fa5", label="inertia (SSE)")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia")
    ax2 = ax1.twinx()
    ax2.plot(ksel.k, ksel.silhouette, "s--", color="#c0504d", label="silhouette")
    ax2.set_ylabel("silhouette")
    ax1.axvline(k, color="gray", lw=0.8, ls=":")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    ax1.set_title("k の選択（エルボー / シルエット）")
    fig.tight_layout(); fig.savefig(out / "k_selection.png", dpi=150); plt.close(fig)

    # クラスタ中心ヒートマップ（log2 C, 0 = 理想）
    cen_log = centers if space == "log" else np.log2(np.maximum(centers, 1 / 16))
    fig, ax = plt.subplots(figsize=(1.0 * len(nut) + 3, 0.55 * k + 1.8))
    lim = max(3, np.abs(cen_log).max())
    im = ax.imshow(cen_log, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(nut)), list(nut.values()), rotation=30, ha="right")
    counts = np.bincount(labels, minlength=k)
    ax.set_yticks(range(k), [f"C{c} (n={counts[c]})" for c in range(k)])
    for i in range(k):
        for j in range(len(nut)):
            ax.text(j, i, f"{2 ** cen_log[i, j]:.2g}", ha="center", va="center", fontsize=8,
                    color="white" if abs(cen_log[i, j]) > lim * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="log2 C（0 = 推奨比どおり）")
    ax.set_title("クラスタ中心の C 値（セル内は C、青 = 過剰 / 赤 = 不足）")
    fig.tight_layout(); fig.savefig(out / "cluster_centers.png", dpi=150); plt.close(fig)

    # PCA 散布図 + 理想点
    pca = PCA(n_components=2).fit(X)
    P, Pc, Pi = pca.transform(X), pca.transform(centers), pca.transform(ideal[None])[0]
    # 食品名を全点に小さく載せるため大きめのキャンバスにする（PDF は拡大して読める）
    fig, ax = plt.subplots(figsize=(24, 18))
    for c in range(k):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.6, color=cmap(c), label=f"C{c}")
    for (x, y), nm, c in zip(P, names, labels):
        ax.annotate(short_name(nm), (x, y), xytext=(1.5, 1.5), textcoords="offset points",
                    fontsize=2.8, color=tuple(0.6 * v for v in cmap(c)[:3]))  # 淡色でも読めるよう暗くする
    ax.scatter(Pc[:, 0], Pc[:, 1], s=160, marker="X", color="black", edgecolor="white")
    for c in range(k):
        ax.annotate(f"C{c}", Pc[c], xytext=(5, 5), textcoords="offset points", fontsize=14, weight="bold")
    ax.scatter(*Pi, s=300, marker="*", color="gold", edgecolor="black", zorder=5, label="理想 (全成分 1)")
    # 主成分の負荷ベクトル
    scale = np.abs(P).max() * 0.45
    for j, lab in enumerate(nut.values()):
        v = pca.components_[:, j] * scale
        ax.arrow(Pi[0], Pi[1], v[0], v[1], color="gray", alpha=0.6, head_width=0.08)
        ax.text(Pi[0] + v[0] * 1.08, Pi[1] + v[1] * 1.08, lab, color="dimgray", fontsize=12)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title(f"k-means++ (k={k}) の PCA 射影（{space} 空間）")
    ax.legend(markerscale=2, fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(out / "pca_clusters.png", dpi=200)
    fig.savefig(out / "pca_clusters.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
