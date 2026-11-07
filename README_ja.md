# torch-ssbroyden — PyTorch 用の self-scaled Broyden 準ニュートン法

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23998783.svg)](https://doi.org/10.5281/zenodo.23998783)

`torch-ssbroyden` は、self-scaled Broyden（SSBroyden）法の省メモリ版と密行列版を、`torch.optim.LBFGS` と同じインターフェースの `torch.optim.Optimizer` サブクラスとして提供します。点を取り直しながら学習するミニバッチ学習用のヘルパーも含みます。PINN 向けに作りましたが、滑らかな損失関数なら何にでも使えます。

(English: [README.md](README.md))

| クラス | 中身 |
|---|---|
| `torch_ssbroyden.LSSBroyden` | 省メモリ版 self-scaled Broyden（O(mn) メモリ）。`update="bfgs"` で L-BFGS としても使える |
| `torch_ssbroyden.SSBroyden` | 密行列版 SSBroyden（n×n、小規模モデル向け。旧名 `DenseSSBroyden` も使える） |
| `torch_ssbroyden.SegmentTrainer` | ミニバッチ学習（点を取り直しながら学習する）用のヘルパー。区間固定法で、区間の長さと点数を自動で決める |

SSBroyden の更新式は Kiyani et al., CMAME 446 (2025) 118308 の Sec. 1.3 に従っています。省メモリ版は、各組の (τ, φ) を保存し、H = cγI + PAPᵀ を 2m 次元のグラム空間で組み直します。

## インストール

```bash
pip install "torch-ssbroyden @ git+https://github.com/Shiratori-Lab/torch-ssbroyden.git"          # 最新版
pip install "torch-ssbroyden @ git+https://github.com/Shiratori-Lab/torch-ssbroyden.git@v0.4.1"   # 版を指定する場合

# clone して使う場合
git clone https://github.com/Shiratori-Lab/torch-ssbroyden.git
pip install -e torch-ssbroyden              # 本体
pip install -e "torch-ssbroyden[fast]"      # scipy も入れる（L-SSBroyden の行列の組み直しが速くなる）
```

Python 3.10 以上、PyTorch 2.1 以上が必要です。import 名は `torch_ssbroyden` です。

## 使い方

### L-SSBroyden / SSBroyden（`torch.optim.LBFGS` と同じインターフェース）

```python
import torch
from torch_ssbroyden import LSSBroyden
torch.set_default_dtype(torch.float64)              # 高精度を狙うなら fp64

opt = LSSBroyden(model.parameters(), history_size=100, max_iter=20)
def closure():
    opt.zero_grad()
    loss = loss_fn(); loss.backward(); return loss
for epoch in range(N):
    opt.step(closure)                                # 1 回の step で最大 max_iter 反復
```

| 引数 | 既定値 | 説明 |
|---|---|---|
| `history_size` | 20 | 保存する (s, y) の組の数 m |
| `update` | `"ssbroyden"` | `"ssbroyden"` / `"ssbfgs"` / `"bfgs"` |
| `h0` | `"latest"` | `"latest"`：γ = sᵀy/yᵀy（L-BFGS 流）。`"identity"`：γ = 1。`"bb1"`：γ = sᵀs/sᵀy。正の数：γ をその値に固定 |
| `curvature_guard` | `"relative"` | 更新を飛ばす基準。`"absolute"` は torch.optim.LBFGS と同じ sᵀy ≤ 1e-10 |
| `line_search_fn` | `"strong_wolfe"` | `None` にすると固定ステップ `lr` |
| `tau_beta` | 0.0 | 実験的オプション（τ の規則の変種）。既定の 0 が論文どおりの規則 |
| その他 | | `lr`, `max_iter`, `max_eval`, `tolerance_grad`, `tolerance_change` は LBFGS と同じ |

### ミニバッチ学習（SegmentTrainer）

コロケーション点を取り直しながら学習するときは、毎反復取り直すのではなく、同じ点を一定期間（区間）使い続け、曲率のメモリは区間をまたいで保持します。区間の長さと点数は、独立した確認用バッチで自動的に決まります。

```python
from torch_ssbroyden import LSSBroyden, SegmentTrainer
def sample(n, generator):          # n 点のバッチ（内部点・境界点の組など、loss_fn が受け取れる形）
    ...
def loss_fn(batch):                # そのバッチでの loss（backward は呼ばない）
    ...
opt = LSSBroyden(model.parameters(), history_size=20)
trainer = SegmentTrainer(opt, loss_fn, sample, n0=4096, n_max=262144)
trainer.run(max_evals=20000)       # max_point_evals（点数×評価回数）でも指定できる
print(trainer.segments[-1], trainer.n)
```

| 引数 | 既定値 | 説明 |
|---|---|---|
| `n0` | （必須） | 開始時の点数。GPU では、1回の評価時間が上がり始める点数より小さくしても速くならない |
| `n_max` | `n0` | 点数の上限（メモリで決める） |
| `m_min` | 200 | 区間の最低の長さ（反復数） |
| `theta` | 0.5 | 汎化ギャップ (f_probe − f_batch)/f_batch がこれを超えたら点数を 2 倍にする |
| `probe_every`, `patience`, `rel_tol` | 10, 3, 0.01 | 確認用バッチの loss を 10 反復ごとに測り、直近 3 回で 1% 以上改善しなければ区間を終える |
| `segment_iters` | `None` | 区間の長さを固定する（自動の判定を使わない） |
| `sampler` | `Uniform(sample)` | 次の区間の点の選び方（下記） |
| `probe` | `True` | `False` で確認用バッチを使わない（`segment_iters` が必要） |

`LSSBroyden` と密行列の `SSBroyden` のどちらでも使えます。
自前のループで使う場合は、バッチを替えた直後に `opt.new_batch()` を呼んでください（古いバッチと新しいバッチをまたぐ曲率の組を作らないようにします）。

使用例：`examples/poisson_minibatch.py`

### 残差に基づく適応的な点の選び方（RAD、RAR-D、RAR）

区間固定法は、点を残差に応じて入れ替える方法にもそのまま使えます。点を替えるのは区間の切れ目だけです。

```python
from torch_ssbroyden.sampling import RAD, RARD
def residual_fn(batch):            # 適応させる成分の、1点ごとの残差
    return {"pde": pde_residual(model, batch["pde"])}
sampler = RAD(sample, residual_fn, k=1.0, c=1.0, keys=["pde"])          # 残差で選び直す
sampler = RARD(sample, residual_fn, n_add=0.1, keys=["pde"])            # 区間ごとに 10% 追加（RAR-D）
sampler = RARD(sample, residual_fn, n_add=0.1, greedy=True, keys=["pde"])  # 残差の大きい順に追加（RAR）
trainer = SegmentTrainer(opt, loss_fn, sample, n0=2000, sampler=sampler,
                         segment_iters=200, theta=None, probe=False)
```

- `RAD`：一様な候補点を `factor * n` 個（既定 10 倍）作り、確率 ∝ |r|^k / mean|r|^k + c で n 個を選ぶ（Wu et al. 2023）。
- `RARD`：今の点を残し、同じ密度（k = 2、c = 0）で点を足す。`greedy=True` なら残差の大きい順（Lu et al. 2021）。
- `sampler(n, generator, trainer) -> batch` の形の関数なら何でも渡せます。候補点の残差を評価した点数は `trainer.sampler_point_evals` に記録されます。

## 注意点

- **精度**：PINN で高精度を狙うときは float64 にする。
- **初期行列 `h0`**：L-SSBroyden の `h0` は問題によって最適な選択が変わる。2D Poisson では m=100 で `"identity"` の方が良く、Rosenbrock では m=10 で `"identity"` だと収束が止まった。
- **torch.optim.LBFGS**：sᵀy > 1e-10 という絶対基準で更新を判定するため、loss が 1e-7 程度まで下がると収束が止まる。高精度が必要なら `LSSBroyden(update="bfgs")` を使う。
- **直線探索は外さない**（`line_search_fn="strong_wolfe"`）。歩幅を固定すると、全点学習でも発散することがあり、毎反復バッチを替えるとさらに壊れやすくなる。
- **ミニバッチや適応的な点の選び方を使うとき**は、点を替えるのを区間の切れ目だけにする（`SegmentTrainer`、または自前のループで `opt.new_batch()`）。記憶は引き継ぎ、2 つのバッチをまたぐ曲率の組は作らない。区間の長さは、そのバッチで収束するのに必要な反復数（10²〜10³）にする。我々の試験では、これで固定した点集合と同等の精度になり、毎反復替える場合は 1.5〜3 桁悪かった。
- **直線探索**は PyTorch の strong Wolfe 直線探索の写し（`torch_ssbroyden/_line_search.py`、BSD-3-Clause）を使っているので、PyTorch の非公開関数には依存しない。

## ライセンスと引用

Apache License 2.0（写した直線探索の部分は BSD-3-Clause。ファイル冒頭を参照）。

使った場合は、このソフトウェア（DOI [10.5281/zenodo.23998783](https://doi.org/10.5281/zenodo.23998783)）、付随する arXiv の論文、SSBroyden の論文を引用してください（[CITATION.cff](CITATION.cff)）。

- S. Shiratori, *torch-ssbroyden: self-scaled Broyden quasi-Newton optimizers for PyTorch*, Zenodo, 2026. https://doi.org/10.5281/zenodo.23998783
- S. Shiratori, *L-SSBroyden: limited-memory and mini-batch self-scaled Broyden training of physics-informed neural networks*, arXiv, 2026.
- E. Kiyani et al., *Comput. Methods Appl. Mech. Engrg.* 446 (2025) 118308.

## テスト

```bash
pip install -e ".[test]"
pytest
```
