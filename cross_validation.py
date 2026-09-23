"""
Validação cruzada aninhada (nested) construída sobre folders_frist.

Etapa 1 (externa): usa `BalancedStratifiedGroupKFold` (de folders_frist)
para dividir TODOS os sonogramas em k_outer folds e fixa UM deles como
teste — o mesmo mecanismo de holdout de folders_frist (R4): esse fold
nunca entra em treino/validação.

Etapa 2 (interna): pega só os sonogramas que sobraram (fora do teste) e os
reparticiona DO ZERO, com o mesmo algoritmo de folders_frist (grupo
indivisível R1, estratificação R2, balanceamento de tamanho R3 — os alvos
são recalculados sobre o que sobrou, não sobre o dataset inteiro), em
k_inner folds NOVOS, informados pelo usuário. A validação cruzada roda
sobre esses k_inner folds: a cada rodada um vira validação e os demais
treino, sempre longe do teste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from folders_frist import (
    BalancedStratifiedGroupKFold,
    _link_or_copy,
    assign_groups_to_folds,
    build_dataframe_from_folders,
    build_group_table,
    materialize_folds,
)


# --------------------------------------------------------------------------- #
# 1. API principal
# --------------------------------------------------------------------------- #
class NestedGroupCV:
    """
    Uso:
        nested = NestedGroupCV(k_outer=10, test_fold=0, k_inner=5, seed=42)
        df = nested.fit(df)   # ganha 'fold_externo', 'is_teste', 'fold_cv'

        for tr, va in nested.split(df):   # k_inner rodadas
            ...

        te = nested.test_indices(df)      # holdout fixo, fora do split()
    """

    def __init__(
        self,
        k_outer: int = 10,
        test_fold: int = 0,
        k_inner: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        seed: int = 42,
        n_refine: int = 20_000,
    ):
        self.k_outer = k_outer
        self.test_fold = test_fold
        self.k_inner = k_inner
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.n_refine = n_refine
        self.group_col = "sonograma_id"
        self.label_col = "classe"

    def fit(
        self,
        df: pd.DataFrame,
        group_col: str = "sonograma_id",
        label_col: str = "classe",
    ) -> pd.DataFrame:
        self.group_col, self.label_col = group_col, label_col

        # Etapa 1: separa o fold de teste (regras de folders_frist R1-R4)
        outer = BalancedStratifiedGroupKFold(
            k=self.k_outer,
            alpha=self.alpha,
            beta=self.beta,
            seed=self.seed,
            n_refine=self.n_refine,
            test_fold=self.test_fold,
        )
        out = outer.fit(df, group_col=group_col, label_col=label_col, fold_col="fold_externo")
        out["is_teste"] = out["fold_externo"] == self.test_fold

        # Etapa 2: reparticiona DO ZERO só quem sobrou, em k_inner folds novos
        restante = out.loc[~out["is_teste"]]
        gt = build_group_table(restante, group_col, label_col)
        assign = assign_groups_to_folds(
            gt,
            k=self.k_inner,
            alpha=self.alpha,
            beta=self.beta,
            seed=self.seed,
            n_refine=self.n_refine,
        )
        mapping = dict(zip(gt.group_ids, assign))
        out["fold_cv"] = out[group_col].map(mapping).fillna(-1).astype(int)

        return out

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        k_inner rodadas de validação cruzada sobre o pool que sobrou do
        teste. A cada rodada, um fold interno vira validação e os demais
        (k_inner - 1) formam o treino. Linhas de teste (fold_cv == -1)
        nunca aparecem em treino nem em validação.

        Retorna índices posicionais (compatíveis com .iloc).
        """
        fold_cv = df["fold_cv"].to_numpy()
        for i in range(self.k_inner):
            va = np.flatnonzero(fold_cv == i)
            tr = np.flatnonzero((fold_cv != i) & (fold_cv != -1))
            yield tr, va

    def test_indices(self, df: pd.DataFrame) -> np.ndarray:
        """Índices posicionais do holdout de teste (fixo, fora do split())."""
        return np.flatnonzero(df["is_teste"].to_numpy())


# --------------------------------------------------------------------------- #
# 2. Verificação e relatório
# --------------------------------------------------------------------------- #
def check_no_leakage_nested(df: pd.DataFrame, group_col: str = "sonograma_id") -> None:
    """
    Levanta AssertionError se algum sonograma aparecer em mais de um
    estado (teste, ou mais de um fold_cv).
    """
    estado = np.where(df["is_teste"], "teste", "cv_" + df["fold_cv"].astype(str))
    n = df.assign(_estado=estado).groupby(group_col)["_estado"].nunique()
    bad = n[n > 1]
    assert bad.empty, f"VAZAMENTO: sonogramas em múltiplos estados -> {list(bad.index)}"

    n_teste = df.loc[df["is_teste"], group_col].nunique()
    n_cv = df.loc[~df["is_teste"], group_col].nunique()
    print(
        f"[ok] {len(n)} sonogramas: {n_teste} no teste (fixo), "
        f"{n_cv} na validação cruzada, cada um em exatamente 1 estado."
    )


def print_nested_summary(df: pd.DataFrame, label_col: str = "classe") -> None:
    """Imprime quantidade e % de imagens no teste e em cada pasta da CV interna."""
    total = len(df)
    teste = df.loc[df["is_teste"]]
    n_teste = len(teste)

    fold_externo = int(teste["fold_externo"].iloc[0]) if n_teste else None
    print(f"\n--- Teste (fixo, pasta externa {fold_externo}) ---")
    print(f"{n_teste} imagens ({100 * n_teste / total:.2f}% do total)")
    for cls, n_cls in teste[label_col].value_counts().items():
        pct = 100 * n_cls / n_teste if n_teste else 0.0
        print(f"    {cls:<30s} {n_cls:6d} imagens ({pct:5.2f}% do teste)")

    cv = df.loc[~df["is_teste"]]
    n_cv = len(cv)
    k_inner = int(cv["fold_cv"].max()) + 1
    counts = pd.crosstab(cv["fold_cv"], cv[label_col])

    print(f"\n--- Pastas de validação cruzada (k_inner={k_inner}) ---")
    for fold in sorted(counts.index):
        n_fold = int(counts.loc[fold].sum())
        pct_fold = 100 * n_fold / n_cv if n_cv else 0.0
        print(f"\nPasta cv_{fold}: {n_fold} imagens ({pct_fold:.2f}% da validação cruzada)")
        for cls in counts.columns:
            n_cls = int(counts.loc[fold, cls])
            pct_cls = 100 * n_cls / n_fold if n_fold else 0.0
            print(f"    {cls:<30s} {n_cls:6d} imagens ({pct_cls:5.2f}% da pasta)")


# --------------------------------------------------------------------------- #
# 3. Materialização em disco
# --------------------------------------------------------------------------- #
def materialize_nested(
    df: pd.DataFrame,
    data_dir: str | Path,
    out_dirname: str = "nested_cv",
    file_col: str = "arquivo",
    group_col: str = "sonograma_id",
    label_col: str = "classe",
    link: bool = True,
) -> Path:
    """
    Materializa a divisão aninhada dentro de `data_dir/out_dirname/`:

        data_dir/out_dirname/teste/<classe>/<sonograma_id>/<recorte>.png
        data_dir/out_dirname/cv/fold_0/<classe>/<sonograma_id>/<recorte>.png
        ...
        data_dir/out_dirname/cv/fold_{k_inner-1}/<classe>/<sonograma_id>/<recorte>.png
        data_dir/out_dirname/cv/manifest.json

    O teste fica em pasta própria e nunca é tocado pela validação cruzada
    (R4). Por padrão cria links simbólicos (link=True); use link=False
    para copiar de fato. Reexecuções são seguras/idempotentes.
    """
    data_dir = Path(data_dir)
    out_root = data_dir / out_dirname

    teste = df.loc[df["is_teste"]]
    for row in teste.itertuples(index=False):
        src = Path(getattr(row, file_col))
        dst = out_root / "teste" / getattr(row, label_col) / getattr(row, group_col) / src.name
        _link_or_copy(src, dst, link)

    cv = df.loc[~df["is_teste"]]
    materialize_folds(
        cv,
        out_root,
        out_dirname="cv",
        file_col=file_col,
        group_col=group_col,
        label_col=label_col,
        fold_col="fold_cv",
        link=link,
    )

    return out_root


# --------------------------------------------------------------------------- #
# 4. Demonstração
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    dados_reais = data_dir.is_dir() and any(data_dir.glob("*/*"))

    if dados_reais:
        print(f"Lendo imagens reais de: {data_dir}\n")
        df = build_dataframe_from_folders(data_dir)
    else:
        print("Pasta 'data' ausente/vazia -> usando dados simulados.\n")
        rng = np.random.default_rng(0)
        especies = ["Molossus", "Artibeus", "Myotis", "Eptesicus", "Sturnira"]
        p_esp = [0.35, 0.28, 0.20, 0.12, 0.05]
        linhas = []
        for s in range(300):
            n_voc = int(rng.integers(1, 61))
            esp = rng.choice(especies, p=p_esp)
            for v in range(n_voc):
                linhas.append(
                    {
                        "arquivo": f"son{s:03d}_voc{v:03d}.png",
                        "sonograma_id": f"son{s:03d}",
                        "classe": esp,
                    }
                )
        df = pd.DataFrame(linhas)

    print(f"Total: {len(df)} imagens / {df.sonograma_id.nunique()} sonogramas\n")

    k_outer = int(input("Quantas pastas externas (k_outer) para escolher o teste? [10] ") or 10)
    test_fold = int(input(f"Qual pasta externa [0..{k_outer - 1}] vira teste? [0] ") or 0)
    k_inner = int(input("Em quantas pastas (k_inner) dividir o restante para a validação cruzada? [5] ") or 5)

    nested = NestedGroupCV(k_outer=k_outer, test_fold=test_fold, k_inner=k_inner)
    df = nested.fit(df)

    check_no_leakage_nested(df)
    print_nested_summary(df)

    print(f"\n--- Validação cruzada interna (k_inner={k_inner}) ---")
    for i, (tr, va) in enumerate(nested.split(df)):
        tot = len(tr) + len(va)
        print(f"rodada {i}: treino {len(tr):5d} ({100 * len(tr) / tot:5.2f}%) | val {len(va):4d} ({100 * len(va) / tot:5.2f}%)")

    te = nested.test_indices(df)
    print(f"\nTeste (fixo, fora de toda a validação cruzada): {len(te)} imagens")

    if dados_reais:
        out_root = materialize_nested(df, data_dir)
        print(f"\n[ok] Divisão aninhada materializada em: {out_root}")
