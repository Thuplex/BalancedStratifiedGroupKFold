"""
Stratified Group K-Fold com balanceamento de tamanho.

Contexto: recortes (vocalizações) extraídos de sonogramas de morcegos.
Restrições atendidas:
  R1) Todos os recortes de um mesmo sonograma caem SEMPRE no mesmo fold
      (grupo indivisível -> elimina vazamento por correlação temporal/ruído
      de fundo/canal de gravação).
  R2) A distribuição de classes de cada fold aproxima a distribuição global
      (estratificação), tratando o sonograma como um VETOR de contagens por
      classe (suporta sonograma multi-espécie).
  R3) O número de imagens por fold fica o mais próximo possível de N/K.
  R4) Um fold é separado como TESTE FIXO (holdout nunca usado em treino ou
      validação em nenhuma rodada). Os K-1 folds restantes revezam entre si
      como treino/validação. Com K=10 isso gera 9 rodadas 80/10/10 (8 folds
      treino, 1 val, sempre o mesmo fold de teste).

Autor: gerado para pipeline de bioacústica.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# 1. Leitura a partir da estrutura de pastas em disco
# --------------------------------------------------------------------------- #
def build_dataframe_from_folders(
    data_dir: str | Path,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> pd.DataFrame:
    """
    Lê recortes a partir da estrutura em disco:

        data_dir/<classe>/<sonograma_id>/<recorte>.png

    Cada subpasta de 1º nível é uma classe (espécie) e cada subpasta de 2º
    nível é um sonograma (grupo indivisível, R1). Retorna 1 linha por
    recorte, com 'arquivo' (caminho completo), 'sonograma_id' e 'classe' —
    o mesmo formato esperado por `build_group_table`/`fit`.
    """
    data_dir = Path(data_dir)
    linhas = []
    for classe_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for sonograma_dir in sorted(p for p in classe_dir.iterdir() if p.is_dir()):
            for img in sorted(sonograma_dir.iterdir()):
                if img.suffix.lower() in extensions:
                    linhas.append(
                        {
                            "arquivo": str(img),
                            "sonograma_id": sonograma_dir.name,
                            "classe": classe_dir.name,
                        }
                    )
    if not linhas:
        raise ValueError(f"Nenhuma imagem encontrada em {data_dir}")
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# 2. Agregação por grupo (sonograma)
# --------------------------------------------------------------------------- #
@dataclass
class GroupTable:
    """Representação agregada dos grupos."""
    group_ids: np.ndarray      # (G,)  identificador do sonograma
    class_matrix: np.ndarray   # (G, C) contagem de recortes por classe
    sizes: np.ndarray          # (G,)  total de recortes do sonograma
    classes: np.ndarray        # (C,)  rótulos das colunas


def build_group_table(
    df: pd.DataFrame,
    group_col: str = "sonograma_id",
    label_col: str = "classe",
) -> GroupTable:
    """Converte a tabela de recortes (1 linha = 1 imagem) em tabela por grupo."""
    if df[group_col].isna().any():
        raise ValueError(f"Existem linhas com {group_col} nulo.")
    if df[label_col].isna().any():
        raise ValueError(f"Existem linhas com {label_col} nulo.")

    pivot = pd.crosstab(df[group_col], df[label_col])
    return GroupTable(
        group_ids=pivot.index.to_numpy(),
        class_matrix=pivot.to_numpy(dtype=np.float64),
        sizes=pivot.to_numpy().sum(axis=1).astype(np.float64),
        classes=pivot.columns.to_numpy(),
    )


# --------------------------------------------------------------------------- #
# 3. Função de custo
# --------------------------------------------------------------------------- #
def _cost(
    fold_class: np.ndarray,   # (K, C)
    fold_size: np.ndarray,    # (K,)
    target_class: np.ndarray, # (C,)
    target_size: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Desvio quadrático NORMALIZADO em relação ao fold ideal.

    A normalização por `target` é essencial: sem ela, uma classe majoritária
    com 10.000 imagens domina o custo e as classes raras ficam mal
    distribuídas. Com ela, errar 5 imagens numa classe de 50 pesa o mesmo
    que errar 1.000 numa classe de 10.000.

    alpha -> peso da estratificação (R2)
    beta  -> peso do balanceamento de tamanho (R3)
    """
    dc = (fold_class - target_class) / np.maximum(target_class, 1.0)
    ds = (fold_size - target_size) / max(target_size, 1.0)
    return alpha * float((dc ** 2).sum()) + beta * float((ds ** 2).sum())


# --------------------------------------------------------------------------- #
# 4. Atribuição gulosa + refinamento local
# --------------------------------------------------------------------------- #
def assign_groups_to_folds(
    gt: GroupTable,
    k: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int = 42,
    n_refine: int = 20_000,
) -> np.ndarray:
    """
    Retorna vetor (G,) com o índice do fold [0..k-1] de cada grupo.

    Etapa A (gulosa): processa os sonogramas do MAIOR para o MENOR e coloca
    cada um no fold que minimiza o custo naquele momento. Ordenar por tamanho
    decrescente é o que garante o balanceamento — os grupos grandes (difíceis
    de acomodar) entram primeiro e os pequenos servem de "ajuste fino".

    Etapa B (refinamento): hill-climbing com movimentos e trocas aleatórias,
    aceitando apenas alterações que reduzem o custo.
    """
    rng = np.random.default_rng(seed)
    g, c = gt.class_matrix.shape

    if k < 2:
        raise ValueError("k deve ser >= 2.")
    if g < k:
        raise ValueError(f"Apenas {g} sonogramas para {k} folds.")

    target_class = gt.class_matrix.sum(axis=0) / k
    target_size = float(gt.sizes.sum()) / k

    # aviso de classe rara
    groups_per_class = (gt.class_matrix > 0).sum(axis=0)
    for cls, n in zip(gt.classes, groups_per_class):
        if n < k:
            print(
                f"[aviso] classe '{cls}' aparece em apenas {n} sonogramas "
                f"(< k={k}): haverá folds sem essa classe."
            )

    # ---- Etapa A: guloso ----
    order = np.lexsort((rng.random(g), -gt.sizes))  # tamanho desc, empate aleatório
    fold_class = np.zeros((k, c), dtype=np.float64)
    fold_size = np.zeros(k, dtype=np.float64)
    assign = np.full(g, -1, dtype=np.int64)

    for gi in order:
        best_f, best_j = -1, np.inf
        for f in range(k):
            fold_class[f] += gt.class_matrix[gi]
            fold_size[f] += gt.sizes[gi]
            j = _cost(fold_class, fold_size, target_class, target_size, alpha, beta)
            fold_class[f] -= gt.class_matrix[gi]
            fold_size[f] -= gt.sizes[gi]
            if j < best_j:
                best_f, best_j = f, j
        assign[gi] = best_f
        fold_class[best_f] += gt.class_matrix[gi]
        fold_size[best_f] += gt.sizes[gi]

    # ---- Etapa B: refinamento ----
    cur = _cost(fold_class, fold_size, target_class, target_size, alpha, beta)

    def apply_move(gi: int, src: int, dst: int, sign: int = 1) -> None:
        fold_class[src] -= sign * gt.class_matrix[gi]
        fold_size[src] -= sign * gt.sizes[gi]
        fold_class[dst] += sign * gt.class_matrix[gi]
        fold_size[dst] += sign * gt.sizes[gi]

    for _ in range(n_refine):
        if rng.random() < 0.5:
            # movimento: realoca 1 grupo
            gi = int(rng.integers(g))
            src = int(assign[gi])
            dst = int(rng.integers(k))
            if src == dst:
                continue
            apply_move(gi, src, dst)
            new = _cost(fold_class, fold_size, target_class, target_size, alpha, beta)
            if new < cur - 1e-12:
                assign[gi], cur = dst, new
            else:
                apply_move(gi, src, dst, sign=-1)
        else:
            # troca: permuta 2 grupos de folds diferentes
            a, b = rng.integers(g, size=2)
            fa, fb = int(assign[a]), int(assign[b])
            if fa == fb:
                continue
            apply_move(int(a), fa, fb)
            apply_move(int(b), fb, fa)
            new = _cost(fold_class, fold_size, target_class, target_size, alpha, beta)
            if new < cur - 1e-12:
                assign[a], assign[b], cur = fb, fa, new
            else:
                apply_move(int(a), fa, fb, sign=-1)
                apply_move(int(b), fb, fa, sign=-1)

    return assign


# --------------------------------------------------------------------------- #
# 5. API principal
# --------------------------------------------------------------------------- #
class BalancedStratifiedGroupKFold:
    """
    Uso:
        splitter = BalancedStratifiedGroupKFold(k=10, test_fold=0, seed=42)
        df = splitter.fit(df, group_col="sonograma_id", label_col="classe")
        # df ganha a coluna 'fold'

        # fold `test_fold` fica FIXO como teste em todas as rodadas; os k-1
        # folds restantes revezam entre treino e validação.
        for tr, va, te in splitter.split(df):   # k-1 rodadas 80/10/10
            ...
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        seed: int = 42,
        n_refine: int = 20_000,
        test_fold: int = 0,
    ):
        if not 0 <= test_fold < k:
            raise ValueError(f"test_fold deve estar em [0, {k}).")
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.n_refine = n_refine
        self.test_fold = test_fold
        self.group_col = "sonograma_id"
        self.label_col = "classe"
        self.fold_col = "fold"

    def fit(
        self,
        df: pd.DataFrame,
        group_col: str = "sonograma_id",
        label_col: str = "classe",
        fold_col: str = "fold",
    ) -> pd.DataFrame:
        self.group_col, self.label_col, self.fold_col = group_col, label_col, fold_col
        gt = build_group_table(df, group_col, label_col)
        assign = assign_groups_to_folds(
            gt, self.k, self.alpha, self.beta, self.seed, self.n_refine
        )
        mapping = dict(zip(gt.group_ids, assign))
        out = df.copy()
        out[fold_col] = out[group_col].map(mapping).astype(int)
        return out

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        O fold `self.test_fold` fica FIXO como teste em TODAS as rodadas —
        nunca entra em treino nem em validação, garantindo um holdout
        realmente nunca visto durante o desenvolvimento do modelo.

        Os k-1 folds restantes revezam entre si: a cada rodada um deles vira
        validação e os demais (k-2) formam o treino. Com k=10 isso dá 9
        rodadas 80/10/10, cada uma com o mesmo conjunto de teste.

        Retorna índices posicionais (compatíveis com .iloc).
        """
        f = df[self.fold_col].to_numpy()
        te = np.flatnonzero(f == self.test_fold)
        remaining = [j for j in range(self.k) if j != self.test_fold]

        for va_fold in remaining:
            va = np.flatnonzero(f == va_fold)
            tr = np.flatnonzero((f != self.test_fold) & (f != va_fold))
            yield tr, va, te

    def single_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Um único split 80/10/10 (a primeira rodada), como DataFrames."""
        tr, va, te = next(iter(self.split(df)))
        return df.iloc[tr].copy(), df.iloc[va].copy(), df.iloc[te].copy()


# --------------------------------------------------------------------------- #
# 6. Verificação e relatório
# --------------------------------------------------------------------------- #
def check_no_leakage(
    df: pd.DataFrame,
    group_col: str = "sonograma_id",
    fold_col: str = "fold",
) -> None:
    """Levanta AssertionError se algum sonograma aparecer em mais de um fold."""
    n = df.groupby(group_col)[fold_col].nunique()
    bad = n[n > 1]
    assert bad.empty, f"VAZAMENTO: sonogramas em múltiplos folds -> {list(bad.index)}"
    print(f"[ok] {len(n)} sonogramas, cada um em exatamente 1 fold.")


def fold_report(
    df: pd.DataFrame,
    group_col: str = "sonograma_id",
    label_col: str = "classe",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Tabela: imagens, sonogramas e % de cada classe por fold."""
    counts = pd.crosstab(df[fold_col], df[label_col])
    rep = pd.DataFrame(index=counts.index)
    rep["n_imagens"] = counts.sum(axis=1)
    rep["n_sonogramas"] = df.groupby(fold_col)[group_col].nunique()
    rep["desvio_%"] = (
        100 * (rep["n_imagens"] - rep["n_imagens"].mean()) / rep["n_imagens"].mean()
    ).round(2)
    global_pct = df[label_col].value_counts(normalize=True)
    for cls in counts.columns:
        rep[f"{cls}_%"] = (100 * counts[cls] / rep["n_imagens"]).round(2)
    rep.attrs["global_%"] = (100 * global_pct).round(2).to_dict()
    return rep


def print_fold_summary(
    df: pd.DataFrame,
    label_col: str = "classe",
    fold_col: str = "fold",
) -> None:
    """
    Imprime, por pasta (fold): quantidade e % de imagens em relação ao
    total, e quantidade e % de cada espécie dentro daquela pasta.
    """
    total = len(df)
    counts = pd.crosstab(df[fold_col], df[label_col])

    print("\n--- Imagens por pasta ---")
    for fold in sorted(counts.index):
        n_fold = int(counts.loc[fold].sum())
        pct_fold = 100 * n_fold / total
        print(f"\nPasta {fold}: {n_fold} imagens ({pct_fold:.2f}% do total)")
        for cls in counts.columns:
            n_cls = int(counts.loc[fold, cls])
            pct_cls = 100 * n_cls / n_fold if n_fold else 0.0
            print(f"    {cls:<30s} {n_cls:6d} imagens ({pct_cls:5.2f}% da pasta)")


# --------------------------------------------------------------------------- #
# 7. Materialização em disco
# --------------------------------------------------------------------------- #
def _link_or_copy(src: Path, dst: Path, link: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_fold_manifest(
    df: pd.DataFrame,
    out_root: str | Path,
    group_col: str = "sonograma_id",
    label_col: str = "classe",
    fold_col: str = "fold",
    filename: str = "manifest.json",
) -> Path:
    """
    Grava em `out_root/filename` um relatório JSON informando em qual pasta
    (fold) cada sonograma foi colocado:

        {
          "<sonograma_id>": {
            "pasta": "fold_0",
            "fold": 0,
            "classe": "...",
            "n_imagens": 23
          },
          ...
        }

    Retorna o caminho do arquivo gravado.
    """
    out_root = Path(out_root)
    k = int(df[fold_col].max()) + 1
    width = len(str(k - 1))

    grouped = df.groupby(group_col).agg(
        classe=(label_col, "first"),
        fold=(fold_col, "first"),
        n_imagens=(fold_col, "size"),
    )

    manifest = {
        str(sonograma): {
            "pasta": f"fold_{int(row.fold):0{width}d}",
            "fold": int(row.fold),
            "classe": row.classe,
            "n_imagens": int(row.n_imagens),
        }
        for sonograma, row in grouped.iterrows()
    }

    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / filename
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def materialize_folds(
    df: pd.DataFrame,
    data_dir: str | Path,
    out_dirname: str = "folds",
    file_col: str = "arquivo",
    group_col: str = "sonograma_id",
    label_col: str = "classe",
    fold_col: str = "fold",
    link: bool = True,
    flatten: bool = False,
) -> Path:
    """
    Materializa a PARTIÇÃO EM FOLDS (a divisão em si, antes de qualquer
    escolha de treino/val/teste) dentro de `data_dir/out_dirname/`:

        data_dir/out_dirname/fold_00/<classe>/<sonograma_id>/<recorte>.png
        ...
        data_dir/out_dirname/fold_09/<classe>/<sonograma_id>/<recorte>.png

    Cada sonograma inteiro cai em uma única pasta de fold (R1), preservando
    a distribuição de classes (R2) e tamanho (R3) já garantidas por `fit`.

    Por padrão cria links simbólicos (link=True), evitando duplicar os
    arquivos de imagem em disco; use link=False para copiar de fato.
    Reexecuções são seguras/idempotentes (entradas já existentes são
    puladas).

    `flatten=True` remove o nível `<sonograma_id>` do caminho de destino
    (fica `.../<classe>/<recorte>.png`), para compatibilidade com
    consumidores que esperam os arquivos direto na pasta da classe (ex.:
    `src/utils/io.py: gather_paths` deste projeto). Só é seguro quando os
    nomes de arquivo já são únicos dentro da classe.

    Também grava `out_dirname/manifest.json`, informando em qual pasta cada
    sonograma foi colocado (ver `write_fold_manifest`).

    Retorna o caminho da pasta raiz criada.
    """
    data_dir = Path(data_dir)
    out_root = data_dir / out_dirname
    k = int(df[fold_col].max()) + 1
    width = len(str(k - 1))

    for row in df.itertuples(index=False):
        src = Path(getattr(row, file_col))
        fold = getattr(row, fold_col)
        base = out_root / f"fold_{int(fold):0{width}d}" / getattr(row, label_col)
        dst = base / src.name if flatten else base / getattr(row, group_col) / src.name
        _link_or_copy(src, dst, link)

    write_fold_manifest(df, out_root, group_col, label_col, fold_col)

    return out_root


def materialize_split(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    data_dir: str | Path,
    out_dirname: str = "split",
    file_col: str = "arquivo",
    group_col: str = "sonograma_id",
    label_col: str = "classe",
    link: bool = True,
    flatten: bool = False,
) -> Path:
    """
    Materializa UM split 80/10/10 (train/val/test) dentro de
    `data_dir/out_dirname/`, no formato esperado por bibliotecas do tipo
    ImageFolder:

        data_dir/out_dirname/train/<classe>/<sonograma_id>/<recorte>.png
        data_dir/out_dirname/val/<classe>/<sonograma_id>/<recorte>.png
        data_dir/out_dirname/test/<classe>/<sonograma_id>/<recorte>.png

    Recebe os três DataFrames devolvidos por `BalancedStratifiedGroupKFold`
    (por exemplo os de `.single_split(df)`, ou qualquer rodada de
    `.split(df)` já indexada com `.iloc`). O conjunto de teste nunca se
    mistura com treino/validação (R4).

    Por padrão cria links simbólicos (link=True); use link=False para
    copiar de fato. Reexecuções são seguras/idempotentes.

    `flatten=True` remove o nível `<sonograma_id>` do caminho de destino
    (fica `.../<classe>/<recorte>.png`) — ver nota em `materialize_folds`.
    """
    data_dir = Path(data_dir)
    out_root = data_dir / out_dirname

    for nome, parte in (("train", tr), ("val", va), ("test", te)):
        for row in parte.itertuples(index=False):
            src = Path(getattr(row, file_col))
            base = out_root / nome / getattr(row, label_col)
            dst = base / src.name if flatten else base / getattr(row, group_col) / src.name
            _link_or_copy(src, dst, link)

    return out_root


# --------------------------------------------------------------------------- #
# 8. Registro de contagens por classe/fold (auditoria)
# --------------------------------------------------------------------------- #
def write_fold_counts(
    df: pd.DataFrame,
    out_root: str | Path,
    label_col: str = "classe",
    group_col: str = "sonograma_id",
    fold_col: str = "fold",
    filename: str = "fold_counts.csv",
) -> Path:
    """Grava `out_root/filename`: nº de imagens e de sonogramas por classe e por fold."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / filename
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "classe", "n_imagens", "n_sonogramas"])
        for (fold, cls), grupo in df.groupby([fold_col, label_col]):
            writer.writerow([int(fold), cls, len(grupo), grupo[group_col].nunique()])
    return dest


# --------------------------------------------------------------------------- #
# 9. CLI orientada a YAML (integração com resize_images.py/make_splits.py)
# --------------------------------------------------------------------------- #
def load_cv_split_config(path: str | Path) -> dict:
    """Lê a seção `cv_split` de um preprocession_configs*.yaml / cv_split_*.yaml."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cv = data["cv_split"]
    return {
        "src_root": cv["src_root"],
        "out_root": cv["out_root"],
        "k": cv.get("k", 10),
        "test_fold": cv.get("test_fold", 0),
        "alpha": cv.get("alpha", 1.0),
        "beta": cv.get("beta", 1.0),
        "seed": cv.get("seed", 42),
        "n_refine": cv.get("n_refine", 20_000),
        "link": cv.get("link", True),
        "flatten_output": cv.get("flatten_output", True),
    }


def run_from_config(config_path: str | Path) -> Path:
    """
    Lê `src_root` (layout data_dir/<classe>/<sonograma_id>/<recorte>.png),
    calcula os `k` folds group-aware e materializa em disco:

      out_root/fold_0 .. fold_<k-1>   -> partição bruta, uma pasta por fold
      out_root/manifest.json          -> em qual fold cada sonograma caiu
      out_root/fold_counts.csv        -> imagens/sonogramas por classe e fold

    Consumo desses folds pela validação cruzada (rotacionando os k-1 folds
    de treino/validação em torno do `test_fold` fixo) ainda é um passo
    futuro — ver README, seção "Split por grupo".

    Retorna `out_root`.
    """
    cfg = load_cv_split_config(config_path)

    df = build_dataframe_from_folders(cfg["src_root"])
    print(f"Lendo {cfg['src_root']}")
    print(f"Total: {len(df)} imagens / {df.sonograma_id.nunique()} sonogramas/gravações\n")

    splitter = BalancedStratifiedGroupKFold(
        k=cfg["k"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        seed=cfg["seed"],
        n_refine=cfg["n_refine"],
        test_fold=cfg["test_fold"],
    )
    df = splitter.fit(df)

    check_no_leakage(df)
    print_fold_summary(df)

    out_root = Path(cfg["out_root"])
    data_dir, out_dirname = out_root.parent, out_root.name

    folds_dir = materialize_folds(
        df, data_dir, out_dirname=out_dirname,
        link=cfg["link"], flatten=cfg["flatten_output"],
    )
    width = len(str(cfg["k"] - 1))
    print(f"\n[ok] {cfg['k']} pastas (fold_{0:0{width}d}..fold_{cfg['k']-1:0{width}d}) materializadas em: {folds_dir}")

    counts_path = write_fold_counts(df, out_root)
    print(f"[ok] Contagens por classe/fold registradas em: {counts_path}")

    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="YAML com a seção `cv_split` (ex.: configs/cv_split_unbalanced.yaml). "
             "Se omitido, roda a demonstração com dados simulados ou a pasta ./data ao lado do script.",
    )
    args = parser.parse_args()

    if args.config:
        run_from_config(args.config)
    else:
        _run_demo()


# --------------------------------------------------------------------------- #
# 10. Demonstração (dados simulados ou pasta ./data ao lado do script)
# --------------------------------------------------------------------------- #
def _run_demo() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    dados_reais = data_dir.is_dir() and any(data_dir.glob("*/*"))

    if dados_reais:
        print(f"Lendo imagens reais de: {data_dir}\n")
        df = build_dataframe_from_folders(data_dir)
    else:
        print("Pasta 'data' ausente/vazia -> usando dados simulados.\n")
        rng = np.random.default_rng(0)

        # Simula 300 sonogramas com nº de vocalizações MUITO variável (1 a 60)
        especies = ["Molossus", "Artibeus", "Myotis", "Eptesicus", "Sturnira"]
        p_esp = [0.35, 0.28, 0.20, 0.12, 0.05]  # dataset desbalanceado
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

    splitter = BalancedStratifiedGroupKFold(k=10, alpha=1.0, beta=2.0, test_fold=0)
    df = splitter.fit(df)

    check_no_leakage(df)
    print_fold_summary(df)

    rep = fold_report(df)
    print("\n--- Distribuição por fold (tabela) ---")
    print(rep.to_string())
    print("\nDistribuição global (%):", rep.attrs["global_%"])

    print(f"\n--- Splits 80/10/10 (teste fixo = fold {splitter.test_fold}) ---")
    for i, (tr, va, te) in enumerate(splitter.split(df)):
        tot = len(tr) + len(va) + len(te)
        print(
            f"rodada {i}: treino {len(tr):5d} ({100*len(tr)/tot:5.2f}%) | "
            f"val {len(va):4d} ({100*len(va)/tot:5.2f}%) | "
            f"teste {len(te):4d} ({100*len(te)/tot:5.2f}%)"
        )

    if dados_reais:
        folds_dir = materialize_folds(df, data_dir)
        print(f"\n[ok] Partição em folds materializada em: {folds_dir}")

        tr_df, va_df, te_df = splitter.single_split(df)
        split_dir = materialize_split(tr_df, va_df, te_df, data_dir)
        print(f"[ok] Split train/val/test materializado em: {split_dir}")


if __name__ == "__main__":
    main()