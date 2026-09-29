"""
Divisão aninhada K0 x K1 centrada em GRUPO (sonograma), sem nunca passar
por um DataFrame com 1 linha por imagem.

Processo (ver conversa):
  1. K0 é calculado UMA VEZ (fixo para todas as rodadas): divide todos os
     sonogramas em K0 pastas.
  2. Para CADA uma das K0 rodadas (uma por pasta de K0):
       - a pasta da rodada vira o teste fixo daquela rodada;
       - as K0-1 pastas restantes são reagrupadas num único pool;
       - esse pool é reparticionado DO ZERO em K1 pastas novas (a partição
         K1 é recalculada a cada rodada, porque o pool muda a cada vez).
  3. Dentro de uma rodada, a validação cruzada com K1 pastas reaproveita a
     MESMA atribuição de fold_cv nas suas K1 combinações de treino/val —
     nunca recalcula o agrupamento, só re-rotula qual pasta é "validação"
     naquela combinação.

Isso garante que um sonograma nunca tenha suas imagens espalhadas entre
pastas diferentes, nem em K0 nem em K1, em nenhum momento do processo.

O NÚCLEO do algoritmo (GroupTable, _cost, assign_groups_to_folds) mora
na seção 0 deste arquivo; o resto é só o "entra e sai" ao redor dele. É a
implementação atual do projeto para a validação cruzada aninhada; os
antigos `cross_validation.py` (rodada única) e `folders_frist.py` (split
único, materializado em pastas) foram removidos por ficarem redundantes
com `K0K1ManifestCV`.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# 0. Núcleo: tabela por grupo, função de custo e atribuição grupo -> pasta
# --------------------------------------------------------------------------- #
@dataclass
class GroupTable:
    """Representação agregada dos grupos."""
    group_ids: np.ndarray      # (G,)  identificador do sonograma
    class_matrix: np.ndarray   # (G, C) contagem de recortes por classe
    sizes: np.ndarray          # (G,)  total de recortes do sonograma
    classes: np.ndarray        # (C,)  rótulos das colunas


# --------------------------------------------------------------------------- #
# 0b. Função de custo
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
    return float(_fold_cost(fold_class, fold_size, target_class, target_size, alpha, beta).sum())


def _fold_cost(
    fold_class: np.ndarray,   # (..., C)
    fold_size: np.ndarray,    # (...,)
    target_class: np.ndarray, # (C,)
    target_size: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Parcela de `_cost` de cada pasta (somar sobre as pastas dá `_cost`).
    Aceita lotes (...) — o refinamento usa isso pra avaliar de uma vez todos
    os destinos/parceiros possíveis de um grupo, já que mover ou trocar
    grupos só altera o custo das DUAS pastas envolvidas.
    """
    dc = (fold_class - target_class) / np.maximum(target_class, 1.0)
    ds = (fold_size - target_size) / max(target_size, 1.0)
    return alpha * (dc ** 2).sum(axis=-1) + beta * ds ** 2


# --------------------------------------------------------------------------- #
# 0c. Atribuição gulosa + refinamento local
# --------------------------------------------------------------------------- #
def assign_groups_to_folds(
    gt: GroupTable,
    k: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Retorna vetor (G,) com o índice do fold [0..k-1] de cada grupo.

    Etapa A (gulosa): processa os sonogramas do MAIOR para o MENOR e coloca
    cada um no fold que minimiza o custo naquele momento. Ordenar por tamanho
    decrescente é o que garante o balanceamento — os grupos grandes (difíceis
    de acomodar) entram primeiro e os pequenos servem de "ajuste fino".
    Sonogramas de mesmo tamanho mantêm a ordem de `gt` (a de leitura:
    classe, depois id).

    Etapa B (refinamento sistemático): varre, em ordem fixa, todos os
    movimentos (1 grupo -> outra pasta) e todas as trocas (2 grupos de
    pastas diferentes), aceitando só as que reduzem o custo, e repete as
    varreduras até uma inteira não melhorar nada. Termina sempre (o custo
    cai estritamente a cada alteração aceita) e o resultado é um ótimo
    local: nenhum movimento ou troca isolada ainda reduz o custo.

    Totalmente determinístico: sem sorteio em nenhuma etapa, o resultado
    depende só de `gt`, `k`, `alpha` e `beta`.
    """
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
    order = np.argsort(-gt.sizes, kind="stable")  # tamanho desc, empate pela ordem de leitura (classe, id)
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

    # ---- Etapa B: refinamento sistemático ----
    X, S = gt.class_matrix, gt.sizes
    targets = (target_class, target_size, alpha, beta)

    def apply_move(gi: int, src: int, dst: int) -> None:
        fold_class[src] -= X[gi]
        fold_size[src] -= S[gi]
        fold_class[dst] += X[gi]
        fold_size[dst] += S[gi]
        assign[gi] = dst

    melhorou = True
    while melhorou:
        melhorou = False

        # movimento: cada grupo, em ordem, vai pra pasta que mais reduz o
        # custo (empate -> menor índice de pasta, via argmin)
        for gi in range(g):
            src = int(assign[gi])
            atual = _fold_cost(fold_class, fold_size, *targets)                       # (K,)
            src_sem_gi = _fold_cost(fold_class[src] - X[gi], fold_size[src] - S[gi], *targets)
            dst_com_gi = _fold_cost(fold_class + X[gi], fold_size + S[gi], *targets)  # (K,)
            delta = (src_sem_gi - atual[src]) + (dst_com_gi - atual)
            delta[src] = np.inf
            dst = int(np.argmin(delta))
            if delta[dst] < -1e-12:
                apply_move(gi, src, dst)
                melhorou = True

        # troca: cada grupo `a`, em ordem, troca com o parceiro `b` (de outra
        # pasta) que mais reduz o custo (empate -> menor índice de grupo)
        for a in range(g):
            fa = int(assign[a])
            fb = assign                                                               # (G,)
            atual = _fold_cost(fold_class, fold_size, *targets)                       # (K,)
            d_class, d_size = X - X[a], S - S[a]   # o que a pasta de `a` ganha na troca
            fa_novo = _fold_cost(fold_class[fa] + d_class, fold_size[fa] + d_size, *targets)
            fb_novo = _fold_cost(fold_class[fb] - d_class, fold_size[fb] - d_size, *targets)
            delta = (fa_novo - atual[fa]) + (fb_novo - atual[fb])                     # (G,)
            delta[fb == fa] = np.inf
            b = int(np.argmin(delta))
            if delta[b] < -1e-12:
                fb_b = int(assign[b])
                apply_move(a, fa, fb_b)
                apply_move(b, fb_b, fa)
                melhorou = True

    return assign


# --------------------------------------------------------------------------- #
# 1. Leitura DIRETO em nível de grupo (nunca cria 1 linha por imagem)
# --------------------------------------------------------------------------- #
IMG_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


def nome_portavel(p: Path) -> str:
    """
    Nome de `p` em Unicode NFC. macOS pode gravar nomes acentuados em NFD e
    Linux/Windows em NFC — mesmo texto visual, strings diferentes; normalizar
    faz o mesmo nome virar a mesma string (e a mesma ordem) em qualquer SO.
    """
    return unicodedata.normalize("NFC", p.name)


def listar_ordenado(
    pasta: str | Path,
    pastas: bool = False,
    extensions: tuple[str, ...] = IMG_EXTENSIONS,
) -> list[Path]:
    """
    Subpastas (`pastas=True`) ou imagens de `pasta`, em ordem IDÊNTICA em
    qualquer máquina/SO:
      - ordena pelo nome NFC como string (ordem de code point), e não por
        `Path` — `WindowsPath` compara sem diferenciar maiúsculas, `PosixPath`
        diferencia, então `sorted(Path)` muda de ordem entre SOs;
      - ignora nomes ocultos (`.ipynb_checkpoints`, `.DS_Store`, `._x.png` do
        macOS...), que existem numa máquina e não na outra.
    """
    itens = [p for p in Path(pasta).iterdir() if not p.name.startswith(".")]
    if pastas:
        itens = [p for p in itens if p.is_dir()]
    else:
        itens = [p for p in itens if p.is_file() and p.suffix.lower() in extensions]
    return sorted(itens, key=nome_portavel)


def build_group_table_from_folders(
    data_dir: str | Path,
    extensions: tuple[str, ...] = IMG_EXTENSIONS,
) -> GroupTable:
    """
    Varre `data_dir/<classe>/<sonograma_id>/*.png` e monta a GroupTable
    direto — sem nunca criar uma linha por imagem (já nasce agregado).

    Assume 1 classe por sonograma (verdade neste dataset: a classe é a
    pasta-mãe do sonograma), então a contagem por classe do grupo é só
    [0, ..., n, ..., 0], com n = nº de imagens da pasta.

    A ordem dos grupos (que decide os empates de `assign_groups_to_folds`)
    vem de `listar_ordenado`, então é a mesma em qualquer máquina/SO.
    """
    data_dir = Path(data_dir)
    group_ids, classe_por_grupo, sizes = [], [], []

    for classe_dir in listar_ordenado(data_dir, pastas=True):
        for sonograma_dir in listar_ordenado(classe_dir, pastas=True):
            n = len(listar_ordenado(sonograma_dir, extensions=extensions))
            if n == 0:
                continue
            group_ids.append(nome_portavel(sonograma_dir))
            classe_por_grupo.append(nome_portavel(classe_dir))
            sizes.append(n)

    # o manifest é indexado por sonograma_id: um id repetido (em 2 classes,
    # ou 2 nomes que só diferem em maiúscula/acentuação) seria sobrescrito
    # em silêncio
    vistos: dict[str, str] = {}
    for gid, classe in zip(group_ids, classe_por_grupo):
        if gid in vistos:
            raise ValueError(
                f"sonograma_id '{gid}' repetido em '{vistos[gid]}' e '{classe}'."
            )
        vistos[gid] = classe

    classes = np.array(sorted(set(classe_por_grupo)))
    idx_classe = {c: i for i, c in enumerate(classes)}
    class_matrix = np.zeros((len(group_ids), len(classes)), dtype=np.float64)
    for i, (classe, n) in enumerate(zip(classe_por_grupo, sizes)):
        class_matrix[i, idx_classe[classe]] = n

    return GroupTable(
        group_ids=np.array(group_ids),
        class_matrix=class_matrix,
        sizes=np.array(sizes, dtype=np.float64),
        classes=classes,
    )


# --------------------------------------------------------------------------- #
# 2. Manifest (dict) <-> GroupTable
# --------------------------------------------------------------------------- #
def _classe_do_grupo(gt: GroupTable, i: int) -> str:
    return str(gt.classes[int(np.argmax(gt.class_matrix[i]))])


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 3. Divisão aninhada K0 x K1 — K0 fixo, K1 recalculado a cada rodada
# --------------------------------------------------------------------------- #
class K0K1ManifestCV:
    """
    Uso:
        cv = K0K1ManifestCV(k0=10, k1=5)
        manifest = cv.fit_from_folders("data/all_sonogram_folder")
        # manifest["config"]        -> k0, k1, alpha, beta, src_root
        # manifest["sonogramas"][sonograma_id] -> {caminho, classe, n_imagens,
        #                                          rodadas: {"fold_0": {...}, ...}}

        for tr_ids, va_ids in cv.split(manifest, rodada=3):   # k1 combinações
            ...  # tr_ids/va_ids são listas de sonograma_id, não de imagens

        teste_ids = cv.test_groups(manifest, rodada=3)
    """

    def __init__(
        self,
        k0: int = 10,
        k1: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.k0 = k0
        self.k1 = k1
        self.alpha = alpha
        self.beta = beta

    def _rodada_key(self, round_idx: int) -> str:
        width = len(str(self.k0 - 1))
        return f"fold_{round_idx:0{width}d}"

    def fit_from_folders(self, data_dir: str | Path) -> dict:
        gt = build_group_table_from_folders(data_dir)
        return self.fit(gt, src_root=data_dir)

    def fit(self, gt: GroupTable, src_root: str | Path) -> dict:
        src_root = Path(src_root)

        # K0: calculado 1x só, fixo pras k0 rodadas
        outer_assign = assign_groups_to_folds(
            gt, self.k0, self.alpha, self.beta
        )

        sonogramas: dict[str, dict] = {}
        for i, group_id in enumerate(gt.group_ids):
            classe = _classe_do_grupo(gt, i)
            sonogramas[str(group_id)] = {
                # sempre com "/" (as_posix) — o manifest sai idêntico em
                # Windows e Linux, e Path("a/b") funciona nos dois
                "caminho": (src_root / classe / str(group_id)).as_posix(),
                "classe": classe,
                "n_imagens": int(gt.sizes[i]),
                "rodadas": {},
            }

        # K1: reparticionado DO ZERO em cada uma das k0 rodadas — o pool
        # que sobra é diferente a cada rodada, então a partição também é
        for round_idx in range(self.k0):
            is_teste = outer_assign == round_idx
            gt_restante = GroupTable(
                group_ids=gt.group_ids[~is_teste],
                class_matrix=gt.class_matrix[~is_teste],
                sizes=gt.sizes[~is_teste],
                classes=gt.classes,
            )
            inner_assign = assign_groups_to_folds(
                gt_restante, self.k1, self.alpha, self.beta
            )
            fold_cv_por_grupo = dict(zip(gt_restante.group_ids, inner_assign))

            rodada_key = self._rodada_key(round_idx)
            for i, group_id in enumerate(gt.group_ids):
                sonogramas[str(group_id)]["rodadas"][rodada_key] = {
                    "is_teste": bool(is_teste[i]),
                    "fold_cv": int(fold_cv_por_grupo[group_id])
                    if group_id in fold_cv_por_grupo
                    else None,
                }

        return {
            "config": {
                "k0": self.k0,
                "k1": self.k1,
                "alpha": self.alpha,
                "beta": self.beta,
                "src_root": src_root.as_posix(),
            },
            "sonogramas": sonogramas,
        }

    def split(self, manifest: dict, rodada: int) -> Iterator[tuple[list[str], list[str]]]:
        """
        k1 combinações de treino/validação DENTRO da rodada `rodada` de K0
        (a pasta de K0 daquela rodada é o teste fixo — nunca aparece
        aqui). O mesmo fold_cv, calculado 1x para essa rodada, é só
        reetiquetado entre as k1 combinações — nunca recalculado.
        Retorna listas de `sonograma_id` (não de imagens).
        """
        chave = self._rodada_key(rodada)
        cv = {
            gid: info["rodadas"][chave]
            for gid, info in manifest["sonogramas"].items()
            if not info["rodadas"][chave]["is_teste"]
        }
        for i in range(self.k1):
            va = [gid for gid, r in cv.items() if r["fold_cv"] == i]
            tr = [gid for gid, r in cv.items() if r["fold_cv"] != i]
            yield tr, va

    def test_groups(self, manifest: dict, rodada: int) -> list[str]:
        """IDs de sonograma do holdout de teste da rodada `rodada` (fixo, fora de `split()`)."""
        chave = self._rodada_key(rodada)
        return [
            gid
            for gid, info in manifest["sonogramas"].items()
            if info["rodadas"][chave]["is_teste"]
        ]


# --------------------------------------------------------------------------- #
# 3b. Arquivo 2 — índice invertido (pasta -> sonogramas), pra auditoria
# --------------------------------------------------------------------------- #
def derive_audit_index(manifest: dict) -> dict:
    """
    Deriva, a partir do Arquivo 1 (`manifest`), o índice pasta -> lista de
    sonograma_id de cada rodada de K0. É 100% derivável — gerar sob
    demanda em vez de manter como fonte própria evita que os dois arquivos
    fiquem dessincronizados.
    """
    k0 = manifest["config"]["k0"]
    k1 = manifest["config"]["k1"]
    width = len(str(k0 - 1))

    indice: dict[str, dict] = {}
    for round_idx in range(k0):
        rodada_key = f"fold_{round_idx:0{width}d}"
        # chaves pré-criadas em ordem crescente — senão a ordem de inserção
        # no dict reflete a ordem de varredura dos sonogramas, não o índice
        # do fold_cv (fica "aleatória" de rodada pra rodada)
        bucket: dict[str, list[str]] = {"teste": [], **{f"fold_cv_{i}": [] for i in range(k1)}}
        for gid, info in manifest["sonogramas"].items():
            r = info["rodadas"][rodada_key]
            if r["is_teste"]:
                bucket["teste"].append(gid)
            else:
                bucket[f"fold_cv_{r['fold_cv']}"].append(gid)
        indice[rodada_key] = bucket
    return indice


# --------------------------------------------------------------------------- #
# 3c. Registro de contagens por classe/fold (auditoria) — fold_counts.csv
#     com as duas colunas de fold (rodada de K0 e pasta de K1)
# --------------------------------------------------------------------------- #
def write_k0k1_counts(
    manifest: dict,
    out_root: str | Path,
    filename: str = "fold_counts.csv",
) -> Path:
    """
    Grava `out_root/filename`: nº de imagens, de sonogramas e % de
    composição por classe, para cada combinação (rodada de K0, pasta
    daquela rodada).

    Colunas:
      fold      -> rodada de K0 (externa), ex.: fold_0
      fold_cv   -> pasta daquela rodada: "all" (agregado da rodada
                   inteira — teste + todos os fold_cv juntos, vem sempre
                   primeiro no bloco de cada rodada, como referência
                   global), "teste" (holdout externo, R4) ou "fold_cv_N"
                   (pasta interna de K1)
      classe, n_imagens, n_sonogramas
      pct_imagens -> % que aquela classe representa DENTRO daquele bucket
                     (fold, fold_cv) — as linhas de um mesmo bucket somam
                     100%; serve pra comparar a composição de cada pasta
                     contra o "all" da mesma rodada (conferindo a
                     estratificação, R2).

    Retorna o caminho do arquivo gravado.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / filename

    k0 = manifest["config"]["k0"]
    width = len(str(k0 - 1))

    contagens: dict[tuple[str, str, str], dict[str, int]] = {}
    totais_bucket: dict[tuple[str, str], int] = {}

    def _somar(rodada_key: str, bucket: str, classe: str, n_imagens: int) -> None:
        chave = (rodada_key, bucket, classe)
        registro = contagens.setdefault(chave, {"n_imagens": 0, "n_sonogramas": 0})
        registro["n_imagens"] += n_imagens
        registro["n_sonogramas"] += 1
        totais_bucket[(rodada_key, bucket)] = totais_bucket.get((rodada_key, bucket), 0) + n_imagens

    for round_idx in range(k0):
        rodada_key = f"fold_{round_idx:0{width}d}"
        for info in manifest["sonogramas"].values():
            r = info["rodadas"][rodada_key]
            bucket = "teste" if r["is_teste"] else f"fold_cv_{r['fold_cv']}"
            classe = info["classe"]
            # bucket específico (teste ou fold_cv_N) + "all", agregando a
            # rodada inteira (teste + todos os fold_cv juntos), pra servir
            # de referência global no início de cada bloco de rodada
            _somar(rodada_key, bucket, classe, info["n_imagens"])
            _somar(rodada_key, "all", classe, info["n_imagens"])

    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "fold_cv", "classe", "n_imagens", "n_sonogramas", "pct_imagens"])
        for (rodada_key, bucket, classe), registro in sorted(contagens.items()):
            total_bucket = totais_bucket[(rodada_key, bucket)]
            pct = 100 * registro["n_imagens"] / total_bucket if total_bucket else 0.0
            writer.writerow(
                [rodada_key, bucket, classe, registro["n_imagens"], registro["n_sonogramas"], round(pct, 2)]
            )

    return dest


# --------------------------------------------------------------------------- #
# 4. CLI orientada a YAML
# --------------------------------------------------------------------------- #
def load_manifest_cv_config(path: str | Path) -> dict:
    """Lê um config_manifest_cv.yaml (chaves no nível raiz do arquivo)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return {
        "src_root": cfg["src_root"],
        "out_root": cfg.get("out_root", "data"),
        "k0": cfg.get("k0", 10),
        "k1": cfg.get("k1", 5),
        "alpha": cfg.get("alpha", 1.0),
        "beta": cfg.get("beta", 1.0),
    }


def run_from_config(config_path: str | Path) -> Path:
    """
    Lê `src_root` (layout data_dir/<classe>/<sonograma_id>/<recorte>.png),
    calcula a divisão aninhada K0 x K1 e grava em `out_root`:

      manifest_k0k1.json           -> Arquivo 1 (sonogramas x rodadas de K0)
      manifest_k0k1_auditoria.json -> Arquivo 2 (pasta -> sonogramas, auditoria)
      fold_counts.csv              -> contagens/percentuais por classe

    Retorna `out_root`.
    """
    cfg = load_manifest_cv_config(config_path)

    cv = K0K1ManifestCV(
        k0=cfg["k0"],
        k1=cfg["k1"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
    )
    manifest = cv.fit_from_folders(cfg["src_root"])

    sonogramas = manifest["sonogramas"]
    n_grupos = len(sonogramas)
    n_imagens = sum(info["n_imagens"] for info in sonogramas.values())
    print(f"Lendo {cfg['src_root']}")
    print(f"Total: {n_imagens} imagens / {n_grupos} sonogramas")
    print(f"K0={cfg['k0']} pastas fixas | K1={cfg['k1']} pastas recalculadas a cada rodada\n")

    out_root = Path(cfg["out_root"])

    manifest_path = write_manifest(manifest, out_root / "manifest_k0k1.json")
    print(f"[ok] Arquivo 1 (sonogramas x rodadas de K0) gravado em: {manifest_path}")

    indice = derive_audit_index(manifest)
    indice_path = write_manifest(indice, out_root / "manifest_k0k1_auditoria.json")
    print(f"[ok] Arquivo 2 (auditoria, pasta -> sonogramas) gravado em: {indice_path}")

    counts_path = write_k0k1_counts(manifest, out_root)
    print(f"[ok] Contagens por classe/fold/fold_cv (com %) registradas em: {counts_path}")

    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="YAML com as configurações da divisão K0 x K1 (ex.: config_manifest_cv.yaml). "
             "Se omitido, roda a demonstração com a pasta ./data ao lado do script.",
    )
    args = parser.parse_args()

    if args.config:
        run_from_config(args.config)

if __name__ == "__main__":
    main()
