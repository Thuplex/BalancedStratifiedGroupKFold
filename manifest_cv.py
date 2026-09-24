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

Reaproveita o NÚCLEO do algoritmo (GroupTable, assign_groups_to_folds) de
folders_frist.py — não duplica a lógica de custo/atribuição, só o "entra e
sai" ao redor dela. É a implementação atual do projeto para a validação
cruzada aninhada; o antigo `cross_validation.py` (rodada única) foi
removido por ficar redundante com `K0K1ManifestCV`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from folders_frist import GroupTable, assign_groups_to_folds


# --------------------------------------------------------------------------- #
# 1. Leitura DIRETO em nível de grupo (nunca cria 1 linha por imagem)
# --------------------------------------------------------------------------- #
def build_group_table_from_folders(
    data_dir: str | Path,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> GroupTable:
    """
    Varre `data_dir/<classe>/<sonograma_id>/*.png` e monta a GroupTable
    direto — sem nunca criar uma linha por imagem (folders_frist.py monta
    um df de N imagens e só depois agrega; aqui já nasce agregado).

    Assume 1 classe por sonograma (verdade neste dataset: a classe é a
    pasta-mãe do sonograma), então a contagem por classe do grupo é só
    [0, ..., n, ..., 0], com n = nº de imagens da pasta.
    """
    data_dir = Path(data_dir)
    group_ids, classe_por_grupo, sizes = [], [], []

    for classe_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for sonograma_dir in sorted(p for p in classe_dir.iterdir() if p.is_dir()):
            n = sum(1 for f in sonograma_dir.iterdir() if f.suffix.lower() in extensions)
            if n == 0:
                continue
            group_ids.append(sonograma_dir.name)
            classe_por_grupo.append(classe_dir.name)
            sizes.append(n)

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
        cv = K0K1ManifestCV(k0=10, k1=5, seed=42)
        manifest = cv.fit_from_folders("data/all_sonogram_folder")
        # manifest["config"]        -> k0, k1, alpha, beta, seed, ...
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
        seed: int = 42,
        n_refine: int = 20_000,
    ):
        self.k0 = k0
        self.k1 = k1
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.n_refine = n_refine

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
            gt, self.k0, self.alpha, self.beta, self.seed, self.n_refine
        )

        sonogramas: dict[str, dict] = {}
        for i, group_id in enumerate(gt.group_ids):
            classe = _classe_do_grupo(gt, i)
            sonogramas[str(group_id)] = {
                "caminho": str(src_root / classe / str(group_id)),
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
                gt_restante, self.k1, self.alpha, self.beta, self.seed, self.n_refine
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
                "seed": self.seed,
                "n_refine": self.n_refine,
                "src_root": str(src_root),
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
# 3c. Registro de contagens por classe/fold (auditoria) — equivalente ao
#     fold_counts.csv de folders_frist.py, com as duas colunas de fold
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
        "seed": cfg.get("seed", 42),
        "n_refine": cfg.get("n_refine", 20_000),
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
        seed=cfg["seed"],
        n_refine=cfg["n_refine"],
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
