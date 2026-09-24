"""
Carregamento de imagens a partir do manifest gerado por manifest_cv.py
(`K0K1ManifestCV`) — só entra em jogo na hora de treinar. A divisão em si
(K0/K1, os arquivos JSON) nunca depende disto; é o próximo passo depois de
`sonograma_id` sair de `.split()`/`.test_groups()`.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except ImportError:  # só necessário se for de fato instanciar ManifestGroupDataset
    Image = None

try:
    from torch.utils.data import Dataset
except ImportError:  # opcional — a classe funciona por duck-typing sem torch instalado
    Dataset = object


class ManifestGroupDataset(Dataset):
    """
    Recebe uma lista de `sonograma_id` (ex.: o `tr`/`va` de
    `K0K1ManifestCV.split`, ou `.test_groups()`, em manifest_cv.py) e
    resolve as imagens de cada um a partir do campo `caminho` já gravado
    no manifest (Arquivo 1) — nenhum plano.json, nenhuma pasta
    materializada.
    """

    def __init__(
        self,
        group_ids: list[str],
        manifest: dict,
        transform=None,
        label_to_idx: dict | None = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        sonogramas = manifest["sonogramas"]
        self.entries = []
        for gid in group_ids:
            info = sonogramas[gid]
            pasta = Path(info["caminho"])
            classe = info["classe"]
            for img in sorted(pasta.iterdir()):
                if img.suffix.lower() in extensions:
                    self.entries.append((img, classe))
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, classe = self.entries[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_idx[classe]


if __name__ == "__main__":
    from manifest_cv import K0K1ManifestCV, load_manifest

    manifest_path = Path("data/manifest/manifest_k0k1.json")
    if not manifest_path.is_dir() and not manifest_path.is_file():
        manifest_path = Path("data/manifest_k0k1.json")

    manifest = load_manifest(manifest_path)
    cv = K0K1ManifestCV(k0=manifest["config"]["k0"], k1=manifest["config"]["k1"])

    classes = sorted({info["classe"] for info in manifest["sonogramas"].values()})
    label_to_idx = {classe: i for i, classe in enumerate(classes)}

    tr_ids, va_ids = next(cv.split(manifest, rodada=0))
    te_ids = cv.test_groups(manifest, rodada=0)

    train_ds = ManifestGroupDataset(tr_ids, manifest, label_to_idx=label_to_idx)
    val_ds = ManifestGroupDataset(va_ids, manifest, label_to_idx=label_to_idx)
    test_ds = ManifestGroupDataset(te_ids, manifest, label_to_idx=label_to_idx)

    print(f"rodada fold_0 | classes: {label_to_idx}")
    print(f"treino: {len(train_ds)} imagens | val: {len(val_ds)} imagens | teste: {len(test_ds)} imagens")
    print(f"amostra treino[0]: {train_ds[0][0].size}, rótulo {train_ds[0][1]}")
