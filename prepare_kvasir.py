"""Stage the authors' Kvasir split using the original Kvasir-SEG archive."""

import csv
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/polyp/source/kvasir-seg.zip"
SOURCE = ROOT / "data/polyp/source/Kvasir-SEG"
LISTS = ROOT / "data/polyp/author_lists"
TARGET = ROOT / "data/polyp/target/Kvasir"
SPLIT_SIZES = {"train": 800, "val": 100, "test": 100}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not ARCHIVE.is_file():
        raise SystemExit(f"Missing {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Corrupt archive member: {bad}")
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"Unsafe archive member: {member.filename}")
        if not SOURCE.is_dir():
            archive.extractall(ARCHIVE.parent)

    source_images = {p.stem: p for p in (SOURCE / "images").glob("*.jpg")}
    source_masks = {p.stem: p for p in (SOURCE / "masks").glob("*.jpg")}
    if len(source_images) != 1000 or source_images.keys() != source_masks.keys():
        raise SystemExit("Source archive must contain 1,000 matched image/mask pairs")

    splits = {}
    for split, count in SPLIT_SIZES.items():
        path = LISTS / f"{split}_list_kvasir.txt"
        if not path.is_file():
            raise SystemExit(f"Missing author list: {path}")
        names = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
        if len(names) != count or len(set(names)) != count:
            raise SystemExit(f"Unexpected count or duplicate in {path}: {len(names)}")
        splits[split] = names
    all_names = [name for names in splits.values() for name in names]
    if len(set(all_names)) != 1000 or set(all_names) != source_images.keys():
        raise SystemExit("Author lists overlap or do not match the Kvasir archive")

    TARGET.mkdir(parents=True, exist_ok=True)
    with (TARGET / "split_manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "author_filename", "source_image", "source_mask"])
        for split, names in splits.items():
            for part in ("images", "masks"):
                (TARGET / split / part).mkdir(parents=True, exist_ok=True)
            for name in names:
                for part, source in (("images", source_images[name]), ("masks", source_masks[name])):
                    link = TARGET / split / part / source.name
                    if link.is_symlink():
                        link.unlink()
                    elif link.exists():
                        raise SystemExit(f"Refusing to replace {link}")
                    link.symlink_to(source)
                writer.writerow([split, f"{name}.png", source_images[name].name, source_masks[name].name])
    lines = [f"{sha256(ARCHIVE)}  {ARCHIVE.name}"]
    lines += [f"{sha256(LISTS / f'{split}_list_kvasir.txt')}  {split}_list_kvasir.txt" for split in SPLIT_SIZES]
    (TARGET / "source_sha256.txt").write_text("\n".join(lines) + "\n")
    print("Author Kvasir split ready: train=800 val=100 test=100")


if __name__ == "__main__":
    main()
