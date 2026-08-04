from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


def safe_target(root: Path, relative_path: Path) -> Path:
    target = (root / relative_path).resolve()
    root = root.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe archive path: {relative_path}")
    return target


def copy_member(zip_file: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, *, force: bool) -> bool:
    if info.is_dir():
        return False

    if target.exists() and not force and target.stat().st_size == info.file_size:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(info, "r") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return True


def extract_keyframes(archive_path: Path, keyframe_root: Path, *, force: bool) -> tuple[int, int]:
    written = 0
    skipped = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for index, info in enumerate(members, start=1):
            normalized = info.filename.replace("\\", "/")
            marker = "/keyframe/"
            if marker not in normalized or info.is_dir():
                continue

            relative = Path(normalized.split(marker, 1)[1])
            target = safe_target(keyframe_root, relative)
            if copy_member(archive, info, target, force=force):
                written += 1
            else:
                skipped += 1

            if (written + skipped) % 5000 == 0:
                print(f"[keyframe] {archive_path.name}: written={written} skipped={skipped}")

    return written, skipped


def extract_inner_embedding_zip(inner_zip_path: Path, embedding_model_root: Path, *, force: bool) -> tuple[int, int]:
    written = 0
    skipped = 0

    with zipfile.ZipFile(inner_zip_path) as archive:
        for info in archive.infolist():
            normalized = info.filename.replace("\\", "/")
            marker = "/embedding/"
            if info.is_dir():
                continue

            if normalized.startswith("embedding/"):
                relative_text = normalized[len("embedding/"):]
            elif marker in normalized:
                relative_text = normalized.split(marker, 1)[1]
            else:
                continue

            relative = Path(relative_text)
            target = safe_target(embedding_model_root, relative)
            if copy_member(archive, info, target, force=force):
                written += 1
            else:
                skipped += 1

            if (written + skipped) % 5000 == 0:
                print(f"[embedding] {inner_zip_path.name}: written={written} skipped={skipped}")

    return written, skipped


def relative_after_dir(normalized_path: str, dir_name: str) -> Path | None:
    prefix = f"{dir_name}/"
    marker = f"/{dir_name}/"
    if normalized_path.startswith(prefix):
        relative_text = normalized_path[len(prefix):]
    elif marker in normalized_path:
        relative_text = normalized_path.split(marker, 1)[1]
    else:
        return None
    return Path(relative_text)


def extract_inner_keyframe_zip(inner_zip_path: Path, keyframe_root: Path, *, force: bool) -> tuple[int, int]:
    written = 0
    skipped = 0

    with zipfile.ZipFile(inner_zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            relative = relative_after_dir(normalized, "keyframe")
            if relative is None:
                continue

            target = safe_target(keyframe_root, relative)
            if copy_member(archive, info, target, force=force):
                written += 1
            else:
                skipped += 1

            if (written + skipped) % 5000 == 0:
                print(f"[keyframe] {inner_zip_path.name}: written={written} skipped={skipped}")

    return written, skipped


def extract_siglip_archives(
    data_root: Path,
    *,
    force: bool,
    extract_embeddings: bool = True,
    extract_keyframes: bool = True,
) -> tuple[int, int, int, int]:
    embedding_model_root = data_root / "embedding" / "SigLIP"
    keyframe_root = data_root / "keyframe"
    embedding_written_total = 0
    embedding_skipped_total = 0
    keyframe_written_total = 0
    keyframe_skipped_total = 0

    outer_archives = sorted(data_root.glob("SigLIP-*.zip"))
    temp_root = data_root / ".extract_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    for outer_path in outer_archives:
        print(f"[embedding] opening {outer_path.name}")
        with zipfile.ZipFile(outer_path) as outer:
            inner_members = [
                info for info in outer.infolist()
                if not info.is_dir() and info.filename.replace("\\", "/").lower().endswith(".zip")
            ]
            for inner_info in inner_members:
                inner_name = Path(inner_info.filename.replace("\\", "/")).name
                with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
                    inner_path = Path(temp_dir) / inner_name
                    with outer.open(inner_info, "r") as source, inner_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)

                    if extract_embeddings:
                        written, skipped = extract_inner_embedding_zip(inner_path, embedding_model_root, force=force)
                        embedding_written_total += written
                        embedding_skipped_total += skipped
                        print(f"[embedding] done {inner_info.filename}: written={written} skipped={skipped}")

                    if extract_keyframes:
                        written, skipped = extract_inner_keyframe_zip(inner_path, keyframe_root, force=force)
                        keyframe_written_total += written
                        keyframe_skipped_total += skipped
                        print(f"[keyframe] done {inner_info.filename}: written={written} skipped={skipped}")

    try:
        temp_root.rmdir()
    except OSError:
        pass

    return embedding_written_total, embedding_skipped_total, keyframe_written_total, keyframe_skipped_total


def normalize_numeric_embedding_names(embedding_model_root: Path) -> tuple[int, int]:
    renamed = 0
    conflicts = 0

    for path in embedding_model_root.rglob("*.pt"):
        if not re.fullmatch(r"\d+\.pt", path.name):
            continue
        frame_idx = int(path.stem)
        target = safe_target(path.parent, Path(f"keyframe_{frame_idx}.pt"))
        if target.exists():
            conflicts += 1
            continue
        path.rename(target)
        renamed += 1

    return renamed, conflicts


def normalize_numeric_keyframe_names(keyframe_root: Path) -> tuple[int, int]:
    renamed = 0
    conflicts = 0

    for path in keyframe_root.rglob("*.webp"):
        if not re.fullmatch(r"\d+\.webp", path.name):
            continue
        frame_idx = int(path.stem)
        target = safe_target(path.parent, Path(f"keyframe_{frame_idx}.webp"))
        if target.exists():
            conflicts += 1
            continue
        path.rename(target)
        renamed += 1

    return renamed, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AIC keyframe and SigLIP embedding archives.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--skip-keyframes", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-siglip-keyframes", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files even when sizes match.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    keyframe_root = data_root / "keyframe"

    data_root.mkdir(parents=True, exist_ok=True)
    keyframe_root.mkdir(parents=True, exist_ok=True)
    (data_root / "embedding" / "SigLIP").mkdir(parents=True, exist_ok=True)

    if not args.skip_keyframes:
        keyframe_archives = [
            path for path in sorted(data_root.glob("*.zip"))
            if not path.name.startswith("SigLIP-")
        ]
        for archive_path in keyframe_archives:
            print(f"[keyframe] opening {archive_path.name}")
            written, skipped = extract_keyframes(archive_path, keyframe_root, force=args.force)
            print(f"[keyframe] done {archive_path.name}: written={written} skipped={skipped}")

    if not args.skip_embeddings or not args.skip_siglip_keyframes:
        embedding_written, embedding_skipped, keyframe_written, keyframe_skipped = extract_siglip_archives(
            data_root,
            force=args.force,
            extract_embeddings=not args.skip_embeddings,
            extract_keyframes=not args.skip_siglip_keyframes,
        )
        if not args.skip_embeddings:
            print(f"[embedding] total SigLIP: written={embedding_written} skipped={embedding_skipped}")
        if not args.skip_siglip_keyframes:
            print(f"[keyframe] total SigLIP: written={keyframe_written} skipped={keyframe_skipped}")

    if not args.skip_embeddings:
        renamed, conflicts = normalize_numeric_embedding_names(data_root / "embedding" / "SigLIP")
        print(f"[embedding] normalized numeric names: renamed={renamed} conflicts={conflicts}")

    if not args.skip_keyframes or not args.skip_siglip_keyframes:
        renamed, conflicts = normalize_numeric_keyframe_names(data_root / "keyframe")
        print(f"[keyframe] normalized numeric names: renamed={renamed} conflicts={conflicts}")


if __name__ == "__main__":
    main()
