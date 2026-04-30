import argparse
import random
import shutil
from pathlib import Path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def clear_split_dirs(output_dir: Path, classes: list[str]) -> None:
    for split in ["train", "val", "test"]:
        for class_name in classes:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for old_file in class_dir.iterdir():
                if old_file.is_file():
                    old_file.unlink()


def split_one_class(
    class_name: str,
    source_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    files = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not files:
        raise ValueError(f"No images found in class folder: {source_dir}")

    random.Random(seed).shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        target_dir = output_dir / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(split_files):
            dst_name = f"{class_name}_{i:04d}{src.suffix.lower()}"
            shutil.copy2(src, target_dir / dst_name)

    print(
        f"{class_name:8s} -> total={n_total:4d} | train={len(train_files):4d} | "
        f"val={len(val_files):4d} | test={len(test_files):4d}"
    )

    if n_test == 0:
        print(
            f"Warning: class '{class_name}' has 0 test images. Add more images for this class."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split raw images by class into train/val/test folders"
    )
    parser.add_argument("--raw_dir", type=str, default="raw_images")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["keyboard", "mouse", "laptop", "monitor"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="Delete existing files in data/train, data/val, data/test before copying",
    )

    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")

    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    for class_name in args.classes:
        class_source = raw_dir / class_name
        if not class_source.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_source}. Create it and add images."
            )

    if args.clear_output:
        clear_split_dirs(output_dir, args.classes)

    print("Preparing dataset split...")
    for class_name in args.classes:
        split_one_class(
            class_name=class_name,
            source_dir=raw_dir / class_name,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print("Done. You can now train with: python src/train.py --data_dir data")


if __name__ == "__main__":
    main()
