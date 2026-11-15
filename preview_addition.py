
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        default=None,
        help="Image ID (e.g., 00100). Defaults to first available pair."
    )
    parser.add_argument(
        "--root",
        default="processed_dataset_complementary_degraded/test",
        help="Path to dataset test folder."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save preview image."
    )

    args = parser.parse_args()
    root = Path(args.root)

    # Automatically select first image if no ID is provided.
    if args.id is None:
        images = sorted((root / "ir").glob("*.jpg"))

        if not images:
            raise FileNotFoundError(
                f"No IR images found in {root / 'ir'}"
            )

        image_id = images[0].stem
    else:
        image_id = Path(args.id).stem

    rgb_path = root / "rgb" / f"{image_id}.jpg"
    ir_path = root / "ir" / f"{image_id}.jpg"

    rgb = cv2.imread(str(rgb_path))
    ir = cv2.imread(str(ir_path))

    if rgb is None or ir is None:
        raise FileNotFoundError(
            f"Missing RGB or IR image for {image_id}"
        )

    # Resize IR if dimensions differ.
    if ir.shape[:2] != rgb.shape[:2]:
        ir = cv2.resize(
            ir,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # 50/50 pixelwise fusion.
    fused = cv2.addWeighted(
        rgb, 0.5,
        ir, 0.5,
        0
    )

    images = [
        ("RGB (degraded)", rgb),
        ("IR (degraded)", ir),
        ("50/50 Addition", fused)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (title, image) in zip(axes, images):
        ax.imshow(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Complementary Degradation — {image_id}",
        fontsize=14
    )

    plt.tight_layout()

    if args.save:
        output_dir = Path("evaluation/addition_previews")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{image_id}_addition.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

        print(f"Saved preview: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
