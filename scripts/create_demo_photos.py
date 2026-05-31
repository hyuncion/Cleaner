from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def make_image(index: int, out_dir: Path) -> None:
    w, h = random.choice([(1024, 768), (768, 1024), (1200, 900), (900, 1200)])
    bg = tuple(random.randint(30, 230) for _ in range(3))
    im = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(im)

    for _ in range(random.randint(3, 12)):
        x1 = random.randint(0, w - 10)
        y1 = random.randint(0, h - 10)
        x2 = min(w, x1 + random.randint(20, w // 3))
        y2 = min(h, y1 + random.randint(20, h // 3))
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=random.randint(2, 8))

    draw.text((24, 24), f"demo photo {index:03d}", fill=(255, 255, 255))

    # Some intentionally blurry / screenshot-like samples.
    if index % 7 == 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=4))
        name = f"blurry_{index:03d}.jpg"
    elif index % 9 == 0:
        name = f"Screenshot_{index:03d}.png"
    else:
        name = f"photo_{index:03d}.jpg"

    im.save(out_dir / name, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_photos"))
    parser.add_argument("--count", type=int, default=80)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        make_image(i, args.out)

    print(f"Created {args.count} demo photos in {args.out.resolve()}")


if __name__ == "__main__":
    main()
