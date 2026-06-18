"""Generate CrimeLens AI logo."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

assets_dir = Path(__file__).resolve().parent.parent / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
logo_path = assets_dir / "logo.png"

size = 256
img = Image.new("RGBA", (size, size), (13, 27, 42, 255))
draw = ImageDraw.Draw(img)

# Shield shape
shield = [
    (128, 20), (220, 60), (220, 140),
    (128, 230), (36, 140), (36, 60),
]
draw.polygon(shield, fill=(26, 35, 126, 255), outline=(79, 195, 247, 255), width=3)

# Inner circle
draw.ellipse([78, 70, 178, 170], fill=(79, 195, 247, 200), outline=(255, 255, 255, 255), width=2)

# Cross/star symbol
draw.line([(128, 90), (128, 150)], fill=(26, 35, 126, 255), width=6)
draw.line([(98, 120), (158, 120)], fill=(26, 35, 126, 255), width=6)

# Text
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except OSError:
    font = ImageFont.load_default()

draw.text((62, 185), "CRIMELENS", fill=(79, 195, 247, 255), font=font)

img.save(logo_path, "PNG")
print(f"Logo saved to {logo_path}")
