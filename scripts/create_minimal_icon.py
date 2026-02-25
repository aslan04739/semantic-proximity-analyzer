from pathlib import Path
from PIL import Image, ImageDraw

base = Path("/Users/aslan/Documents/Semantic proximity ")
iconset = base / "assets" / "icon_minimal.iconset"
iconset.mkdir(parents=True, exist_ok=True)

sizes = [16, 32, 64, 128, 256, 512]

for size in sizes:
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    stroke = max(2, size // 16)
    radius = int(size * 0.18)
    center = (size // 2, size // 2)
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.ellipse(bbox, outline=(17, 17, 17, 255), width=stroke)
    img.save(iconset / f"icon_{size}x{size}.png")

    img2x = img.resize((size * 2, size * 2), Image.LANCZOS)
    img2x.save(iconset / f"icon_{size}x{size}@2x.png")

img = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)
stroke = 64
radius = int(1024 * 0.18)
center = (512, 512)
bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
draw.ellipse(bbox, outline=(17, 17, 17, 255), width=stroke)
img.save(iconset / "icon_1024x1024.png")
