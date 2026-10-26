"""Render the repository's simple A/chart mark into Windows icon sizes."""

from pathlib import Path
from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[1]
assets = root / "webui" / "assets"
image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((8, 8, 504, 504), radius=112, fill="#101b2d")
draw.line([(110, 364), (240, 116), (282, 116), (402, 364)], fill="#f1f5f9", width=40, joint="curve")
draw.line([(156, 298), (222, 298), (278, 240), (336, 256), (400, 180)], fill="#36dbac", width=30, joint="curve")
draw.polygon([(362, 177), (416, 160), (411, 214)], fill="#36dbac")
image.save(assets / "app.png")
image.save(assets / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
image.save(assets / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
