from PIL import Image
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_path = project_root / "data" / "input.png"
output_path = project_root / "data" / "input.ppm"

img = Image.open(input_path)
img = img.convert("RGB")

width, height = img.size

with open(output_path, "w") as f:
    f.write(f"P3\n{width} {height}\n255\n")
    
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            f.write(f"{r} {g} {b} ")
        f.write("\n")