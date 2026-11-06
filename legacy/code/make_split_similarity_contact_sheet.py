"""Create a compact visual audit of the highest cross-split cosine pairs."""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/"results/split_similarity/top_cross_split_pairs.csv").open(encoding="utf-8")))
selected=[]
for comparison in ("validation-train","test-train","validation-test"):
    selected += [r for r in rows if r["comparison"]==comparison][:5]
w,h=980,220; canvas=Image.new("RGB",(w,h*len(selected)),"white"); draw=ImageDraw.Draw(canvas)
for i,r in enumerate(selected):
    y=i*h
    for j,key in enumerate(("left","right")):
        with Image.open(r[key]) as im:
            im=im.convert("RGB"); im.thumbnail((430,175)); canvas.paste(im,(10+j*480,y+35))
    draw.text((10,y+8),f'{r["comparison"]}  cosine={float(r["cosine"]):.4f}',fill="black")
canvas.save(ROOT/"results/split_similarity/top_pairs_contact_sheet.jpg",quality=92)
print(ROOT/"results/split_similarity/top_pairs_contact_sheet.jpg")
