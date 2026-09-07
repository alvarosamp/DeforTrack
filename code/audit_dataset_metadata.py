"""Audit whether the exported dataset supports geographic/temporal/biome splits."""
import json, re
from pathlib import Path
from PIL import Image, ExifTags

ROOT=Path(r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3")
OUT=Path(r"results\metadata_audit.json")
date_tags={"DateTime","DateTimeOriginal","DateTimeDigitized"}; gps_tag="GPSInfo"
report={"root":str(ROOT),"splits":{},"method":"EXIF and export-structure audit; visual appearance was not used to infer provenance."}
for split in ("train","valid","test"):
    images=sorted((ROOT/split/"images").glob("*")); gps=date=exif=errors=0; examples=[]
    extensions={}; sidecars={}
    for p in images:
        extensions[p.suffix.lower()]=extensions.get(p.suffix.lower(),0)+1
        try:
            with Image.open(p) as im:
                raw=im.getexif(); decoded={ExifTags.TAGS.get(k,k):v for k,v in raw.items()}
                exif += bool(decoded); gps += gps_tag in decoded; date += any(k in decoded for k in date_tags)
                if decoded and len(examples)<5: examples.append({"file":p.name,"tags":sorted(map(str,decoded))})
        except Exception: errors+=1
    for folder in (ROOT/split).iterdir():
        if folder.name!="images": sidecars[folder.name]=sum(1 for x in folder.rglob("*") if x.is_file())
    report["splits"][split]={"images":len(images),"extensions":extensions,"with_any_exif":exif,"with_gps":gps,"with_capture_date":date,"read_errors":errors,"other_export_files":sidecars,"exif_examples":examples}
all_files=[p.name for s in report["splits"] for p in (ROOT/s/"images").glob("*")]
tokens=("amazon","atlantic","cerrado","caatinga","pantanal","pampa","mata_atlantica","amazonia")
report["filename_token_hits"]={t:sum(bool(re.search(t,n,re.I)) for n in all_files) for t in tokens}
report["conclusion"]="A geographic, temporal, or biome-aware split is unsupported when GPS, capture dates, and authoritative biome/source sidecars are absent. Filename tokens alone are not authoritative metadata."
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=False))
