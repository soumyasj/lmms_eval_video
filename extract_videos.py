import glob
import os
import sys
import zipfile

HF_HOME = "/mnt/lustre/work/kuehne/kqr828/.cache/huggingface"
cache_dir = os.path.join(HF_HOME, "video_mmmu")
snap = os.path.join(cache_dir, "datasets--lmms-lab--VideoMMMU", "snapshots")
zips = glob.glob(os.path.join(snap, "**", "*.zip"), recursive=True)
print("found zips:", [os.path.basename(z) for z in zips], flush=True)

for zf in zips:
    print(f"--- extracting {os.path.basename(zf)} ---", flush=True)
    with zipfile.ZipFile(zf) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("__MACOSX") or "/._" in name or os.path.basename(name).startswith("._"):
                continue
            target = os.path.join(cache_dir, name)
            if os.path.exists(target) and not name.endswith("/"):
                continue
            z.extract(info, cache_dir)
    print(f"done {os.path.basename(zf)}", flush=True)

# report
for sub in ["Art", "Science", "Humanities", "Engineering", "Medicine", "Business", "question_only"]:
    d = os.path.join(cache_dir, sub)
    n = len(glob.glob(os.path.join(d, "*.mp4"))) if os.path.isdir(d) else 0
    print(f"{sub}: {n} mp4", flush=True)
print("ALL DONE", flush=True)
