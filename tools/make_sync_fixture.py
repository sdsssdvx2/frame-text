from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts";OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS,DURATION=480,270,20,12.0
OFFSET=0.75


def scene(content_t: float, *, source_subtitle: bool) -> Image.Image:
    # Structured, time-varying content with periodic scene cuts.
    block=int(content_t//2)
    palette=[
        (36,58,96),(112,50,42),(48,100,68),
        (102,76,34),(63,47,105),(112,96,80)
    ]
    bg=palette[block % len(palette)]
    img=Image.new("RGB",(W,H),bg)
    d=ImageDraw.Draw(img)

    x=int((content_t*73)% (W+80))-80
    y=35+int(30*math.sin(content_t*1.31))
    d.rectangle([x,y,x+85,y+65],fill=(220,180,80))
    cx=int(W*(0.5+0.35*math.sin(content_t*.77)))
    cy=110+int(45*math.cos(content_t*.93))
    d.ellipse([cx-38,cy-38,cx+38,cy+38],fill=(70,180,200))

    # Simulate burned subtitles only in the source. Visual sync's default ROI
    # excludes this bottom zone.
    if source_subtitle and 1.0 <= content_t <= 10.5:
        d.rectangle([80,H-48,W-80,H-18],fill=(0,0,0))
        d.rectangle([105,H-42,W-105,H-25],fill=(245,245,245))
    return img


def encode(folder: Path, target: Path):
    subprocess.run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-framerate",str(FPS),"-i",str(folder/"%05d.png"),
        "-c:v","mpeg4","-q:v","1","-pix_fmt","yuv420p","-r",str(FPS),
        str(target)
    ],check=True)


def main():
    src_frames=OUT/"sync_src_frames"
    dst_frames=OUT/"sync_dst_frames"
    for d in (src_frames,dst_frames):
        if d.exists():shutil.rmtree(d)
        d.mkdir()

    total=int(DURATION*FPS)
    for i in range(total):
        t=i/FPS
        scene(t,source_subtitle=True).save(src_frames/f"{i:05d}.png")

        content_t=t-OFFSET
        if content_t < 0:
            img=Image.new("RGB",(W,H),(0,0,0))
        else:
            img=scene(content_t,source_subtitle=False)
        img.save(dst_frames/f"{i:05d}.png")

    src=OUT/"sync-source.mp4"
    dst=OUT/"sync-target-plus-0750ms.mp4"
    encode(src_frames,src)
    encode(dst_frames,dst)
    shutil.rmtree(src_frames);shutil.rmtree(dst_frames)
    print(src);print(dst);print(f"expected offset={OFFSET}")


if __name__=="__main__":main()
