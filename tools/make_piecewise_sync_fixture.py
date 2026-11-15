from __future__ import annotations
import math,shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts";OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS,SOURCE_DURATION=480,270,20,16.0
LEADER=.75
MID_EDIT_AT=7.0
MID_INSERT=.50
TARGET_DURATION=SOURCE_DURATION+LEADER+MID_INSERT

def scene(t:float,source_subtitle:bool)->Image.Image:
    block=int(t//1.6)
    palette=[(30,60,100),(120,45,55),(45,105,70),(125,85,30),(72,48,118),(120,110,85),(35,105,120)]
    bg=palette[block%len(palette)]
    img=Image.new("RGB",(W,H),bg);d=ImageDraw.Draw(img)
    x=int((t*89)%(W+90))-90
    y=32+int(35*math.sin(t*1.17))
    d.rectangle([x,y,x+95,y+62],fill=(235,175,70))
    cx=int(W*(.5+.35*math.sin(t*.63)))
    cy=105+int(55*math.cos(t*1.03))
    d.ellipse([cx-35,cy-35,cx+35,cy+35],fill=(50,185,205))
    # distinct diagonal line changes continuously
    y2=int(25+(t*37)%(H-90))
    d.line([20,y2,W-20,y2+35],fill=(245,245,245),width=4)
    if source_subtitle and 1<t<SOURCE_DURATION-1:
        d.rectangle([75,H-46,W-75,H-16],fill=(0,0,0))
        d.rectangle([110,H-39,W-110,H-25],fill=(250,250,250))
    return img

def target_content_time(t:float):
    if t<LEADER:return None
    x=t-LEADER
    if x<MID_EDIT_AT:return x
    if x<MID_EDIT_AT+MID_INSERT:return MID_EDIT_AT
    return x-MID_INSERT

def encode(folder,target):
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-framerate",str(FPS),
                    "-i",str(folder/"%05d.png"),"-c:v","mpeg4","-q:v","1","-pix_fmt","yuv420p",
                    "-r",str(FPS),str(target)],check=True)

def main():
    a=OUT/"piece_src_frames";b=OUT/"piece_dst_frames"
    for d in (a,b):
        if d.exists():shutil.rmtree(d)
        d.mkdir()
    for i in range(int(SOURCE_DURATION*FPS)):
        t=i/FPS;scene(t,True).save(a/f"{i:05d}.png")
    for i in range(int(TARGET_DURATION*FPS)):
        t=i/FPS;ct=target_content_time(t)
        img=Image.new("RGB",(W,H),(0,0,0)) if ct is None else scene(ct,False)
        img.save(b/f"{i:05d}.png")
    src=OUT/"piecewise-source.mp4";dst=OUT/"piecewise-target.mp4"
    encode(a,src);encode(b,dst)
    shutil.rmtree(a);shutil.rmtree(b)
    print(src);print(dst)
    print("expected: +0.75 before source 7.0s, +1.25 after")

if __name__=="__main__":main()
