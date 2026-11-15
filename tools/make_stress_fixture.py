from __future__ import annotations
import json, math, os, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"; OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS,DURATION=640,360,20,12.0
CUES=[
 {"start":0.70,"end":1.30,"text":"짧은 자막"},
 {"start":1.90,"end":4.00,"text":"밝은 장면에서도 보여야 해"},
 {"start":4.20,"end":6.00,"text":"두 줄로 나오는\n대사도 처리"},
 {"start":6.50,"end":9.20,"text":"배경이 바뀌어도 같은 대사"},
 {"start":9.50,"end":11.30,"text":"마지막 자막"},
]

def font_path():
    windir = Path(os.environ.get("WINDIR", r"C:\\Windows"))
    candidates = [
        windir / "Fonts" / "malgun.ttf",
        windir / "Fonts" / "malgunsl.ttf",
        windir / "Fonts" / "gulim.ttc",
        windir / "Fonts" / "batang.ttc",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise RuntimeError("Korean test font not found")

def ffmpeg_path():
    bundled = ROOT / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg executable not found; expected bundled bin/ffmpeg or PATH")

def cue_at(t):
    for i,c in enumerate(CUES):
        if c["start"]<=t<c["end"]: return i,c
    return None,None

def scene_color(t):
    # hard scene cuts deliberately include one inside cue #4.
    if t<3.0:return (38,44,56)
    if t<5.0:return (205,210,220)
    if t<7.0:return (72,50,45)
    if t<9.8:return (25,88,82)
    return (150,82,105)

def render(t):
    base=scene_color(t)
    img=Image.new("RGB",(W,H),base)
    d=ImageDraw.Draw(img)

    # animated geometry across the whole frame, including the subtitle ROI.
    # Motion is intentionally slow enough that pairwise edge change stays cheap.
    x=int((t*35)%(W+160))-160
    d.ellipse([x,190,x+160,350],fill=(max(0,base[0]-18),min(255,base[1]+28),min(255,base[2]+18)))
    bx=int((W-90)*(0.5+0.5*math.sin(t*0.7)))
    d.rectangle([bx,220,bx+90,280],outline=(230,180,90),width=3)

    idx,cue=cue_at(t)
    if cue:
        fsize=26 if idx==0 else 30
        font=ImageFont.truetype(font_path(),fsize)
        lines=cue["text"].split("\n")
        boxes=[d.textbbox((0,0),line,font=font,stroke_width=3) for line in lines]
        hs=[b[3]-b[1] for b in boxes]
        y=H-25-(sum(hs)+(len(lines)-1)*7)
        fill=(255,238,112) if idx==4 else (255,255,255)
        for line,b,lh in zip(lines,boxes,hs):
            tw=b[2]-b[0]; tx=(W-tw)//2
            # shadow + thick black outline
            d.text((tx+2,y+3),line,font=font,fill=(0,0,0))
            d.text((tx,y),line,font=font,fill=fill,stroke_width=3,stroke_fill=(0,0,0))
            y+=lh+7
    return img

def main():
    frames=OUT/"stress_frames"
    if frames.exists():shutil.rmtree(frames)
    frames.mkdir()
    for i in range(int(DURATION*FPS)):
        render(i/FPS).save(frames/f"{i:05d}.png")
    video=OUT/"synthetic-korean-hardsub-stress.mp4"
    subprocess.run([ffmpeg_path(),"-y","-hide_banner","-loglevel","error","-framerate",str(FPS),
                    "-i",str(frames/"%05d.png"),"-c:v","mpeg4","-q:v","1",
                    "-pix_fmt","yuv420p","-r",str(FPS),str(video)],check=True)
    truth=OUT/"synthetic-stress-truth.json"
    truth.write_text(json.dumps({"fps":FPS,"duration":DURATION,"cues":CUES},ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.rmtree(frames)
    print(video); print(truth)

if __name__=="__main__":main()
