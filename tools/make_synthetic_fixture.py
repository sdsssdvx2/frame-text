from __future__ import annotations
import json,math,os,shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"artifacts";OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS,DURATION=640,360,15,10.0
CUES=[
 {"start":1.0,"end":2.8,"text":"괜찮아"},
 {"start":3.4,"end":5.6,"text":"오늘은 조금 늦었어"},
 {"start":6.0,"end":8.4,"text":"두 줄 자막도\n정확히 잡아야 해"},
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
def text_at(t):
    return next((c["text"] for c in CUES if c["start"]<=t<c["end"]),"")
def render(t):
    img=Image.new("RGB",(W,H),(43,47,55));d=ImageDraw.Draw(img)
    for y in range(0,205,5):
        v=int(30+15*math.sin(y/30+t));d.rectangle([0,y,W,y+5],fill=(45+v,40+v//2,55+v//3))
    x=int((t*90)%(W+120))-120;d.rounded_rectangle([x,60,x+120,150],18,fill=(120,80,150))
    text=text_at(t)
    if text:
        f=ImageFont.truetype(font_path(),31);lines=text.split("\n")
        boxes=[d.textbbox((0,0),line,font=f,stroke_width=3) for line in lines]
        heights=[b[3]-b[1] for b in boxes];y=H-28-(sum(heights)+(len(lines)-1)*7)
        for line,b,lh in zip(lines,boxes,heights):
            tw=b[2]-b[0];d.text(((W-tw)//2,y),line,font=f,fill="white",stroke_width=3,stroke_fill="black");y+=lh+7
    return img
def main():
    frames=OUT/"synthetic_frames"
    if frames.exists():shutil.rmtree(frames)
    frames.mkdir()
    for i in range(int(DURATION*FPS)):render(i/FPS).save(frames/f"{i:05d}.png")
    video=OUT/"synthetic-korean-hardsub.mp4"
    subprocess.run([ffmpeg_path(),"-y","-hide_banner","-loglevel","error","-framerate",str(FPS),"-i",str(frames/"%05d.png"),
                    "-c:v","mpeg4","-q:v","1","-pix_fmt","yuv420p","-r",str(FPS),str(video)],check=True)
    (OUT/"synthetic-truth.json").write_text(json.dumps({"fps":FPS,"duration":DURATION,"cues":CUES},ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.rmtree(frames);print(video)
if __name__=="__main__":main()
