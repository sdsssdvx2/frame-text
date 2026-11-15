from pathlib import Path
from .models import Cue

def _srt_time(v:float)->str:
    ms=max(0,round(v*1000));h,r=divmod(ms,3600000);m,r=divmod(r,60000);s,ms=divmod(r,1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _ass_time(v:float)->str:
    cs=max(0,round(v*100));h,r=divmod(cs,360000);m,r=divmod(r,6000);s,cs=divmod(r,100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def write_srt(cues:list[Cue],path:str|Path)->Path:
    path=Path(path);lines=[]
    for i,c in enumerate(cues,1):lines += [str(i),f"{_srt_time(c.start)} --> {_srt_time(c.end)}",c.text,""]
    path.write_text("\n".join(lines),encoding="utf-8");return path

def _ass_plain_text(text: str) -> str:
    r"""Encode literal OCR text, never user-supplied ASS override commands.

    ASS has no universal literal-brace escape. Use FFmpeg's conservative
    libavcodec/ass.c convention: \{{} inserts a visible opening brace
    without starting an override block in libass. A backslash followed by
    U+2060 WORD JOINER cannot accidentally become \N, \h or a style tag.
    Generate intended \N hard line breaks only from real newlines.
    """
    out = []
    for char in text.replace("\r\n", "\n").replace("\r", "\n"):
        if char == "\\":
            out.append("\\\u2060")
        elif char == "{":
            out.append(r"\{{}")
        elif char == "\n":
            out.append(r"\N")
        else:
            out.append(char)
    return "".join(out)


def write_ass(cues:list[Cue],path:str|Path)->Path:
    path=Path(path)
    head="""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Pretendard,54,&H00FFFFFF,&H000000FF,&H00101010,&H70000000,0,0,0,0,100,100,0,0,1,3,1,2,80,80,54,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    body=[]
    for c in cues:
        text=_ass_plain_text(c.text)
        body.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{text}")
    path.write_text(head+"\n".join(body)+("\n" if body else ""),encoding="utf-8");return path


def ensure_subtitle_pair_available(base: str | Path) -> None:
    """Skip wasted OCR when an SRT or ASS destination already exists.

    This is a preflight optimization. The writer's exclusive O_EXCL
    creation remains the authority against concurrent output races.
    """
    import os
    base = Path(base)
    for suffix in (".srt", ".ass"):
        if os.path.lexists(base.with_suffix(suffix)):
            raise FileExistsError("subtitle output already exists")


def write_subtitle_pair_exclusive(
    cues: list[Cue], base: str | Path
) -> tuple[Path, Path]:
    """Publish SRT and ASS without ever replacing pre-existing user files.

    Temporary files use the existing canonical writers, keeping exactly the
    same bytes and timebase as the legacy output. Exclusive creation closes
    the long-running extraction's check-then-write race. If either final write
    fails, only files newly created during this call are rolled back.
    """
    import os
    import shutil
    import tempfile

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    final_srt, final_ass = base.with_suffix(".srt"), base.with_suffix(".ass")
    created: list[Path] = []

    with tempfile.TemporaryDirectory(prefix=".hse-subtitles-", dir=base.parent) as tmp:
        temp_base = Path(tmp) / "candidate"
        staged_srt = write_srt(cues, temp_base.with_suffix(".srt"))
        staged_ass = write_ass(cues, temp_base.with_suffix(".ass"))
        try:
            for staged, destination in (
                (staged_srt, final_srt),
                (staged_ass, final_ass),
            ):
                # xb is O_CREAT|O_EXCL: it also rejects existing hardlinks and
                # symlinks, unlike a preflight .exists() check.
                with destination.open("xb") as output:
                    created.append(destination)
                    with staged.open("rb") as source:
                        shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
        except BaseException:
            for destination in reversed(created):
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    return final_srt, final_ass
