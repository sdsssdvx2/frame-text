"""Real Windows Tk controls -> single OCR pass -> temporal candidate review images.

All GUI widgets and the private FFmpeg frame review run for real on scratch
media. The OCR adapter returns deterministic fake observations: this is
a GUI/storage integration test, NOT free-model accuracy or end-user mouse
approval. No operating HSE install, source program, or external account is
opened; user-consent dialogs are explicitly stubbed in this scratch trial.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

from hardsub_extractor import gui as gui_module
from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.models import Cue
from hardsub_extractor.writers import write_srt, write_ass
from hardsub_extractor.runtime_paths import resolve_binary
import subprocess
import tkinter as tk


def sample_segment_audit(diagnostics):
    for i in range(29):
        t = i * 7.0 / 28
        lines = []
        records = []
        if 1.5 <= t <= 3.5:
            lines.append("첫 번째 대사")
            records.append({"line":0,"cx":0.51,"cy":0.30})
        elif 4.3 <= t <= 6.3:
            lines.append("두 번째 대사")
            records.append({"line":0,"cx":0.51,"cy":0.30})
        records.append({"line":len(lines),"cx":0.36,"cy":0.76})
        lines.append("방송 안내")
        text = "\n".join(lines)
        obs = SimpleNamespace(pts=t,text=text,metadata={"detections":records})
        diagnostics.record_segment(
            i,SimpleNamespace(start=t,end=t+0.1),[obs],
            SimpleNamespace(text=text))


def create_real_scratch_video(video):
    command=[
        resolve_binary("ffmpeg"),"-nostdin","-hide_banner","-loglevel","error",
        "-f","lavfi","-i","testsrc2=size=640x360:rate=10:duration=7",
        "-map","0:v:0","-an","-sn","-dn","-c:v","mpeg4","-q:v","5",
        "-t","7","-y",str(video),
    ]
    r=subprocess.run(command,stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                     timeout=90,check=False)
    if r.returncode or not video.is_file() or not 1000<video.stat().st_size<40000000:
        raise RuntimeError("scratch_motion_video_unavailable")


def check():
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix="hse-temporal-actual-tk-") as temp:
        root_dir=Path(temp)
        video=root_dir/"scratch-motion.mp4"
        output=root_dir/"captions"
        model_dir=root_dir/"no-model-needed"
        model_dir.mkdir()
        create_real_scratch_video(video)
        before_video=sha256(video.read_bytes()).hexdigest()
        actual_extractions=[]
        notices=[]
        def fake_extract(source,engine,out,**kwargs):
            if source!=str(video) or out!=output or not kwargs.get("streaming"):
                raise RuntimeError("real_gui_did_not_freeze_local_source")
            audit=kwargs.get("diagnostics")
            if audit is None:
                raise RuntimeError("real_gui_temporal_audit_not_forwarded")
            sample_segment_audit(audit)
            originals=[
                Cue(0.0,1.0,"방송 안내"),
                Cue(1.5,3.5,"첫 번째 대사\n방송 안내"),
                Cue(4.3,6.3,"두 번째 대사\n방송 안내"),
                Cue(6.3,7.0,"방송 안내"),
            ]
            srt=write_srt(originals,output.with_suffix(".srt"))
            ass=write_ass(originals,output.with_suffix(".ass"))
            actual_extractions.append((str(srt),str(ass)))
            return SimpleNamespace(srt_path=srt,ass_path=ass,cue_count=4)

        root=tk.Tk()
        root.withdraw()
        try:
            app=ExtractorGUI(root)
            app.source_mode_var.set("local")
            app._source_mode_changed()
            app.url_var.set(str(video))
            app.out_var.set(str(output))
            app.ocr_language_var.set("ja")
            app.review_mode_var.set("반복 워터마크 후보")

            with (mock.patch.object(gui_module,"acquire_local_embedded_text",
                                    return_value=None),
                  mock.patch.object(gui_module,"acquire_local_sidecar_text",
                                    return_value=None),
                  mock.patch.object(gui_module,"acquire_local_embedded_image",
                                    return_value=None),
                  mock.patch.object(gui_module,"bundled_model_dir",
                                    return_value=model_dir),
                  mock.patch.object(gui_module,"bundled_manifest_path",
                                    return_value=root_dir/"model-manifest.json"),
                  mock.patch.object(gui_module,"RapidOCRAdapter",
                                    return_value=mock.Mock()),
                  mock.patch.object(gui_module,"AdaptiveOCREngine",
                                    return_value=mock.Mock()),
                  mock.patch.object(gui_module,"extract_local_video",
                                    side_effect=fake_extract) as extractor,
                  mock.patch("hardsub_extractor.frame_source.probe_video",
                             return_value=SimpleNamespace(duration=7.0)),
                  mock.patch.object(gui_module.messagebox,"askyesno",
                                    side_effect=lambda *a,**kw: notices.append("approval") or True),
                  mock.patch.object(gui_module.messagebox,"showerror",
                                    side_effect=lambda *a,**kw: notices.append("error")),
                  mock.patch.object(gui_module.messagebox,"showinfo",
                                    side_effect=lambda *a,**kw: notices.append("info"))):
                app.start_button.invoke()
                deadline=time.monotonic()+18
                while time.monotonic()<deadline:
                    root.update()
                    if app.status_var.get()=="완료":
                        break
                    time.sleep(.06)
                else:
                    raise RuntimeError("real_Tk_OCR_worker_did_not_complete")
                if extractor.call_count!=1 or len(actual_extractions)!=1:
                    raise RuntimeError("OCR_pass_not_exactly_once")
                hints=app._temporal_review_hints
                if hints is None or hints[-1]!=(1,2,3,4):
                    raise RuntimeError("real_Tk_lost_temporal_candidate_numbers")
                srt=Path(actual_extractions[0][0])
                ass=Path(actual_extractions[0][1])
                before_subs=(sha256(srt.read_bytes()).digest(),
                             sha256(ass.read_bytes()).digest())
                if str(app.review_button.cget("state"))!="normal":
                    raise RuntimeError("real_Tk_review_button_not_enabled")
                app.review_button.invoke()
                deadline=time.monotonic()+35
                while time.monotonic()<deadline:
                    root.update()
                    if app.status_var.get()=="원본 화면 검토 완료":
                        break
                    time.sleep(.06)
                else:
                    raise RuntimeError("real_Tk_private_frame_review_not_complete")
                dest=root_dir/"captions.review-temporal-after-00000"
                index=json.loads((dest/"index.json").read_text("utf-8"))
                actual=index.get("review_cues") or []
                if index.get("mode")!="temporal" or [
                    x.get("cue") for x in actual] != [1,2,3,4]:
                    raise RuntimeError("real_Tk_temporal_review_wrong_cues")
                if any("temporal_persistent_overlay_review" not in
                       x.get("warning_codes",()) for x in actual):
                    raise RuntimeError("real_Tk_temporal_review_warning_missing")
                names=[f"cue-{i:05d}-{stage}.jpg"
                       for i in range(1,5) for stage in ("early","late")]
                if not all((dest/n).is_file() for n in names):
                    raise RuntimeError("real_Tk_decoded_private_images_missing")
                for name in names+["contact-sheet.jpg"]:
                    with Image.open(dest/name) as img:
                        img.verify()
                        if img.format!="JPEG":
                            raise RuntimeError("real_Tk_private_evidence_corrupt")
                if (before_video!=sha256(video.read_bytes()).hexdigest()
                    or before_subs!=(sha256(srt.read_bytes()).digest(),
                                     sha256(ass.read_bytes()).digest())):
                    raise RuntimeError("real_Tk_review_modified_original_files")
                if len(notices)<2 or "error" in notices:
                    raise RuntimeError("real_Tk_unexpected_modal_or_missing_consent")
                if any(x in json.dumps(index,ensure_ascii=False)
                       for x in ("방송 안내","첫 번째 대사","두 번째 대사")):
                    raise RuntimeError("private_Tk_review_manifest_contains_OCR_text")
                return {
                    "real_Tk_widgets":True,
                    "real_Tk_start_callback":True,
                    "real_worker_single_OCR_pass":True,
                    "real_Tk_review_callback":True,
                    "actual_FFmpeg_scratch_frame_decode":True,
                    "temporal_reviewed_cues":len(actual),
                    "private_JPEG_frames":len(names),
                    "originals_unchanged":True,
                    "dialog_approvals_stubbed":True,
                    "PP_OCR_inference":"MOCK",
                    "real_user_media":"NO",
                    "user_profile_changed":False,
                    "release":"HOLD",
                }
        finally:
            root.destroy()


def main():
    try:
        result=check()
    except Exception as exc:
        print("HSE_REAL_TK_TEMPORAL_ERROR="+type(exc).__name__,flush=True)
        return 3
    import json
    print("HSE_REAL_TK_TEMPORAL="+json.dumps(result,sort_keys=True),
          flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
