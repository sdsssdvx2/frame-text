"""Experimental local tab-screen recording -> existing HSE video OCR.

Chrome/Edge prompts the user to share a specific tab. WebM is streamed only to
127.0.0.1, with a single-use secret, bounded disk usage and no remote API.
After explicit Stop, HSE's existing local FFmpeg/RapidOCR path exports new,
exclusive SRT/ASS files. No user profile, cookies, DOM or player URL access.
This is a research candidate, NOT a tested packaged GUI or DRM workaround.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
import time
from urllib.parse import urlsplit, parse_qs
import webbrowser

MAX_CHUNK = 8 * 1024 * 1024
# At the existing 4 Mb/s VP8 target, a 1-GiB cap stops recordings after
# about 35 minutes. Eight GiB permits multi-hour source recordings, but only
# when real disk space remains; never exhaust the OS volume to reach the cap.
MAX_TOTAL = 8 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 512 * 1024 * 1024
CAPTION_MODES = frozenset(("all", "light", "light-temporal", "pixel-timing", "islands"))
DETECTION_MODES = frozenset(("edges", "persistent-white-3"))
# User-selected OCR reading region, never an automatic detection claim.
# All three keep the historical horizontal width and vertical bottom edge.
CAPTION_REGIONS = {
    "bottom40": (0.04, 0.58, 0.92, 0.40),  # immutable baseline
    "bottom25": (0.04, 0.73, 0.92, 0.25),  # opt-in, raised dialogue may be lost
    "lower65": (0.04, 0.33, 0.92, 0.65),   # two-row/high dialogue opt-in
    "lower80": (0.04, 0.18, 0.92, 0.80),   # wider opt-in, more false-text risk
}


def roi_for_region(region_mode):
    if type(region_mode) is not str or region_mode not in CAPTION_REGIONS:
        raise ValueError("invalid_caption_region_mode")
    from hardsub_extractor.frame_source import NormalizedROI
    return NormalizedROI(*CAPTION_REGIONS[region_mode])


class RecordingStorageLimit(RuntimeError):
    pass


class RecordingQuotaLimit(ValueError):
    pass

PAGE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>HSE · 브라우저 화면에서 자막 추출</title>
<style>body{font:16px system-ui,sans-serif;line-height:1.6;max-width:750px;margin:4rem auto;padding:0 1rem;background:#141b24;color:#edf1f8}button{padding:.7rem 1rem;margin:.4rem .2rem;border:0;border-radius:8px;background:#4b8ff0;color:white}button:disabled{opacity:.45}pre{white-space:pre-wrap;background:#242d3a;padding:1rem;border-radius:9px}</style>
</head><body><h1>HSE 브라우저 탭 자막 추출 · 연구 모드</h1>
<p>영상이 재생되는 탭을 공유 창에서 직접 선택하세요. 가능하면 영상만 전체 화면으로 표시하고, 시작부터 정상 속도로 재생하세요. 공유 화면은 PC의 임시 파일에만 저장한 뒤 무료 로컬 OCR로 SRT/ASS를 생성합니다. 보호된 영상이 검게 표시되면 추출할 수 없습니다.</p>
<p><label for="offset">원본 영상 재생 위치 (시:분:초.밀리초):</label> <input id="offset" value="00:00:00.000" maxlength="12" inputmode="numeric" aria-label="원본 영상 시작 위치"></p>
<p>영상 중간부터 녹화하면 해당 위치를 입력하세요. 녹화 시작 시 보이는 영상 시각과 일치해야 합니다. 화면 공유 선택 중 재생이 진행된다면 정확한 싱크는 달라질 수 있습니다. 입력하지 않으면 00:00:00부터 저장됩니다.</p>
<p><label for="duration">녹화 자동 종료 (시:분:초, 선택):</label> <input id="duration" value="00:00:00" maxlength="8" inputmode="numeric" aria-label="녹화 자동 종료 시간"></p>
<p>00:00:00은 수동 종료입니다. 자동 종료를 선택하면 15초~4시간 동안 녹화한 뒤 로컬 OCR를 시작합니다. 이 시간은 영상 재생 시간이 아닌 실제 녹화 시간으로, 일시정지·광고·백그라운드 브라우저 타이머 지연을 자동 보정하지 않습니다.</p>
<p><label for="caption-mode">자막 인식 방식:</label> <select id="caption-mode"><option value="all">기본 · 다양한 색상 글자</option><option value="light">밝은 글자만 선별 · 실험</option><option value="light-temporal">밝은 글자 + 픽셀 시간 보정 · 실험</option><option value="pixel-timing">문구·개수 보존 + 픽셀 시간만 보정 · 실험</option><option value="islands">자막 블록 분리 · 워터마크 주변 글자 실험</option></select></p>
<p>밝은 글자 모드는 검정·색깔 자막을 놓칠 수 있습니다. 픽셀 시간만 보정 모드는 기존 OCR 글자와 개수를 보존하지만 시간 정확도를 보증하지 않으며 영상 재디코딩에 CPU를 사용합니다. 자막 블록 분리는 기존 OCR의 글자 위치만 사용하므로 추가 AI 호출은 없지만 실제 자막 줄도 버릴 수 있습니다. 일반 영상은 기본 모드로 사용하세요.</p>
<p><label for="region-mode">자막이 표시되는 화면 영역:</label>
<select id="region-mode"><option value="bottom40">하단 40% · 기본</option><option value="lower65">하단 65% · 위로 올라온 자막</option><option value="lower80">하단 80% · 높은 두 줄 자막 (실험)</option></select></p>
<p>넓은 영역을 선택하면 인식 시간이 늘고 장면 내 글자·배너가 자막으로 잘못 인식될 수 있습니다. 기존 방식은 하단 40%입니다. 전체 화면이나 화면 최상단 자막을 보장하지 않습니다.</p>
<p><label for="detection-mode">화면 자막 변화 감지:</label>
<select id="detection-mode"><option value="edges">기본 · 일반 자막</option><option value="persistent-white-3">밝은 자막 3프레임 지속 감지 · 일본어/중국어 연구</option></select></p>
<p>실험 감지는 하단 65% + 기본 또는 자막 블록 분리 OCR에서만 선택할 수 있습니다. 검정·노랑·유색 자막이나 짧게 번쩍이는 자막에는 적합하지 않습니다. 기본 감지기는 바꾸지 않습니다.</p>
<button id="start">공유할 탭 선택 · 녹화 시작</button>
<button id="stop" disabled>녹화 종료 · 자막 추출</button>
<button id="close" disabled>완료 · 서버 종료</button>
<pre id="status">공유 동의 전 · 어떤 탭도 접근하지 않음</pre>
<script nonce="__NONCE__">
"use strict";
const token=__TOKEN__, st=document.getElementById("status");
const start=document.getElementById("start"),stop=document.getElementById("stop"),close=document.getElementById("close");
const offsetInput=document.getElementById("offset"),captionInput=document.getElementById("caption-mode"),
      durationInput=document.getElementById("duration"),
      regionInput=document.getElementById("region-mode"),
      detectionInput=document.getElementById("detection-mode");
function parseOffsetMillis(value){
 const match=/^([0-9]{2}):([0-5][0-9]):([0-5][0-9])(?:\.([0-9]{1,3}))?$/.exec(value.trim());
 if(!match)throw Error("원본 시작 위치는 00:00:00.000 형식으로 입력하세요");
 const millis=Number(match[1])*3600000+Number(match[2])*60000+
              Number(match[3])*1000+Number((match[4]||"").padEnd(3,"0")||0);
 if(!Number.isSafeInteger(millis)||millis>99*3600000)
  throw Error("원본 시작 위치가 허용 범위를 초과했습니다");
 return millis;
}
function parseCaptureDurationMillis(value){
 const match=/^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$/.exec(value.trim());
 if(!match)throw Error("자동 녹화 시간은 00:00:00 형식으로 입력하세요");
 const ms=(Number(match[1])*3600+Number(match[2])*60+Number(match[3]))*1000;
 if(ms!==0&&(ms<15000||ms>4*3600000))
  throw Error("자동 녹화 시간은 15초에서 4시간 사이여야 합니다");
 return ms;
}
let media=null, rec=null, queue=Promise.resolve(), queuedBytes=0, seq=0, failed=false, finishing=false;
let heartbeatActive=false, heartbeatFailures=0, lastHostProofMs=Date.now(),
    sourceOffsetMillis=0, captionMode="all", regionMode="bottom40", detectionMode="edges",
    autoDurationMillis=0, autoStopAt=null;
function status(s){st.textContent=s;}
async function send(path,body){
 const r=await fetch(path,{method:"POST",credentials:"omit",cache:"no-store",headers:{"X-HSE-Token":token,"Content-Type":"application/octet-stream"},body});
 if(!r.ok){
  if(r.status===507)throw Error("녹화 디스크 여유 공간이 부족합니다 · 자막 파일은 발행하지 않았습니다");
  if(r.status===413)throw Error("녹화 크기 한도를 초과했습니다 · 자막 파일은 발행하지 않았습니다");
  throw Error("로컬 전송 오류 "+r.status);
 }
 const result=await r.json();
 lastHostProofMs=Date.now();
 return result;
}
async function verifyCommittedChunk(index,blob){
 // Only used after ALL bounded POST retries have lost their ACK.
 // Verify both committed sequence and the exact original part bytes;
 // a mere request attempt or a matching byte count is not sufficient.
 if(!crypto?.subtle)return false;
 const digest=await crypto.subtle.digest("SHA-256",await blob.arrayBuffer());
 const sha=Array.from(new Uint8Array(digest),
                     b=>b.toString(16).padStart(2,"0")).join("");
 for(let attempt=0;attempt<3;attempt++){
  try{
   const controller=new AbortController();
   const deadline=setTimeout(()=>controller.abort(),4000);
   let response;
   try{
    response=await fetch("/status",{cache:"no-store",credentials:"omit",
      headers:{"X-HSE-Token":token},signal:controller.signal});
   }finally{clearTimeout(deadline);}
   if(!response.ok)throw Error("local_status_unavailable");
   const state=await response.json();
   if(state.state!=="recording")return false;
   if(state.last_chunk_index===index &&
      state.last_chunk_size===blob.size &&
      state.last_chunk_sha256===sha){
      lastHostProofMs=Date.now();return true;
   }
   // Another committed chunk would invalidate this ownership sequence.
   if(state.last_chunk_index>index)return false;
  }catch(_){}
  if(attempt<2)await new Promise(resolve=>setTimeout(resolve,300*(attempt+1)));
 }
 return false;
}
async function sendChunk(index,blob){
 // A Windows loopback response may be lost AFTER the local host accepted
 // a chunk. Its private SHA-256 receipt is checked after 3 lost POST ACKs;
 // the recording never advances on an unverified write.
 for(let attempt=0;attempt<3;attempt++){
  try{
   const result=await send("/chunk?i="+index,blob);
   if(result.state!=="recording"||result.chunks!==index+1)
     throw Error("로컬 녹화 조각 응답이 일치하지 않습니다");
   return;
  }catch(error){
   if(attempt===2){
    if(await verifyCommittedChunk(index,blob))return;
    throw error;
   }
   await new Promise(resolve=>setTimeout(resolve,180*(attempt+1)));
  }
 }
}
async function finishOnce(){
  // The first /finish may create the OCR job, then its HTTP reply may be
  // lost on Windows. Replaying the same empty finish is host-idempotent.
  for(let attempt=0;attempt<4;attempt++){
    try{
      const result=await send("/finish?offset_ms="+sourceOffsetMillis+
                              "&caption_mode="+captionMode+
                              "&region_mode="+regionMode+
                              "&detection_mode="+detectionMode,new Uint8Array(0));
      if(result.state!=="processing")throw Error("변환 시작 실패");
      return;
    }catch(error){
      if(attempt===3)throw error;
      await new Promise(resolve=>setTimeout(resolve,250*(attempt+1)));
    }
  }
}
function fail(e){
 if(failed)return;failed=true;status("중단: "+String(e&&e.message||e));
 start.disabled=true;stop.disabled=true;close.disabled=false;
 if(media)media.getTracks().forEach(t=>t.stop());
 if(rec&&rec.state!=="inactive")rec.stop();
}
async function poll(){
 let lastError=null;
 // A single Windows localhost TCP reset must not turn an already-running
 // or already-completed OCR job into a false user-visible failure.
 for(let attempt=0;attempt<5;attempt++){
  try{
   const r=await fetch("/status",{cache:"no-store",headers:{"X-HSE-Token":token}});
   if(!r.ok)throw Error("로컬 상태 조회 거부: "+r.status);
   const s=await r.json();
   if(s.state==="completed"){
    status("SRT/ASS 생성 완료 · 자막 "+s.cue_count+"개");
    close.disabled=false;return;
   }
   if(s.state==="failed"){
    status("로컬 OCR 실패: "+s.error_type);close.disabled=false;return;
   }
   if(s.state!=="processing")throw Error("잘못된 로컬 OCR 상태");
   const p=s.ocr_progress;
   if(p?.stage==="exporting"){
    status("로컬 OCR 완료 · SRT/ASS 파일 작성 중…");
   }else if(p?.stage==="recognizing"){
    const count=Number.isInteger(p.completed)?p.completed:0;
    let clock="";
    if(Number.isFinite(p.media_seconds)&&p.media_seconds>=0){
     const t=Math.floor(p.media_seconds);
     clock=" · 녹화 영상 "+String(Math.floor(t/3600)).padStart(2,"0")+
       ":"+String(Math.floor(t%3600/60)).padStart(2,"0")+
       ":"+String(t%60).padStart(2,"0");
    }
    status("로컬 OCR 처리 중 · 검사 "+count+"회"+clock);
   }else status("로컬 OCR 처리 중…");
   setTimeout(()=>poll().catch(fail),1600);
   return;
  }catch(error){
   lastError=error;
   if(attempt<4)await new Promise(resolve=>setTimeout(resolve,300*(attempt+1)));
  }
 }
 throw lastError;
}
// Poll only during an active share. If GUI requests stop, close the media
// tracks through the normal recorder onstop sequence. Lost helper connection
// also releases the browser's capture permission, never leaving sharing live.
async function heartbeat(){
 if(heartbeatActive||!rec||rec.state!=="recording"||finishing||failed)return;
 heartbeatActive=true;
 try{
  const controller=new AbortController();
  const watchdog=setTimeout(()=>controller.abort(),2800);
  let response;
  try{
   response=await fetch("/status",{cache:"no-store",credentials:"omit",
      headers:{"X-HSE-Token":token},signal:controller.signal});
  }finally{clearTimeout(watchdog);}
  if(!response.ok)throw Error("local_status_unavailable");
  const snapshot=await response.json();
  heartbeatFailures=0;lastHostProofMs=Date.now();
  if(Number.isSafeInteger(snapshot.recording_bytes) &&
     Number.isSafeInteger(snapshot.recording_limit_bytes)){
   status("탭 녹화 중 · 저장 "+(snapshot.recording_bytes/1073741824).toFixed(2)+
          " / "+(snapshot.recording_limit_bytes/1073741824).toFixed(0)+" GiB");
  }
  if(snapshot.stop_requested){
   finishing=true;stop.disabled=true;
   status("HSE 종료 요청 · 화면 공유를 끝내고 자막 추출 중");
   rec.stop();
  }
 }catch(_){
  heartbeatFailures++;
  // A transient Windows loopback GET reset is not proof that the helper
  // exited: successful chunk POST acknowledgements or SHA receipts also
  // establish that the authenticated local sink is still responding.
  // But a truly disconnected helper closes sharing within a bounded 10s.
  if(heartbeatFailures>=3 && Date.now()-lastHostProofMs>=10000)
   fail(Error("HSE 연결 확인이 10초 이상 끊겨 화면 공유를 종료했습니다"));
 }finally{heartbeatActive=false;}
}
function maybeStopAtDeadline(){
 if(autoStopAt===null||!rec||rec.state!=="recording"||finishing||failed)
  return false;
 // Absolute monotonic clock avoids accumulating background timer drift.
 // The callback itself can be delayed in a background Chrome tab.
 if(performance.now()<autoStopAt)return false;
 finishing=true;stop.disabled=true;autoStopAt=null;
 status("지정된 녹화 시간이 끝나 자동으로 자막 추출 중");
 rec.stop();
 return true;
}
setInterval(()=>{if(!maybeStopAtDeadline())void heartbeat();},800);
start.onclick=async()=>{
 try{
  sourceOffsetMillis=parseOffsetMillis(offsetInput.value);
  autoDurationMillis=parseCaptureDurationMillis(durationInput.value);
 }catch(e){status(String(e.message||e));return;}
 captionMode=captionInput.value;
 regionMode=regionInput.value;
 detectionMode=detectionInput.value;
 if(!["all","light","light-temporal","pixel-timing","islands"].includes(captionMode)){
  status("지원되지 않는 자막 인식 모드입니다");return;
 }
 if(!["bottom40","lower65","lower80"].includes(regionMode)){
  status("지원되지 않는 자막 인식 영역입니다");return;
 }
 if(!["edges","persistent-white-3"].includes(detectionMode)){
  status("지원되지 않는 자막 감지 방식입니다");return;
 }
 if(detectionMode==="persistent-white-3" &&
    (regionMode!=="lower65" || !["all","islands"].includes(captionMode))){
  status("밝은 자막 감지는 하단 65% + 기본 OCR에서만 선택하세요");return;
 }
 start.disabled=true;offsetInput.disabled=true;captionInput.disabled=true;
 durationInput.disabled=true;regionInput.disabled=true;detectionInput.disabled=true;
 try{
  if(!navigator.mediaDevices?.getDisplayMedia)throw Error("화면 공유 미지원 브라우저");
  // Must be called directly from a user-gesture handler; no site selection
  // or profile access is automated.
  media=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:30},audio:false});
  if(media.getVideoTracks().length!==1)throw Error("비디오 공유 트랙이 없습니다");
  const mime=["video/webm;codecs=vp8","video/webm"].find(x=>MediaRecorder.isTypeSupported(x));
  if(!mime)throw Error("WebM 인코더를 사용할 수 없습니다");
  rec=new MediaRecorder(media,{mimeType:mime,videoBitsPerSecond:4000000});
  media.getVideoTracks()[0].onended=()=>{if(rec?.state!=="inactive"&&!finishing){finishing=true;rec.stop();}};
  // A fatal MediaRecorder error is never an acceptable truncated video.
  // Some engines still emit dataavailable followed by stop after onerror.
  rec.onerror=()=>fail(Error("녹화 인코더 오류 · 부분 영상은 자막으로 발행하지 않습니다"));
  rec.ondataavailable=e=>{
   if(failed||!e.data?.size)return;
   const blob=e.data;
   // MediaRecorder timeslice is advisory. Browser stalls can deliver a
   // multi-second Blob, even with rec.start(1000). Keep HTTP requests at
   // the existing 8 MiB host bound by slicing the opaque WebM byte stream
   // without decoding or re-encoding it. Concatenation stays byte-identical.
   const partLimit=8388608,bufferLimit=134217728;
   if(!Number.isSafeInteger(blob.size)||
      blob.size>bufferLimit||queuedBytes+blob.size>bufferLimit){
    fail(Error("녹화 전송 대기 버퍼 초과 · 자막 파일은 발행하지 않았습니다"));return;
   }
   queuedBytes+=blob.size;
   for(let begin=0;begin<blob.size;begin+=partLimit){
    const part=blob.slice(begin,Math.min(blob.size,begin+partLimit));
    const i=seq++;
    queue=queue.then(()=>failed?undefined:sendChunk(i,part)).catch(fail);
   }
   queue=queue.finally(()=>{queuedBytes-=blob.size;});
  };
  rec.onstop=async()=>{
   stop.disabled=true;if(media)media.getTracks().forEach(t=>t.stop());
   if(failed)return;
   if(!seq){
    fail(Error("녹화 데이터가 없어 자막 파일을 만들지 않았습니다"));
    return;
   }
   try{await queue;if(failed)return;
    await finishOnce();
    status("녹화 완료 · 로컬 OCR 시작");await poll();
   }catch(e){fail(e);}
  };
  rec.start(1000);
  autoStopAt=autoDurationMillis?performance.now()+autoDurationMillis:null;
  lastHostProofMs=Date.now();stop.disabled=false;
  status("선택한 탭 녹화 중 · 영상 종료 후 추출 버튼을 누르세요");
 }catch(e){fail(e);}
};
stop.onclick=()=>{if(rec?.state==="recording"&&!finishing){
 finishing=true;stop.disabled=true;rec.stop();
}};
close.onclick=async()=>{
 close.disabled=true;
 try{
  await send("/close",new Uint8Array(0));
  status("로컬 작업 서버 종료됨");
 }catch(_){
  status("서버 종료 응답을 확인하지 못했습니다 · HSE 창에서 안전 종료하세요");
  close.disabled=false;
 }
};
</script></body></html>'''


def offline_ocr(video: Path, output_base: Path, *, language='ko',
                on_progress=None, output_time_offset_ms: int = 0,
                caption_mode: str = "all", region_mode: str = "bottom40",
                detection_mode: str = "edges"):
    from hardsub_extractor.pipeline import extract
    from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
    from hardsub_extractor.extraction_progress import ExtractionMonitor
    from hardsub_extractor.runtime_paths import bundled_model_dir, bundled_manifest_path

    from hardsub_extractor.model_assets import normalize_ocr_language
    language = normalize_ocr_language(language)
    if type(caption_mode) is not str or caption_mode not in CAPTION_MODES:
        raise ValueError("invalid_caption_recognition_mode")
    if type(detection_mode) is not str or detection_mode not in DETECTION_MODES:
        raise ValueError("invalid_caption_detection_mode")
    if detection_mode != "edges" and (
        region_mode != "lower65" or caption_mode not in ("all","islands")
    ):
        raise ValueError("unsupported_experimental_detection_combination")
    models, manifest = bundled_model_dir(), bundled_manifest_path()
    if not models.is_dir() or not manifest.is_file():
        raise FileNotFoundError("free_local_ocr_models_missing")
    engine = AdaptiveOCREngine(RapidOCRAdapter(
        language=language, model_dir=str(models), manifest_path=str(manifest)))
    refiner = None
    if caption_mode in ("light", "light-temporal"):
        from hardsub_extractor.ocr.light_text import LightCaptionEngine
        engine = LightCaptionEngine(engine)
    if caption_mode in ("light-temporal", "pixel-timing"):
        # The pixel-only option deliberately keeps the unfiltered OCR text;
        # only the already existing optional optical boundary refiner runs.
        from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
        engine = LightTemporalWitnessEngine(engine)
        refiner = engine.refine_timings
    if caption_mode == "islands":
        # The spatial-caption policy is separately tested on 761 public
        # images. Its video/generalization quality remains research-only.
        from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine
        engine = CaptionIslandEngine(engine)
    roi=roi_for_region(region_mode)
    # Never change default HSE crop; the non-default profiles are explicit.
    args={"roi":roi} if region_mode!="bottom40" else {}
    if detection_mode == "persistent-white-3":
        from hardsub_extractor.detector import DetectorConfig
        args["detector"] = DetectorConfig(signature_mode="persistent_white_3")
    return extract(
        str(video), engine, output_base, sample_fps=8.0,
        streaming=True, monitor=ExtractionMonitor(
            on_progress=on_progress, exclusive_outputs=True),
        output_time_offset_ms=output_time_offset_ms,
        timing_refiner=refiner,
        **args,
    )


class CaptureSession:
    """One recording per explicit launch; never overwrite existing subtitles."""

    def __init__(self, temp: Path, output_dir: Path, processor=None):
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError("output_directory_must_exist_without_symlink")
        self.temp = temp
        self.output_dir = output_dir
        self.processor = processor or offline_ocr
        self.token = secrets.token_urlsafe(32)
        self.nonce = secrets.token_urlsafe(18)
        self.video = temp / "recorded-tab.webm"
        self.writer = self.video.open("xb")
        self.lock = threading.RLock()
        self.state = "recording"
        self.next_index = 0
        self.last_accepted = None
        self.total = 0
        self.cue_count = 0
        self.error_type = None
        self.worker = None
        self.stop_requested = False
        self.published_base = None
        self.output_offset_ms = 0
        self.caption_mode = "all"
        self.region_mode = "bottom40"
        self.detection_mode = "edges"
        self.ocr_stage = None
        self.ocr_completed = 0
        self.ocr_media_seconds = None

    def mark_progress(self, event):
        # Only safe count/time metadata from the existing OCR monitor.
        # Never retain OCR text, source URLs or decoded screen pixels here.
        if (event.stage not in ("recognizing", "exporting", "completed")
                or type(event.completed) is not int or event.completed < 0):
            return
        with self.lock:
            if self.state != "processing":
                return
            self.ocr_stage = event.stage
            if event.stage == "recognizing":
                self.ocr_completed = max(self.ocr_completed, event.completed)
                if event.media_seconds is not None:
                    self.ocr_media_seconds = max(
                        self.ocr_media_seconds or 0.0, event.media_seconds)

    def append(self, index, payload):
        with self.lock:
            if self.state != "recording" or type(index) is not int:
                raise ValueError("capture_sequence_invalid")
            if not payload or len(payload) > MAX_CHUNK:
                raise ValueError("capture_size_out_of_bounds")
            digest = hashlib.sha256(payload).digest()
            if index == self.next_index-1:
                if self.last_accepted != (index, len(payload), digest):
                    raise ValueError("capture_replay_data_mismatch")
                return {"state":"recording","chunks":self.next_index,
                        "replayed":True}
            if index != self.next_index:
                raise ValueError("capture_sequence_invalid")
            if self.total+len(payload) > MAX_TOTAL:
                raise RecordingQuotaLimit("capture_total_quota")
            if index == 0 and payload[:4] != bytes.fromhex("1a45dfa3"):
                raise ValueError("recording_not_webm")
            if shutil.disk_usage(self.temp).free < MIN_FREE_BYTES + len(payload):
                raise RecordingStorageLimit("capture_disk_reserve")
            try:
                written = self.writer.write(payload)
                if written != len(payload):
                    raise OSError("partial_recording_write")
            except OSError:
                # Never retry writing the same sequence into a possibly
                # partially corrupted WebM stream after a disk write error.
                self.state = "failed"
                self.error_type = "OSError"
                self.writer.close()
                raise
            self.next_index += 1
            self.total += len(payload)
            self.last_accepted = (index, len(payload), digest)
            return {"state":"recording","chunks":self.next_index,
                    "replayed":False}

    def finish(self, offset_ms=0, caption_mode="all", region_mode="bottom40",
               detection_mode="edges"):
        # The original playback-position offset is committed with the first
        # successful finish and MUST match on an uncertain-ACK replay.
        if (type(offset_ms) is not int
                or not 0 <= offset_ms <= 99 * 3600 * 1000):
            raise ValueError("invalid_subtitle_time_offset")
        if type(caption_mode) is not str or caption_mode not in CAPTION_MODES:
            raise ValueError("invalid_caption_recognition_mode")
        if type(region_mode) is not str or region_mode not in CAPTION_REGIONS:
            raise ValueError("invalid_caption_region_mode")
        if type(detection_mode) is not str or detection_mode not in DETECTION_MODES:
            raise ValueError("invalid_caption_detection_mode")
        if detection_mode != "edges" and (
            region_mode != "lower65" or caption_mode not in ("all","islands")
        ):
            raise ValueError("unsupported_experimental_detection_combination")
        with self.lock:
            if self.state in ("processing", "completed"):
                if caption_mode != self.caption_mode:
                    raise ValueError("capture_finish_mode_replay_mismatch")
                if region_mode != self.region_mode:
                    raise ValueError("capture_finish_region_replay_mismatch")
                if detection_mode != self.detection_mode:
                    raise ValueError("capture_finish_detection_replay_mismatch")
                if offset_ms != self.output_offset_ms:
                    raise ValueError("capture_finish_offset_replay_mismatch")
                # The OCR worker is already owned by the first request.
                # An uncertain HTTP acknowledgement must never start a
                # second task or overwrite a previous SRT/ASS pair.
                return {"state":"processing", "replayed":True}
            if self.state != "recording" or self.next_index < 1 or self.total < 4096:
                raise RuntimeError("capture_not_complete")
            self.writer.flush()
            os.fsync(self.writer.fileno())
            self.writer.close()
            name = "hse-tab-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(6)
            base = self.output_dir / name
            if any(base.with_suffix(x).exists() or base.with_suffix(x).is_symlink() for x in (".srt",".ass")):
                raise FileExistsError("capture_output_collision")
            self.published_base = base
            self.output_offset_ms = offset_ms
            self.caption_mode = caption_mode
            self.region_mode = region_mode
            self.detection_mode = detection_mode
            self.state = "processing"
            self.worker = threading.Thread(target=self._process, args=(base,), daemon=True)
            self.worker.start()
            return {"state":"processing", "replayed":False}

    def _process(self, base):
        try:
            result = self.processor(self.video, base)
            count = int(result.cue_count)
            if count < 1 or any(not base.with_suffix(x).is_file() for x in (".srt",".ass")):
                raise RuntimeError("capture_subtitle_exports_incomplete")
            with self.lock:
                self.cue_count, self.state = count, "completed"
        except Exception as exc:
            with self.lock:
                self.error_type, self.state = type(exc).__name__, "failed"

    def request_stop(self):
        with self.lock:
            self.stop_requested=True

    def snapshot(self):
        with self.lock:
            return {"state":self.state,
                    "stop_requested":self.stop_requested,
                    "cue_count":self.cue_count if self.state=="completed" else 0,
                    "error_type":self.error_type if self.state=="failed" else None,
                    "ocr_progress":(
                        {"stage":self.ocr_stage,
                         "completed":self.ocr_completed,
                         "media_seconds":self.ocr_media_seconds}
                        if self.state=="processing" and self.ocr_stage else None),
                    "recording_bytes":self.total,
                    "recording_limit_bytes":MAX_TOTAL,
                    "last_chunk_index":(
                        self.last_accepted[0] if self.state=="recording"
                        and self.last_accepted is not None else None),
                    "last_chunk_size":(
                        self.last_accepted[1] if self.state=="recording"
                        and self.last_accepted is not None else None),
                    "last_chunk_sha256":(
                        self.last_accepted[2].hex() if self.state=="recording"
                        and self.last_accepted is not None else None),
                    "billable_api_calls":0, "subtitle_text_exported_to_browser":False}

    def local_result(self):
        # This includes paths only for the owner's local GUI callback.
        # Ordinary /status HTTP replies never expose filesystem paths.
        with self.lock:
            result={"state":self.state,
                    "cue_count":self.cue_count if self.state=="completed" else 0,
                    "error_type":self.error_type if self.state=="failed" else None}
            if self.state=="completed" and self.published_base is not None:
                result["srt_path"]=str(self.published_base.with_suffix(".srt"))
                result["ass_path"]=str(self.published_base.with_suffix(".ass"))
            return result

    def close(self):
        with self.lock:
            if self.state == "processing":
                raise RuntimeError("ocr_still_running")
            if not self.writer.closed:
                self.writer.close()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, session):
        super().__init__(("127.0.0.1",0),_Handler)
        self.session=session
        self.url="http://127.0.0.1:"+str(self.server_port)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self,*args):
        pass  # Never log user paths, raw media, screen text or URL query strings.

    def reply(self,obj,status=200):
        raw=json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(raw)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def good_host(self):
        return self.headers.get("Host") == "127.0.0.1:"+str(self.server.server_port)

    def authorized(self):
        return (self.good_host()
                and self.headers.get("X-HSE-Token")==self.server.session.token
                and self.headers.get("Origin") in (None,self.server.url))

    def do_GET(self):
        if not self.good_host():
            return self.reply({"error":"host_rejected"},403)
        if self.path=="/":
            s=self.server.session
            raw=PAGE.replace("__TOKEN__",json.dumps(s.token)).replace("__NONCE__",s.nonce).encode()
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(raw)))
            self.send_header("Cache-Control","no-store")
            self.send_header("Referrer-Policy","no-referrer")
            self.send_header("X-Frame-Options","DENY")
            self.send_header("Content-Security-Policy",
                "default-src 'none'; connect-src 'self'; script-src 'nonce-"+s.nonce+"'; style-src 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            return self.wfile.write(raw)
        if self.path=="/status" and self.authorized():
            return self.reply(self.server.session.snapshot())
        return self.reply({"error":"not_authorized"},403)

    def do_POST(self):
        # Even unauthorized HTTP/1.0 requests can carry a short body. On
        # Windows, replying without reading that body resets the connection
        # and hides the intended fail-closed 403/409 response from Chrome.
        # Read at most the explicit 8-MiB cap, then validate identity and
        # protocol before writing anything to the private recording.
        size=self.headers.get("Content-Length")
        if size is None or not size.isdecimal():
            self.close_connection=True
            return self.reply({"error":"length_required"},411)
        n=int(size)
        if n>MAX_CHUNK:
            self.close_connection=True
            return self.reply({"error":"chunk_too_large"},413)
        try:
            self.connection.settimeout(8)
            data=self.rfile.read(n)
        except (TimeoutError,OSError):
            self.close_connection=True
            return self.reply({"error":"incomplete_request"},408)
        if len(data)!=n:
            self.close_connection=True
            return self.reply({"error":"truncated_request"},400)
        if not self.authorized():
            return self.reply({"error":"not_authorized"},403)
        if self.headers.get("Content-Type")!="application/octet-stream":
            return self.reply({"error":"unexpected_content_type"},415)
        parsed=urlsplit(self.path)
        try:
            if parsed.path=="/chunk":
                parts=parse_qs(parsed.query,strict_parsing=True).get("i",[])
                if len(parts)!=1 or not parts[0].isdecimal() or n==0:
                    raise ValueError("bad_chunk_sequence")
                outcome=self.server.session.append(int(parts[0]),data)
            elif parsed.path=="/finish" and n==0:
                if parsed.query:
                    fields=parse_qs(parsed.query,strict_parsing=True,
                                    keep_blank_values=True)
                    if (set(fields) not in (
                            {"offset_ms"},
                            {"offset_ms","caption_mode"},
                            {"offset_ms","caption_mode","region_mode"},
                            {"offset_ms","caption_mode","region_mode","detection_mode"},
                            ) or len(fields["offset_ms"])!=1):
                        raise ValueError("invalid_time_offset_query")
                    value=fields["offset_ms"][0]
                    if (not value.isascii() or not value.isdecimal()
                            or len(value)>9):
                        raise ValueError("invalid_time_offset_query")
                    offset_ms=int(value)
                    modes=fields.get("caption_mode", ["all"])
                    if len(modes)!=1 or modes[0] not in CAPTION_MODES:
                        raise ValueError("invalid_caption_recognition_mode")
                    caption_mode=modes[0]
                    regions=fields.get("region_mode",["bottom40"])
                    if len(regions)!=1 or regions[0] not in CAPTION_REGIONS:
                        raise ValueError("invalid_caption_region_mode")
                    region_mode=regions[0]
                    detect=fields.get("detection_mode",["edges"])
                    if len(detect)!=1 or detect[0] not in DETECTION_MODES:
                        raise ValueError("invalid_caption_detection_mode")
                    detection_mode=detect[0]
                else:
                    offset_ms=0
                    caption_mode="all"
                    region_mode="bottom40"
                    detection_mode="edges"
                outcome=self.server.session.finish(
                    offset_ms, caption_mode, region_mode, detection_mode)
            elif parsed.path=="/close" and n==0:
                self.server.session.close()
                outcome={"state":"closed"}
                threading.Thread(target=self.server.shutdown,daemon=True).start()
            else:
                raise ValueError("invalid_operation")
            return self.reply(outcome)
        except RecordingStorageLimit:
            return self.reply({"error":"capture_disk_space_low"},507)
        except RecordingQuotaLimit:
            return self.reply({"error":"capture_quota_exceeded"},413)
        except (ValueError,RuntimeError,FileExistsError,OSError) as exc:
            return self.reply({"error_type":type(exc).__name__},409)


def serve(output_dir: Path, open_browser=True, *, ocr_language="ko",
          stop_event=None, on_ready=None, on_finished=None):
    """User-approved local tab recording. The caller owns any stop event."""
    from hardsub_extractor.model_assets import normalize_ocr_language
    language = normalize_ocr_language(ocr_language)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("choose_existing_private_output_directory")
    with tempfile.TemporaryDirectory(prefix="hse-private-tab-capture-") as tmp:
        def process_tab(video, base):
            extras = {}
            if session.output_offset_ms:
                extras["output_time_offset_ms"]=session.output_offset_ms
            if session.caption_mode!="all":
                extras["caption_mode"]=session.caption_mode
            if session.region_mode!="bottom40":
                extras["region_mode"]=session.region_mode
            if session.detection_mode!="edges":
                extras["detection_mode"]=session.detection_mode
            return offline_ocr(video, base, language=language,
                               on_progress=session.mark_progress, **extras)
        session=CaptureSession(Path(tmp), output_dir, process_tab)
        server=_Server(session)
        # Enter the serving loop before the GUI can request shutdown.
        # BaseServer.shutdown() must not race a server that has not started;
        # the caller's stop_event may already be set at launch time.
        serving=threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval":.2}, daemon=True,
        )
        try:
            if stop_event is None or not stop_event.is_set():
                serving.start()
                if on_ready is not None:
                    on_ready(server.url+"/")
                if open_browser:
                    webbrowser.open(server.url+"/")
                started=None
                terminal=None
                while serving.is_alive():
                    # A one-recording helper has no further work after
                    # completed/failed. Release the GUI and its server
                    # automatically; never tear down a running OCR worker.
                    requested=(stop_event.wait(.15) if stop_event is not None
                               else False)
                    if stop_event is None:
                        time.sleep(.15)
                    now=time.monotonic()
                    if requested and started is None:
                        session.request_stop()
                        started=now
                    state=session.snapshot()["state"]
                    if state in ("completed","failed"):
                        if terminal is None:
                            terminal=now
                        # Allow status polls to show the final SRT/ASS result.
                        # Explicit GUI stop takes the original shorter grace.
                        if now-terminal<(2.5 if started is not None else 8.0):
                            continue
                        server.shutdown()
                        break
                    if started is None:
                        # No user request yet. Keep capture and OCR alive.
                        continue
                    if state=="recording" and now-started<9:
                        continue
                    if state=="processing":
                        # Never delete temporary video while OCR is running.
                        continue
                    server.shutdown()
                    break
                serving.join()
        finally:
            if serving.is_alive():
                server.shutdown()
                serving.join()
            server.server_close()
            if session.worker is not None:
                # Never destroy the temporary video before OCR has finished.
                session.worker.join()
            result=session.local_result()
            session.close()
            if on_finished is not None:
                on_finished(result)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--no-open-browser",action="store_true")
    p.add_argument("--ocr-language",choices=("ko","en","ja","zh"),default="ko")
    args=p.parse_args(argv)
    try:
        serve(args.output_dir,open_browser=not args.no_open_browser,
              ocr_language=args.ocr_language)
    except (ValueError,OSError,RuntimeError) as exc:
        print("HSE_TAB_CAPTURE_ERROR="+type(exc).__name__,flush=True)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
