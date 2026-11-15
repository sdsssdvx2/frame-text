"""Isolated licensed-film JA/ZH subtitle image benchmark, research only.

A deterministic 24-image CC0-annotation sample from GeekLinkDev's film-frame
dataset (real film backgrounds, synthetic burned text), half with nearby
watermarks. No images/ground truths are written to the repository or logs.
Not a time-coded video, natural Japanese broadcaster or release quality gate.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen
import cv2
import numpy as np
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key
from hardsub_extractor.tab_capture import roi_for_region

SOURCE_REV="baa9ccbb7ffad36d56cc0ea115256a09bdf427f4"
MANIFEST_BLOB="c7b8ca725bd97b1e0a1e918af957febe69df09ea"
URL_BASE="https://raw.githubusercontent.com/GeekLinkDev/geeklink-ocr-benchmark/"
PINS_TXT="""\
v01_0140 136f6a454af5ff2d37e14f15f7bfb8860575976d 61188
v01_0270 f767cd9f1231f7b8b09cc2d079865f618ae6d4fd 78499
v01_0146 7ef596938e73480a03287d4ea85067183efbcea4 162042
v01_0294 257e4b5363d18c32cd9195e0951bf7acbe067d2a 98347
v01_0148 f6c2449fcc7810cd5648a57dca1c27f9a64f706e 131610
v01_0284 7d392a47eec6dc107db978d4f3e831f8503a09ed 73362
v01_0094 2acc10911297ef8e1401600da1184def632cbaf2 72562
v01_0266 c713f44bbc81df3c337963f785e27f80d4b5283a 62053
v02_0128 81ce6224fc92173ab3f73d505f05e5acd9320474 24986
v02_0304 c8e1923b5cfa59797d996f31a1f3d55feb4b7c3c 28037
v02_0164 dd99a6f6855f49df542e5f0ea6cc6c57bdecee24 45141
v02_0268 019223428876987854a1bc917386a0279c5dc1f5 24026
v02_0142 fc55f5191114b19b1e13b1b89353f708e9a07192 17132
v02_0276 bedc5b94a4c6c191ab040c555722d4ac3d9812a4 28439
v02_0124 a1157048ced509d4679ed2474266c4ccd152df05 29694
v02_0306 e4d766a36bf17cf4c5e9da0151a3d34c90037098 49396
v03_0098 b6e321a6fb223c3543f94ee4d3ada3c6314cb428 45229
v03_0194 2a8ae926dbaf2631a5a889f3f0f3acec641c6869 34477
v03_0110 e8d702ffaa449be728a85108104135b1c8f609f7 47736
v03_0224 3b362d2a5932c97214af355d15076b4c27e6db70 64107
v03_0100 0742420a963142a1e99688d708a161de48083432 33867
v03_0190 55da07f12f2706f249f25bf457f9f8d6ec96dc4c 34244
v03_0166 7365982361ad2dc977bebd2585bd12d0bc713581 42852
v03_0262 1d24790ef3136cda39def0449e7f407a0b807c08 52002
"""
PINS={x.split()[0]:(x.split()[1],int(x.split()[2]))
      for x in PINS_TXT.strip().splitlines()}


def object_sha(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def fetch_checked(relative,expected_sha,maximum,expected_length=None):
    if (not relative or ".." in relative or "\\" in relative
        or relative.startswith("/")
        or not all(c.isascii() and (c.isalnum() or c in "/._-")
                   for c in relative)):
        raise ValueError("invalid public path")
    url=URL_BASE+SOURCE_REV+"/"+relative
    with urlopen(Request(url,headers={
        "User-Agent":"HSE-Pinned-Licensed-CJK-Quality/1"}),timeout=25) as resp:
        if resp.status!=200 or resp.geturl()!=url:
            raise ValueError("public source origin changed")
        data=resp.read(maximum+1)
    if (not data or len(data)>maximum or
        expected_length is not None and len(data)!=expected_length or
        object_sha(data)!=expected_sha):
        raise ValueError("public source pin mismatch")
    return data


def chosen_rows(blob):
    rows=[json.loads(x) for x in blob.decode("utf-8").splitlines() if x.strip()]
    if len(rows)!=1140 or len(PINS)!=24:
        raise ValueError("unexpected public corpus size")
    groups=defaultdict(list)
    for r in rows:
        if r["lang"] in ("ja","zh"):
            wm=str(r["has_watermark"])
            if wm not in ("True","False"):
                raise ValueError("unexpected watermark marker")
            groups[(r["video_id"],r["lang"],wm)].append(r)
    if len(groups)!=12 or any(len(a)<3 for a in groups.values()):
        raise ValueError("public strata changed")
    selected=[]
    for key,items in sorted(groups.items()):
        for n in (len(items)//3,2*len(items)//3):
            row=items[n]
            sid=row["id"]
            if (sid not in PINS
                or row["image"]!="data/images/"+sid+".jpg"
                or not isinstance(row["ground_truth"],str)
                or not row["ground_truth"].strip()):
                raise ValueError("stratified source pin mismatch")
            selected.append(row)
    if set(PINS)!={row["id"] for row in selected}:
        raise ValueError("selection changed")
    return selected


def distance(a,b):
    if len(a)<len(b):a,b=b,a
    prev=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        cur=[i]+[0]*len(b)
        for j,d in enumerate(b,1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(c!=d))
        prev=cur
    return prev[-1]


def score(image,ref,engine,roi):
    height,width=image.shape[:2]
    x,y,w,h=roi_for_region(roi).pixels(width,height)
    before=engine.total_calls
    tic=time.perf_counter()
    pred=engine.recognize(Frame(0.0,np.ascontiguousarray(
        image[y:y+h,x:x+w]))).text.replace("\n","")
    seconds=time.perf_counter()-tic
    refnorm=comparison_key(ref);prednorm=comparison_key(pred)
    return (distance(pred,ref),max(1,len(ref)),
            distance(prednorm,refnorm),max(1,len(refnorm)),
            int(pred==ref),int(bool(pred.strip())),
            engine.total_calls-before,seconds)


def measure(model_dir,manifest):
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("local model folder missing")
    rows=chosen_rows(fetch_checked("data/manifest.jsonl",MANIFEST_BLOB,250000))
    engines={lang:AdaptiveOCREngine(RapidOCRAdapter(
        language=lang,model_dir=str(model_dir),manifest_path=str(manifest)))
        for lang in ("ja","zh")}
    buckets=defaultdict(list);total_bytes=0
    for row in rows:
        pin,size=PINS[row["id"]]
        raw=fetch_checked(row["image"],pin,300000,size)
        total_bytes+=len(raw)
        image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
        if (image is None or image.ndim!=3 or image.shape[2]!=3
            or not 240<=image.shape[0]<=3000
            or not 240<=image.shape[1]<=4000):
            raise ValueError("pinned public image invalid")
        for roi in ("bottom40","lower65"):
            bucket=(row["lang"],str(row["has_watermark"])=="True",roi)
            buckets[bucket].append(score(
                image,row["ground_truth"],engines[row["lang"]],roi))
        del raw,image
    output=[]
    for (lang,wm,roi),data in sorted(buckets.items()):
        if len(data)!=6:raise RuntimeError("stratified CJK study incomplete")
        output.append({
            "language":lang,"watermark_near_caption":wm,
            "roi":roi,"images":len(data),
            "literal_CER":round(sum(x[0] for x in data)/sum(x[1] for x in data),4),
            "normalized_CER":round(sum(x[2] for x in data)/sum(x[3] for x in data),4),
            "exact_strings":sum(x[4] for x in data),
            "nonempty":sum(x[5] for x in data),
            "inference_calls":sum(x[6] for x in data),
            "inference_seconds":round(sum(x[7] for x in data),2),
        })
    if len(output)!=8:raise RuntimeError("missing CJK benchmark bucket")
    return {
        "source":"licensed_real_film_backgrounds_with_constructed_CJK_subtitles",
        "revision":SOURCE_REV,"verified_image_bytes":total_bytes,
        "independent_image_count":len(rows),
        "model_profiles":"pinned_PP_OCRv5_CJK",
        "still_image_CER_only":True,
        "original_native_video_quality_not_proven":True,
        "license_note":"CC0_annotations_and_mixed_PD_CC_BY_SA_frames",
        "release":"HOLD","result":output,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--accept-public-download",action="store_true")
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    if a.self_test:
        assert len(PINS)==24
        assert object_sha(b"test")=="30d74d258442c7c65512eafab474568dd706c430"
        assert distance("abc","adc")==1
        assert distance("漢字","汉字")==1
        print("HSE_CJK_FILM_FRAMES_CONTRACT=PASS;NETWORK=NO;MODELS=NO")
        return 0
    if (not a.accept_public_download or a.model_dir is None
        or a.manifest is None):
        print("HSE_CJK_FILM_FRAMES=NOT_APPROVED;NETWORK=OFF")
        return 2
    try:
        print("HSE_CJK_FILM_FRAMES_MEASURED "+json.dumps(
            measure(a.model_dir,a.manifest),sort_keys=True),flush=True)
        print("HSE_CJK_FILM_FRAMES_RESULT=MEASURED_NOT_RELEASED;"
              "PRODUCTION_CHANGED=NO;USER_MEDIA=NO;PAID_API=NO",flush=True)
        return 0
    except Exception as e:
        print("HSE_CJK_FILM_FRAMES_ERROR "+json.dumps({
            "error_class":type(e).__name__,
            "raw_image_or_text_logged":False,
            "production_changed":False,"release":"HOLD",
        },sort_keys=True),flush=True)
        return 3


if __name__=="__main__":
    raise SystemExit(main())
