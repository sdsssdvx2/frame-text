// Execute the exact browser-control script from the production research module.
// Pure Node built-ins: no Chrome, network, media, OCR, user data or dependencies.
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=readFileSync(path.join(root,'src','hardsub_extractor','tab_capture.py'),'utf8');
const html=source.match(/PAGE = r'''([\s\S]*?)'''/);
assert.ok(html,'existing HSE browser HTML was not found');
const inline=html[1].match(/<script nonce="__NONCE__">([\s\S]*?)<\/script>/);
assert.ok(inline,'existing HSE inline control script was not found');
const script=inline[1].replace('__TOKEN__',JSON.stringify('test-only-secret'));
assert.ok(script.includes('rec.ondataavailable='),'real MediaRecorder callback absent');

function setup(offset='00:00:00.000',{dropAckIndex=-1,mode='all',failStatus=false,
  duration='00:00:00',region='bottom40',detection='edges'}={}){
  const elements=new Map();
  for(const name of ['start','stop','close','offset','duration','caption-mode','region-mode','detection-mode','status'])
    elements.set(name,{disabled:name==='stop'||name==='close',textContent:'',
                       value:name==='offset'?offset:name==='duration'?duration:name==='caption-mode'?mode:name==='region-mode'?region:name==='detection-mode'?detection:''});
  const written=[];
  const attempts=[];
  let finished=false, recorder=null,clock=500;
  const track={stopped:false,onended:null,stop(){this.stopped=true;}};
  const media={getVideoTracks(){return [track];},getTracks(){return [track];}};
  class Recorder{
    constructor(){this.state='inactive';recorder=this;}
    static isTypeSupported(mime){return mime.startsWith('video/webm');}
    start(){this.state='recording';}
    stop(){
      if(this.state==='inactive')return;
      this.state='inactive';
      this.completion=this.onstop();
    }
  }
  async function fetch(url,options={}){
    attempts.push(url);
    if(url.startsWith('/chunk?i=')){
      const i=Number(url.slice('/chunk?i='.length));
      assert.ok(i===written.length || i===written.length-1,
                'chunk index/order must be exact or last identical retry');
      const b=options.body;
      assert.ok(b instanceof Blob,'send the real Blob slice');
      assert.ok(b.size>0 && b.size<=8388608,'server 8 MiB cap');
      const payload=Buffer.from(await b.arrayBuffer());
      if(i===written.length)written.push(payload);
      else assert.ok(written[i].equals(payload),'replayed bytes must match');
      if(i===dropAckIndex)throw Error('Windows lost POST ACK after commit');
      return {ok:true,json:async()=>({state:'recording',chunks:written.length})};
    }
    if(url.startsWith('/finish?offset_ms=')){
      assert.ok(written.length>0,'never finish empty recording');
      assert.equal(url,'/finish?offset_ms='+(
        offset==='00:20:01.234'?'1201234':'0')+
        '&caption_mode='+mode+'&region_mode='+region+'&detection_mode='+detection);
      finished=true;
      return {ok:true,json:async()=>({state:'processing'})};
    }
    if(url==='/status'){
      if(failStatus)throw Error('transient_windows_localhost_GET_reset');
      const last=written.at(-1);
      return {ok:true,json:async()=>({state:finished?'completed':'recording',
          cue_count:3,recording_bytes:written.reduce((n,b)=>n+b.length,0),
          recording_limit_bytes:8589934592,
          last_chunk_index:last?written.length-1:null,
          last_chunk_size:last?.length??null,
          last_chunk_sha256:last?createHash('sha256').update(last).digest('hex'):null})};
    }
    throw Error('Unexpected local host request');
  }
  const context=vm.createContext({
    Blob,Uint8Array,crypto:webcrypto,fetch,MediaRecorder:Recorder,
    performance:{now:()=>clock},
    navigator:{mediaDevices:{getDisplayMedia:async()=>media}},
    document:{getElementById:id=>elements.get(id)},
    setTimeout,clearTimeout,setInterval:()=>1,
    AbortController,
  });
  vm.runInContext(script,context,{timeout:3000});
  return {context,elements,media,track,written,attempts,
          advanceMillis(ms){clock+=ms;},
          get recorder(){return recorder;},
          get finished(){return finished;}};
}

async function largeBlobPreservesWebmAndOffset(){
  const h=setup('00:20:01.234',{mode:'light-temporal',region:'lower65'});
  await h.elements.get('start').onclick();
  assert.equal(h.recorder.state,'recording');
  // Include EBML header, verify no re-encoding and no missing/duplicated byte.
  const bytes=Buffer.alloc(8388608+1027);
  Buffer.from('1a45dfa3','hex').copy(bytes,0);
  for(let i=4;i<bytes.length;i++)bytes[i]=i%251;
  h.recorder.ondataavailable({data:new Blob([bytes])});
  h.elements.get('stop').onclick();
  await h.recorder.completion;
  assert.equal(h.written.length,2,'delayed >8 MiB Blob must be split');
  assert.deepEqual(h.written.map(x=>x.length),[8388608,1027]);
  assert.ok(Buffer.concat(h.written).equals(bytes),'WebM bytes must be identical');
  assert.ok(h.finished,'finish once, after all parts');
  assert.equal(h.attempts.filter(x=>x.startsWith('/finish')).length,1);
  assert.equal(vm.runInContext('queuedBytes',h.context),0);
  assert.equal(vm.runInContext('seq',h.context),2);
  assert.equal(vm.runInContext('failed',h.context),false);
  assert.equal(vm.runInContext('captionMode',h.context),'light-temporal');
  assert.equal(vm.runInContext('regionMode',h.context),'lower65');
  assert.equal(vm.runInContext('detectionMode',h.context),'edges');
  assert.match(h.elements.get('status').textContent,/생성 완료/);
  assert.ok(h.track.stopped,'tab share permission must be released');
}

async function extremeBacklogFailsClosed(){
  const h=setup();
  await h.elements.get('start').onclick();
  // No 128 MiB allocation needed: reject oversized blob by metadata alone.
  h.recorder.ondataavailable({data:{
    size:134217729,
    slice(){throw Error('oversized blob should never be sliced');}
  }});
  assert.equal(vm.runInContext('failed',h.context),true);
  assert.match(h.elements.get('status').textContent,/버퍼 초과/);
  assert.equal(h.written.length,0);
  assert.equal(h.finished,false,'no OCR finish after unsafe backlog');
  assert.ok(h.track.stopped,'screen share released on failure');
}

async function lostPostAckRecoveredByHash(){
  const h=setup('00:00:00.000',{dropAckIndex:0});
  await h.elements.get('start').onclick();
  const bytes=Buffer.alloc(16384,42);
  Buffer.from('1a45dfa3','hex').copy(bytes,0);
  h.recorder.ondataavailable({data:new Blob([bytes])});
  h.elements.get('stop').onclick();
  await h.recorder.completion;
  assert.equal(h.attempts.filter(x=>x==='/chunk?i=0').length,3);
  assert.ok(h.attempts.includes('/status'),'exact committed SHA checked on GET');
  assert.equal(h.written.length,1,'no duplicate bytes after 3 lost ACKs');
  assert.ok(h.written[0].equals(bytes));
  assert.ok(h.finished,'verified persisted part permits exact finish');
  assert.equal(vm.runInContext('failed',h.context),false);
  assert.equal(vm.runInContext('queuedBytes',h.context),0);
}

async function statusResetsWithAcknowledgedChunksRemainBounded(){
  const h=setup('00:00:00.000',{failStatus:true});
  await h.elements.get('start').onclick();
  const bytes=Buffer.alloc(16384,43);
  Buffer.from('1a45dfa3','hex').copy(bytes,0);
  h.recorder.ondataavailable({data:new Blob([bytes])});
  await vm.runInContext('queue',h.context);
  assert.equal(h.written.length,1);
  // Multiple failed /status requests alone must not end an otherwise
  // healthy stream while authenticated WebM POSTs still succeed.
  for(let n=0;n<4;n++)await vm.runInContext('heartbeat()',h.context);
  assert.equal(vm.runInContext('failed',h.context),false);
  assert.equal(h.track.stopped,false);
  assert.ok(Date.now()-vm.runInContext('lastHostProofMs',h.context)<10000);
  // A genuinely absent local helper must still release screen capture.
  vm.runInContext('lastHostProofMs=Date.now()-11000',h.context);
  await vm.runInContext('heartbeat()',h.context);
  assert.equal(vm.runInContext('failed',h.context),true);
  assert.equal(h.track.stopped,true);
  assert.equal(h.finished,false);
}

async function optionalAutoDurationStopsOnlyAtMonotonicDeadline(){
  const h=setup('00:00:00.000',{duration:'00:00:20'});
  await h.elements.get('start').onclick();
  assert.equal(h.recorder.state,'recording');
  assert.equal(vm.runInContext('autoDurationMillis',h.context),20000);
  assert.equal(h.elements.get('duration').disabled,true);
  const data=Buffer.alloc(16384,27);
  Buffer.from('1a45dfa3','hex').copy(data,0);
  h.recorder.ondataavailable({data:new Blob([data])});
  h.advanceMillis(19999);
  assert.equal(vm.runInContext('maybeStopAtDeadline()',h.context),false);
  assert.equal(h.recorder.state,'recording');
  h.advanceMillis(1);
  assert.equal(vm.runInContext('maybeStopAtDeadline()',h.context),true);
  assert.equal(h.recorder.state,'inactive');
  await h.recorder.completion;
  assert.ok(h.finished,'automatic stop must publish OCR output once');
  assert.equal(h.written.length,1);
  assert.ok(h.written[0].equals(data),'WebM bytes identical');
  assert.equal(h.attempts.filter(x=>x.startsWith('/finish?')).length,1);
  assert.equal(vm.runInContext('maybeStopAtDeadline()',h.context),false);
  assert.ok(h.track.stopped);
}
async function invalidAutoDurationNeverRequestsScreen(){
  for(const duration of ['00:00:14','04:00:01','99:99:99','garbled']){
    const h=setup('00:00:00.000',{duration});
    await h.elements.get('start').onclick();
    assert.equal(h.recorder,null,'bad timer must not request media');
    assert.equal(h.elements.get('start').disabled,false);
    assert.equal(h.written.length,0);
    assert.equal(h.finished,false);
  }
}
async function invalidRegionNeverRequestsScreen(){
  const h=setup('00:00:00.000',{region:'unsupported'});
  await h.elements.get('start').onclick();
  assert.equal(h.recorder,null);
  assert.equal(h.written.length,0);
  assert.equal(h.finished,false);
  assert.equal(h.elements.get('start').disabled,false);
  assert.match(h.elements.get('status').textContent,/지원되지 않는 자막 인식 영역/);
}
async function explicitPersistentWhiteOnlyWithMatchingROIAndOCR(){
  const h=setup('00:00:00.000',{region:'lower65',mode:'all',
    detection:'persistent-white-3'});
  await h.elements.get('start').onclick();
  assert.equal(h.recorder.state,'recording');
  assert.equal(h.elements.get('detection-mode').disabled,true);
  assert.equal(vm.runInContext('detectionMode',h.context),'persistent-white-3');
  const bytes=Buffer.alloc(16384,51);
  Buffer.from('1a45dfa3','hex').copy(bytes,0);
  h.recorder.ondataavailable({data:new Blob([bytes])});
  h.elements.get('stop').onclick();
  await h.recorder.completion;
  assert.ok(h.finished);
  assert.ok(h.attempts.includes(
    '/finish?offset_ms=0&caption_mode=all&region_mode=lower65&detection_mode=persistent-white-3'));
  assert.ok(h.track.stopped);
}
async function optionalCaptionIslandAndBrightDetection(){
  const h=setup('00:00:00.000',{mode:'islands',region:'lower65',
    detection:'persistent-white-3'});
  await h.elements.get('start').onclick();
  assert.equal(h.recorder.state,'recording');
  assert.equal(h.elements.get('caption-mode').disabled,true);
  const bytes=Buffer.alloc(16384,61);
  Buffer.from('1a45dfa3','hex').copy(bytes,0);
  h.recorder.ondataavailable({data:new Blob([bytes])});
  h.elements.get('stop').onclick();
  await h.recorder.completion;
  assert.ok(h.finished);
  assert.ok(h.attempts.includes(
    '/finish?offset_ms=0&caption_mode=islands&region_mode=lower65&detection_mode=persistent-white-3'));
  assert.ok(h.track.stopped);
}
async function invalidPersistentWhiteOptionNeverSharesScreen(){
  for(const options of [
    {detection:'invalid'},
    {detection:'persistent-white-3',region:'bottom40'},
    {detection:'persistent-white-3',region:'lower80'},
    {detection:'persistent-white-3',region:'lower65',mode:'light'},
    {detection:'persistent-white-3',region:'lower65',mode:'light-temporal'},
  ]){
    const h=setup('00:00:00.000',options);
    await h.elements.get('start').onclick();
    assert.equal(h.recorder,null,'unsafe option must not trigger tab share');
    assert.equal(h.finished,false);
    assert.equal(h.written.length,0);
    assert.equal(h.elements.get('start').disabled,false);
    assert.match(h.elements.get('status').textContent,
      /지원되지 않는 자막 감지|하단 65% \+ 기본 OCR/);
  }
}
await explicitPersistentWhiteOnlyWithMatchingROIAndOCR();
await optionalCaptionIslandAndBrightDetection();
await invalidPersistentWhiteOptionNeverSharesScreen();
await largeBlobPreservesWebmAndOffset();
await extremeBacklogFailsClosed();
await lostPostAckRecoveredByHash();
await statusResetsWithAcknowledgedChunksRemainBounded();
await optionalAutoDurationStopsOnlyAtMonotonicDeadline();
await invalidAutoDurationNeverRequestsScreen();
await invalidRegionNeverRequestsScreen();
console.log('HSE_REAL_PAGE_BLOB_SPLIT=PASS;WEBM_BYTES_IDENTICAL=YES;LOST_ACK_SHA_RECOVERY=PASS;AUTO_DURATION_MONOTONIC=PASS;AUTO_DURATION_FAIL_CLOSED=PASS;REGION_MODE_PINNED=PASS;REGION_BAD_INPUT_BLOCKED=PASS;PERSISTENT_WHITE_OPT_IN=PASS;CAPTION_ISLANDS_OPT_IN=PASS;INVALID_DETECTION_NO_SCREEN=PASS;TRANSIENT_GET_HEARTBEAT=PASS;BOUNDED_DISCONNECT_STOP=PASS;OFFSET_PINNED=YES;BUFFER_FAIL_CLOSED=YES;LIVE_CHROME=NOT_TESTED');
