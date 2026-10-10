/* Roger Spark WebGPU inference engine — custom model, quantized weights. */
const $ = (s) => document.querySelector(s);
const align4 = (n) => (n + 3) & ~3;

const SHADERS = {
  linear: `
struct P { outSize:u32, inSize:u32, scale:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read> W:array<u32>;
@group(0) @binding(1) var<storage, read> X:array<f32>;
@group(0) @binding(2) var<storage, read_write> Y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
fn wi(i:u32)->f32 { let w=W[i>>2u]; let b=(w>>((i&3u)*8u))&255u; return f32(bitcast<i32>(b<<24u)>>24); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let o=gid.x; if(o>=p.outSize){return;} var sum=0.0;
 for(var i=0u;i<p.inSize;i++){sum += wi(o*p.inSize+i)*p.scale*X[i];} Y[o]=sum;
}`,
  embedding: `
struct P { row:u32, width:u32, scale:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read> W:array<u32>;
@group(0) @binding(1) var<storage, read> X:array<f32>;
@group(0) @binding(2) var<storage, read_write> Y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
fn wi(i:u32)->f32 { let w=W[i>>2u]; let b=(w>>((i&3u)*8u))&255u; return f32(bitcast<i32>(b<<24u)>>24); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x; if(i>=p.width){return;} Y[i]=X[i]+wi(p.row*p.width+i)*p.scale;
}`,
  norm: `
struct P { width:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> X:array<f32>;
@group(0) @binding(1) var<storage, read> G:array<u32>;
@group(0) @binding(2) var<storage, read> B:array<u32>;
@group(0) @binding(3) var<storage, read_write> Y:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
fn halfg(i:u32)->f32 { let z=G[i>>1u]; let h=select(z&65535u,z>>16u,(i&1u)==1u); return unpack2x16float(select(z,z>>16u,(i&1u)==1u)).x; }
fn halfb(i:u32)->f32 { let z=B[i>>1u]; let h=select(z&65535u,z>>16u,(i&1u)==1u); return unpack2x16float(h).x; }
@compute @workgroup_size(1) fn main() {
 var mean=0.0; for(var i=0u;i<p.width;i++){mean+=X[i];} mean/=f32(p.width);
 var variance=0.0; for(var i=0u;i<p.width;i++){let d=X[i]-mean; variance+=d*d;} variance/=f32(p.width);
 let inv=inverseSqrt(variance+0.00001);
 for(var i=0u;i<p.width;i++){Y[i]=(X[i]-mean)*inv*halfg(i)+halfb(i);}
}`,
  split: `
struct P { width:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> X:array<f32>;
@group(0) @binding(1) var<storage, read_write> Q:array<f32>;
@group(0) @binding(2) var<storage, read_write> K:array<f32>;
@group(0) @binding(3) var<storage, read_write> V:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p.width){Q[i]=X[i];K[i]=X[p.width+i];V[i]=X[2u*p.width+i];}}
`,
  cache: `
struct P { position:u32, width:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> K:array<f32>;
@group(0) @binding(1) var<storage, read> V:array<f32>;
@group(0) @binding(2) var<storage, read_write> KC:array<f32>;
@group(0) @binding(3) var<storage, read_write> VC:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p.width){let at=p.position*p.width+i;KC[at]=K[i];VC[at]=V[i];}}
`,
  scores: `
struct P { position:u32, context:u32, width:u32, headDim:u32 };
@group(0) @binding(0) var<storage, read> Q:array<f32>;
@group(0) @binding(1) var<storage, read> KC:array<f32>;
@group(0) @binding(2) var<storage, read_write> S:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let id=gid.x; let heads=p.width/p.headDim; if(id>=heads*p.context){return;}
 let head=id/p.context; let t=id%p.context; if(t>p.position){S[id]=-1e30;return;}
 var sum=0.0; for(var d=0u;d<p.headDim;d++){sum+=Q[head*p.headDim+d]*KC[t*p.width+head*p.headDim+d];}
 S[id]=sum*inverseSqrt(f32(p.headDim));
}`,
  attend: `
struct P { position:u32, context:u32, width:u32, headDim:u32 };
@group(0) @binding(0) var<storage, read> S:array<f32>;
@group(0) @binding(1) var<storage, read> VC:array<f32>;
@group(0) @binding(2) var<storage, read_write> Y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=p.width){return;} let head=i/p.headDim; let d=i%p.headDim; let base=head*p.context;
 var mx=-1e30;for(var t=0u;t<=p.position;t++){mx=max(mx,S[base+t]);}
 var denom=0.0;for(var t=0u;t<=p.position;t++){denom+=exp(S[base+t]-mx);}
 var val=0.0;for(var t=0u;t<=p.position;t++){val+=exp(S[base+t]-mx)/denom*VC[t*p.width+i];}
 Y[i]=val;
}`,
  add: `
struct P { width:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> A:array<f32>;
@group(0) @binding(1) var<storage, read> B:array<f32>;
@group(0) @binding(2) var<storage, read_write> Y:array<f32>;
@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p.width){Y[i]=A[i]+B[i];}}
`,
  gelu: `
struct P { width:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> X:array<f32>;
@group(0) @binding(1) var<storage, read_write> Y:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i<p.width){let x=X[i];let z=0.79788456*(x+0.044715*x*x*x); let t=2.0/(1.0+exp(-2.0*z))-1.0; Y[i]=0.5*x*(1.0+t);}}
`,
  sample: `
struct P { row:u32, width:u32, scale:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read> W:array<u32>;
@group(0) @binding(1) var<storage, read_write> Y:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
fn wi(i:u32)->f32 { let w=W[i>>2u]; let b=(w>>((i&3u)*8u))&255u; return f32(bitcast<i32>(b<<24u)>>24); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let o=gid.x;if(o>=p.width){return;}var s=0.0;for(var i=0u;i<p.row;i++){s+=wi(o*p.row+i)*p.scale*0.0;}Y[o]=s;}
`
};

class ByteLevelTokenizer {
  constructor(json) {
    this.vocab = json.model.vocab;
    this.idToToken = [];
    for (const [token, id] of Object.entries(this.vocab)) this.idToToken[id] = token;
    // Match registered special tokens before applying ordinary byte-level BPE.
    this.specialTokens = (json.added_tokens || []).filter(item => item.special).map(item => item.content).sort((a,b) => b.length-a.length);
    this.specialTokenIds = new Set(this.specialTokens.map(token => this.vocab[token]));
    this.ranks = new Map();
    (json.model.merges || []).forEach((pair, i) => {
      const p = Array.isArray(pair) ? pair : pair.split(" ");
      this.ranks.set(p[0] + "\u0000" + p[1], i);
    });
    this.byteEncoder = new Map(); this.byteDecoder = new Map();
    const bs = [];
    for(let b=33;b<=126;b++)bs.push(b);
    for(let b=161;b<=172;b++)bs.push(b);
    for(let b=174;b<=255;b++)bs.push(b);
    const cs = bs.slice();
    let n=0;
    for(let b=0;b<256;b++){if(!bs.includes(b)){bs.push(b);cs.push(256+n++);}}
    for(let i=0;i<bs.length;i++){const ch=String.fromCodePoint(cs[i]);this.byteEncoder.set(bs[i],ch);this.byteDecoder.set(ch,bs[i]);}
    this.pattern = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.cache = new Map();
  }
  bpe(piece) {
    if(this.cache.has(piece))return this.cache.get(piece);
    let parts=Array.from(piece);
    while(parts.length>1){
      let best=-1, at=-1;
      for(let i=0;i<parts.length-1;i++){const rank=this.ranks.get(parts[i]+"\u0000"+parts[i+1]);if(rank!==undefined&&(best<0||rank<best)){best=rank;at=i;}}
      if(at<0)break;
      parts.splice(at,2,parts[at]+parts[at+1]);
    }
    this.cache.set(piece,parts);return parts;
  }
  encode(text) {
    const ids=[];
    let cursor=0;
    while(cursor<text.length){
      let nextAt=-1,nextToken=null;
      for(const token of this.specialTokens){
        const at=text.indexOf(token,cursor);
        if(at>=0&&(nextAt<0||at<nextAt||(at===nextAt&&token.length>nextToken.length))){nextAt=at;nextToken=token;}
      }
      const end=nextAt<0?text.length:nextAt;
      this.encodeOrdinary(text.slice(cursor,end),ids);
      if(nextAt<0)break;
      ids.push(this.vocab[nextToken]);
      cursor=nextAt+nextToken.length;
    }
    return ids;
  }
  encodeOrdinary(text,ids) {
    for(const match of text.matchAll(this.pattern)){
      const bytes=new TextEncoder().encode(match[0]);
      let piece="";for(const b of bytes)piece+=this.byteEncoder.get(b);
      for(const token of this.bpe(piece)){const id=this.vocab[token];if(id===undefined)throw new Error("Tokenizer is missing token: "+token);ids.push(id);}
    }
  }
  decode(ids) {
    const bytes=[];
    for(const id of ids){const token=this.idToToken[id];if(!token||/^\[.*\]$/.test(token))continue;for(const ch of token){const b=this.byteDecoder.get(ch);if(b!==undefined)bytes.push(b);}}
    return new TextDecoder("utf-8",{fatal:false}).decode(new Uint8Array(bytes));
  }
}

export class RogerSparkWebGPU {
  constructor(device, manifest, weights, tokenizer) {
    this.device=device;this.manifest=manifest;this.config=manifest.config;this.tokenizer=tokenizer;
    this.weights=new Map();this.buffers=[];this.temps=[];
    const bin=weights;
    for(const [name,t] of Object.entries(manifest.tensors)){
      const slice=bin.slice(t.offset,t.offset+t.nbytes);
      const buffer=device.createBuffer({size:Math.max(4,align4(slice.byteLength)),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(buffer,0,slice);
      this.weights.set(name,{buffer,...t});this.buffers.push(buffer);
    }
    this.pipelines={};
    for(const [name,code] of Object.entries(SHADERS)){
      if(name==="sample")continue;
      this.pipelines[name]=device.createComputePipeline({layout:"auto",compute:{module:device.createShaderModule({code}),entryPoint:"main"}});
    }
    this.context=this.config.context_length;this.width=this.config.d_model;this.headDim=this.width/this.config.n_heads;
    this.cacheK=[];this.cacheV=[];
    for(let i=0;i<this.config.n_layers;i++){
      const bytes=this.context*this.width*4;
      this.cacheK.push(device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
      this.cacheV.push(device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
    }
  }
  tensor(name){const t=this.weights.get(name);if(!t)throw new Error("Missing tensor: "+name);return t;}
  temp(bytes){const b=this.device.createBuffer({size:Math.max(4,align4(bytes)),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});this.temps.push(b);return b;}
  params(values,floatIndices=[]){const data=new ArrayBuffer(16),v=new DataView(data);values.forEach((x,i)=>{if(typeof x==="number"){if(floatIndices.includes(i))v.setFloat32(i*4,x,true);else v.setUint32(i*4,x,true);}});const b=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(b,0,data);this.temps.push(b);return b;}
  run(name, entries, dispatch) {
    const pipeline=this.pipelines[name],encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries.map((b,i)=>({binding:i,resource:{buffer:b}}))}));
    pass.dispatchWorkgroups(dispatch);pass.end();this.device.queue.submit([encoder.finish()]);
  }
  vector(n){return this.temp(n*4);}
  linear(name,x,outSize,inSize){
    const w=this.tensor(name);if(w.dtype!=="int8")throw new Error("Expected quantized matrix "+name);
    const y=this.vector(outSize),p=this.params([outSize,inSize,w.scale,0],[2]);
    this.run("linear",[w.buffer,x,y,p],Math.ceil(outSize/64));return y;
  }
  add(a,b,n){const y=this.vector(n),p=this.params([n,0,0,0]);this.run("add",[a,b,y,p],Math.ceil(n/64));return y;}
  norm(x,prefix){
    const g=this.tensor(prefix+".weight"),b=this.tensor(prefix+".bias"),y=this.vector(this.width),p=this.params([this.width,0,0,0]);
    this.run("norm",[x,g.buffer,b.buffer,y,p],1);return y;
  }
  embed(token,position){
    const ew=this.tensor("token_embedding.weight"),pw=this.tensor("position_embedding.weight");
    const tokenOut=this.vector(this.width),p1=this.params([token,this.width,ew.scale,0],[2]);
    this.run("embedding",[ew.buffer,this.zeroVector(),tokenOut,p1],Math.ceil(this.width/64));
    const y=this.vector(this.width),p2=this.params([position,this.width,pw.scale,0],[2]);
    this.run("embedding",[pw.buffer,tokenOut,y,p2],Math.ceil(this.width/64));return y;
  }
  zeroVector(){const b=this.vector(this.width);this.device.queue.writeBuffer(b,0,new Float32Array(this.width));return b;}
  async readVector(buffer,n){
    const out=this.device.createBuffer({size:align4(n*4),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const e=this.device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,out,0,n*4);this.device.queue.submit([e.finish()]);
    await out.mapAsync(GPUMapMode.READ);const result=new Float32Array(out.getMappedRange().slice(0));out.unmap();out.destroy();return result;
  }
  async forwardToken(token,position){
    let x=this.embed(token,position);
    for(let layer=0;layer<this.config.n_layers;layer++){
      const root="blocks."+layer+".";
      const n1=this.norm(x,root+"ln1");
      const qkv=this.linear(root+"attn.qkv.weight",n1,3*this.width,this.width);
      const q=this.vector(this.width),k=this.vector(this.width),v=this.vector(this.width),sp=this.params([this.width,0,0,0]);
      this.run("split",[qkv,q,k,v,sp],Math.ceil(this.width/64));
      const cp=this.params([position,this.width,0,0]);
      this.run("cache",[k,v,this.cacheK[layer],this.cacheV[layer],cp],Math.ceil(this.width/64));
      const scores=this.vector(this.config.n_heads*this.context),ap=this.params([position,this.context,this.width,this.headDim]);
      this.run("scores",[q,this.cacheK[layer],scores,ap],Math.ceil(this.config.n_heads*this.context/64));
      const attended=this.vector(this.width);
      this.run("attend",[scores,this.cacheV[layer],attended,ap],Math.ceil(this.width/64));
      const projected=this.linear(root+"attn.proj.weight",attended,this.width,this.width);
      x=this.add(x,projected,this.width);
      const n2=this.norm(x,root+"ln2");
      const up=this.linear(root+"mlp.0.weight",n2,4*this.width,this.width);
      const act=this.vector(4*this.width),gp=this.params([4*this.width,0,0,0]);
      this.run("gelu",[up,act,gp],Math.ceil(4*this.width/64));
      const down=this.linear(root+"mlp.2.weight",act,this.width,4*this.width);
      x=this.add(x,down,this.width);
    }
    const final=this.norm(x,"final_norm");
    const logits=this.linear("token_embedding.weight",final,this.config.vocab_size,this.width);
    const result=await this.readVector(logits,this.config.vocab_size);this.cleanupTemps();return result;
  }
  async generate(prompt,{maxNewTokens=80,temperature=0.8,onToken=null}={}){
    const ids=this.tokenizer.encode(prompt);
    if(!ids.length)throw new Error("Enter a prompt first.");
    if(ids.length>this.context)throw new Error("Prompt exceeds the "+this.context+" token context limit.");
    this.cacheK.forEach(b=>this.device.queue.writeBuffer(b,0,new Uint8Array(this.context*this.width*4)));
    this.cacheV.forEach(b=>this.device.queue.writeBuffer(b,0,new Uint8Array(this.context*this.width*4)));
    let logits;
    for(let i=0;i<ids.length;i++)logits=await this.forwardToken(ids[i],i);
    const generated=[];
    for(let i=0;i<maxNewTokens&&ids.length<this.context;i++){
      const eosId=this.tokenizer.vocab["[EOS]"];
      // Control markers must never leak into the visible answer or end it early.
      // [EOS] is the only special token that should terminate a response.
      for(const specialId of this.tokenizer.specialTokenIds){
        if(specialId!==eosId)logits[specialId]=-Infinity;
      }
      let next;
      if(temperature<=0){next=0;for(let j=1;j<logits.length;j++)if(logits[j]>logits[next])next=j;}
      else {
        const ranked=Array.from(logits,(v,id)=>({id,v:v/temperature})).sort((a,b)=>b.v-a.v).slice(0,40);
        const max=ranked[0].v,weights=ranked.map(x=>Math.exp(x.v-max)),total=weights.reduce((a,b)=>a+b,0);
        let r=Math.random()*total;next=ranked[ranked.length-1].id;
        for(let j=0;j<ranked.length;j++){r-=weights[j];if(r<=0){next=ranked[j].id;break;}}
      }
      ids.push(next);
      if(next===eosId)break;
      generated.push(next);
      if(typeof onToken==="function")onToken(this.tokenizer.decode(generated));
      if(ids.length<this.context)logits=await this.forwardToken(next,ids.length-1);
    }
    return this.tokenizer.decode(generated);
  }
  cleanupTemps(){for(const b of this.temps)b.destroy();this.temps=[];}
  destroy(){this.cleanupTemps();for(const b of this.buffers)b.destroy();for(const b of this.cacheK)b.destroy();for(const b of this.cacheV)b.destroy();}
}

export async function loadRogerSpark({base="./model/"}={}){
  if(!navigator.gpu)throw new Error("WebGPU isn't available in this browser. Try a current Chrome or Edge build on a supported device.");
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No compatible WebGPU adapter was found.");
  const device=await adapter.requestDevice();
  const [mr,wr,tr]=await Promise.all([
    fetch(base+"manifest.json"),fetch(base+"weights.bin"),fetch(base+"tokenizer.json")
  ]);
  if(!mr.ok||!wr.ok||!tr.ok)throw new Error("Roger Spark’s trained weights are not published yet. On the computer with your checkpoint, run: python3 model/export_webgpu.py --checkpoint models/roger-chat-test/checkpoint.pt --output webgpu/model. Then commit and push the generated webgpu/model/ files to GitHub.");
  const [manifest,weights,tok]=await Promise.all([mr.json(),wr.arrayBuffer(),tr.json()]);
  const tokenizer=new ByteLevelTokenizer(tok);
  for(const marker of ["[USER]","[ROGER]"]){
    const ids=tokenizer.encode(marker);
    if(ids.length!==1||ids[0]!==tokenizer.vocab[marker])throw new Error("Tokenizer self-test failed for "+marker);
  }
  const sample="Hi, world! 2+2 = 4.";
  if(tokenizer.decode(tokenizer.encode(sample))!==sample)throw new Error("Tokenizer self-test failed: text round-trip mismatch");
  return new RogerSparkWebGPU(device,manifest,weights,tokenizer);
}
