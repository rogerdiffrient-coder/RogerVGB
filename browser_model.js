// BPE tokenizer + ONNX Runtime Web inference. Context and output are hard-capped.
const MAX_CONTEXT = 320;
const MAX_OUTPUT = 128;
const MAX_PROMPT = MAX_CONTEXT - MAX_OUTPUT;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const softmax=xs=>{let m=-Infinity;for(const x of xs)if(x>m)m=x;const ex=new Float32Array(xs.length);let s=0;for(let i=0;i<xs.length;i++){ex[i]=Math.exp(xs[i]-m);s+=ex[i];}if(!s)s=1;for(let i=0;i<ex.length;i++)ex[i]/=s;return ex;};

function makeByteMaps(){
  const bytes=[],chars=[];
  for(let b=33;b<=126;b++){bytes.push(b);chars.push(b);}
  for(let b=161;b<=172;b++){bytes.push(b);chars.push(b);}
  for(let b=174;b<=255;b++){bytes.push(b);chars.push(b);}
  let n=0;
  for(let b=0;b<256;b++){
    if(!bytes.includes(b)){bytes.push(b);chars.push(256+n++);}
  }
  const encoder=new Map(),decoder=new Map();
  for(let i=0;i<bytes.length;i++){const ch=String.fromCodePoint(chars[i]);encoder.set(bytes[i],ch);decoder.set(ch,bytes[i]);}
  return {encoder,decoder};
}
const byteMaps=makeByteMaps();
const pretokenPattern=/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

class ByteLevelBPE {
  constructor(data){
    this.vocab=data.model.vocab;
    this.merges=new Map();
    (data.model.merges||[]).forEach((m,i)=>{const pair=Array.isArray(m)?m.join(' '):m;this.merges.set(pair,i);});
    this.idToToken=new Array(Math.max(...Object.values(this.vocab))+1);
    for(const [token,id] of Object.entries(this.vocab))this.idToToken[id]=token;
    this.padId=this.vocab['<pad>']??0;this.eosId=this.vocab['<eos>']??1;
    this.tokenizerDecoder=new TextDecoder();
  }
  encode(text){
    const out=[];
    const pieces=String(text).match(pretokenPattern)||[];
    for(const piece of pieces){
      const symbols=[...new TextEncoder().encode(piece)].map(b=>byteMaps.encoder.get(b));
      while(symbols.length>1){
        let bestRank=Infinity,bestIndex=-1;
        for(let i=0;i<symbols.length-1;i++){
          const rank=this.merges.get(symbols[i]+' '+symbols[i+1]);
          if(rank!==undefined&&rank<bestRank){bestRank=rank;bestIndex=i;}
        }
        if(bestIndex<0)break;
        symbols.splice(bestIndex,2,symbols[bestIndex]+symbols[bestIndex+1]);
      }
      for(const symbol of symbols){
        const id=this.vocab[symbol];
        if(id===undefined)throw new Error('Tokenizer vocabulary missing a byte-level token.');
        out.push(id);
      }
    }
    return out;
  }
  decode(ids){
    const bytes=[];
    for(const id of ids){
      if(id===this.eosId||id===this.padId)continue;
      const token=this.idToToken[id];if(!token||token.startsWith('<')&&token.endsWith('>'))continue;
      for(const ch of token){const b=byteMaps.decoder.get(ch);if(b!==undefined)bytes.push(b);}
    }
    return this.tokenizerDecoder.decode(new Uint8Array(bytes));
  }
}

export class RogerSparkBrowser{
  constructor(){this.ready=false;this.session=null;this.tokenizer=null;this.runtime=null;this.cfg={max_seq:MAX_CONTEXT,vocab_size:6144};}
  async initialize(){
    const ortUrl='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
    this.runtime=await import(ortUrl);
    this.runtime.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    this.runtime.env.wasm.numThreads=1;
    const [tokResponse,cfgResponse]=await Promise.all([
      fetch('./roger_spark_tokenizer.json',{cache:'no-store'}),
      fetch('./roger_spark_config.json',{cache:'no-store'})
    ]);
    if(!tokResponse.ok||!cfgResponse.ok)throw new Error('Tokenizer/config not found. Train and export Roger Spark first.');
    this.tokenizer=new ByteLevelBPE(await tokResponse.json());
    this.cfg=await cfgResponse.json();
    if(this.cfg.parameter_count!==25000000)throw new Error('Model config parameter-count check failed.');
    if(!navigator.gpu)throw new Error('This browser does not expose WebGPU. Open Roger in a WebGPU-capable browser.');
    this.session=await this.runtime.InferenceSession.create('./roger_spark.onnx',{
      executionProviders:['webgpu'],graphOptimizationLevel:'all'
    });
    this.ready=true;return this;
  }
  async reply(prompt,opts={}){
    if(!this.ready)throw new Error('Roger Spark is not initialized.');
    const maxTokens=clamp(Math.floor(Number(opts.maxTokens)||80),1,MAX_OUTPUT);
    const temperature=clamp(Number(opts.temperature)||0.7,0.1,1.2);
    const topP=clamp(Number(opts.topP)||0.9,0.5,1);
    // The story-only checkpoint is trained without dialogue wrappers.
    let ids=this.tokenizer.encode(String(prompt).slice(0,8000));
    ids=ids.slice(-MAX_PROMPT);
    const generated=[];
    for(let step=0;step<maxTokens;step++){
      const context=ids.concat(generated).slice(-MAX_CONTEXT);
      const tensor=new this.runtime.Tensor('int32',Int32Array.from(context),[1,context.length]);
      const result=await this.session.run({input_ids:tensor});
      const logits=result.logits.data;
      const vocab=this.cfg.vocab_size;
      const last=new Float32Array(vocab);
      const offset=(context.length-1)*vocab;
      for(let i=0;i<vocab;i++)last[i]=Number(logits[offset+i])/temperature;
      const probs=softmax(last);
      const order=Array.from(probs,(p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
      let mass=0,choices=[];
      for(const item of order){choices.push(item);mass+=item.p;if(mass>=topP)break;}
      let sum=0;for(const item of choices)sum+=item.p;
      let roll=Math.random()*sum,chosen=choices[choices.length-1].i;
      for(const item of choices){roll-=item.p;if(roll<=0){chosen=item.i;break;}}
      if(chosen===this.tokenizer.eosId)break;
      generated.push(chosen);
    }
    let text=this.tokenizer.decode(generated).replace(/\u0000/g,'').trim();
    return text||'…';
  }
}
