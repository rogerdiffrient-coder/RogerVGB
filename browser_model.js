// Tiny pure-JavaScript inference for Roger Spark's own scratch-trained Transformer.
// This intentionally favors clarity and small model size over maximum speed.
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const softmax=xs=>{const m=Math.max(...xs);const ex=xs.map(x=>Math.exp(x-m));const s=ex.reduce((a,b)=>a+b,0)||1;return ex.map(x=>x/s)};
function linear(x,w,b,out,inSize){const y=new Float32Array(out);for(let o=0;o<out;o++){let s=b?b[o]:0,off=o*inSize;for(let i=0;i<inSize;i++)s+=w[off+i]*x[i];y[o]=s;}return y;}
function norm(x,w,b){let mean=0;for(const v of x)mean+=v;mean/=x.length;let variance=0;for(const v of x){const d=v-mean;variance+=d*d;}variance/=x.length;const inv=1/Math.sqrt(variance+1e-5),y=new Float32Array(x.length);for(let i=0;i<x.length;i++)y[i]=(x[i]-mean)*inv*w[i]+b[i];return y;}
function gelu(x){return .5*x*(1+Math.tanh(.7978845608*(x+.044715*x*x*x)));}
export class RogerSparkBrowser{
  constructor(bundle){this.cfg=bundle.config;this.w={};for(const [k,v] of Object.entries(bundle.weights))this.w[k]=Float32Array.from(v);this.ready=true;}
  encode(s){return [...new TextEncoder().encode(s)].map(b=>b+1);}
  decode(ids){return new TextDecoder().decode(Uint8Array.from(ids.filter(i=>i>=1&&i<=256).map(i=>i-1)));}
  forward(ids){
    const w=this.w,c=this.cfg,W=c.width,H=c.heads,D=W/H,V=c.vocab_size,L=ids.length;
    let x=ids.map((id,p)=>{const a=new Float32Array(W),to=w['token.weight'],po=w['position.weight'];for(let j=0;j<W;j++)a[j]=to[id*W+j]+po[p*W+j];return a;});
    for(let layer=0;layer<c.layers;layer++){
      const pre='blocks.'+layer+'.',n1=x.map(v=>norm(v,w[pre+'norm1.weight'],w[pre+'norm1.bias']));
      const qkv=n1.map(v=>linear(v,w[pre+'attn.in_proj_weight'],w[pre+'attn.in_proj_bias'],3*W,W));
      const att=[];
      for(let p=0;p<L;p++){
        const merged=new Float32Array(W);
        for(let h=0;h<H;h++){
          const scores=new Array(p+1);for(let t=0;t<=p;t++){let dot=0;for(let d=0;d<D;d++)dot+=qkv[p][h*D+d]*qkv[t][W+h*D+d];scores[t]=dot/Math.sqrt(D);}
          const probs=softmax(scores);
          for(let d=0;d<D;d++){let sum=0;for(let t=0;t<=p;t++)sum+=probs[t]*qkv[t][2*W+h*D+d];merged[h*D+d]=sum;}
        }
        const proj=linear(merged,w[pre+'attn.out_proj.weight'],w[pre+'attn.out_proj.bias'],W,W);
        const residual=new Float32Array(W);for(let j=0;j<W;j++)residual[j]=x[p][j]+proj[j];att.push(residual);
      }
      x=att.map(v=>{const n=norm(v,w[pre+'norm2.weight'],w[pre+'norm2.bias']);const hidden=linear(n,w[pre+'ff.0.weight'],w[pre+'ff.0.bias'],4*W,W);for(let i=0;i<hidden.length;i++)hidden[i]=gelu(hidden[i]);const out=linear(hidden,w[pre+'ff.2.weight'],w[pre+'ff.2.bias'],W,4*W);const r=new Float32Array(W);for(let j=0;j<W;j++)r[j]=v[j]+out[j];return r;});
    }
    const last=norm(x[x.length-1],w['norm.weight'],w['norm.bias']),emb=w['token.weight'],logits=new Float32Array(V);
    for(let t=0;t<V;t++){let s=0;for(let j=0;j<W;j++)s+=last[j]*emb[t*W+j];logits[t]=s;}
    return logits;
  }
  async reply(prompt,opts={}){
    const maxSeq=this.cfg.max_seq,limit=clamp(Number(opts.maxTokens)||48,1,64),temp=clamp(Number(opts.temperature)||.7,.1,1.2),topP=clamp(Number(opts.topP)||.9,.5,1);
    const ids=this.encode('User: '+prompt+'\nRoger:').slice(0,maxSeq-1),generated=[];
    for(let step=0;step<limit;step++){
      const context=ids.concat(generated).slice(-maxSeq),logits=this.forward(context);
      const scaled=Array.from(logits, v=>v/temp),probs=softmax(scaled),order=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
      let total=0,choices=[];for(const item of order){choices.push(item);total+=item.p;if(total>=topP)break;}
      const sum=choices.reduce((s,x)=>s+x.p,0);let r=Math.random()*sum,chosen=choices[choices.length-1].i;for(const item of choices){r-=item.p;if(r<=0){chosen=item.i;break;}}
      if(chosen===257)break;generated.push(chosen);
    }
    let text=this.decode(generated).replace(/\u0000/g,'').trim();
    const stop=text.indexOf('\nUser:');if(stop>=0)text=text.slice(0,stop).trim();
    return text||'…';
  }
}
