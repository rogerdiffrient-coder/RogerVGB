import argparse, json, random
from pathlib import Path
import torch
from torch.nn import functional as F
from roger_spark import RogerSpark, ByteTokenizer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=2500)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--max-seq",type=int,default=256)
    parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--layers",type=int,default=4)
    parser.add_argument("--heads",type=int,default=4)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--data",default=str(Path(__file__).with_name("data.jsonl")))
    parser.add_argument("--out",default=str(Path(__file__).with_name("roger_spark.pt")))
    args=parser.parse_args()
    random.seed(42); torch.manual_seed(42)
    device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=ByteTokenizer()
    examples=[]
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item=json.loads(line)
            # Train the model to continue from prompt into answer, then EOS.
            text=f"User: {item['prompt']}\nRoger: {item['response']}"
            examples.append(torch.tensor(tokenizer.encode(text),dtype=torch.long))
    if not examples: raise SystemExit("Training dataset is empty.")
    model=RogerSpark(width=args.width,layers=args.layers,heads=args.heads,max_seq=args.max_seq).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    model.train()
    for step in range(1,args.steps+1):
        batch=[]; targets=[]
        for _ in range(args.batch_size):
            seq=random.choice(examples)
            if len(seq)>args.max_seq+1:
                start=random.randrange(len(seq)-args.max_seq)
                seq=seq[start:start+args.max_seq+1]
            x=seq[:-1]; y=seq[1:]
            pad=args.max_seq-len(x)
            batch.append(F.pad(x,(max(0,pad),0),value=0)[-args.max_seq:])
            targets.append(F.pad(y,(max(0,pad),0),value=-100)[-args.max_seq:])
        x=torch.stack(batch).to(device); y=torch.stack(targets).to(device)
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step==1 or step%100==0: print(f"step {step}/{args.steps} loss={loss.item():.4f} device={device}")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"config":model.config,"state_dict":model.cpu().state_dict(),"tokenizer":"utf8-byte-v1"},out)
    print(f"Saved scratch-trained Roger Spark weights to {out}")
if __name__=="__main__": main()
