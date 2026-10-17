"""Local API for the scratch-trained Roger Spark model. Run train.py first."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch
from roger_spark import RogerSpark, ByteTokenizer, sample_next

HERE=Path(__file__).resolve().parent
WEIGHTS=HERE/"roger_spark.pt"
device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
if not WEIGHTS.exists():
    raise SystemExit("No trained weights found. Run: python train.py")
checkpoint=torch.load(WEIGHTS,map_location=device,weights_only=False)
model=RogerSpark(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["state_dict"]);model.eval()
tokenizer=ByteTokenizer()

def generate(prompt, history=None, temperature=.8, top_p=.9, max_new_tokens=160):
    turns=[]
    for item in (history or [])[-6:]:
        role="Roger" if item.get("role")=="assistant" else "User"
        content=str(item.get("content","")).strip()[:700]
        if content: turns.append(f"{role}: {content}")
    turns.append(f"User: {str(prompt).strip()[:700]}")
    prefix="\n".join(turns)+"\nRoger:"
    ids=torch.tensor([tokenizer.encode(prefix)[:-1]],device=device)
    generated=[]
    with torch.no_grad():
        for _ in range(max(1,min(256,int(max_new_tokens)))):
            logits=model(ids[:, -model.config["max_seq"]:])[:,-1,:]
            nxt=int(sample_next(logits,temperature,top_p).item())
            if nxt==tokenizer.eos_id: break
            generated.append(nxt)
            ids=torch.cat([ids,torch.tensor([[nxt]],device=device)],dim=1)
            if len(ids[0])>=model.config["max_seq"]: break
    return tokenizer.decode(generated).strip()

class Handler(BaseHTTPRequestHandler):
    def _send(self,status,payload):
        data=json.dumps(payload).encode()
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin","*");self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.send_header("Access-Control-Allow-Methods","POST, OPTIONS");self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=="/api/health": return self._send(200,{"ok":True,"model":"Roger Spark 0.1","device":device})
        return self._send(404,{"error":"Not found"})
    def do_OPTIONS(self): self._send(200,{})
    def do_POST(self):
        if self.path!="/api/chat": return self._send(404,{"error":"Not found"})
        try:
            data=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
            answer=generate(data.get("message",""),data.get("history",[]),data.get("temperature",.8),data.get("top_p",.9),data.get("max_new_tokens",160))
            self._send(200,{"reply":answer,"model":"Roger Spark 0.1","device":device})
        except Exception as exc: self._send(500,{"error":str(exc)})
    def log_message(self,*args): pass

print(f"Roger Spark 0.1 custom model API on http://127.0.0.1:8000 ({device})")
HTTPServer(("127.0.0.1",8000),Handler).serve_forever()
