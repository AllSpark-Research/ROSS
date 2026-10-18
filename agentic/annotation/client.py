"""OpenAI-compatible client used by the agentic annotation pipeline."""
import json,os,time,urllib.request,urllib.error,threading,hashlib
from pathlib import Path
ENDPOINT=os.environ.get('ROSS_ENDPOINT','http://localhost:8000/v1/chat/completions')
MODEL=os.environ.get('ROSS_MODEL','judge-model')
RPM=int(os.environ.get('ROSS_RPM','4'))
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
class Client:
 def __init__(self,root,max_calls=int(os.environ.get('ROSS_MAX_CALLS','60'))):
  self.root=Path(root);self.lock=threading.Lock()
  self.count=len(list(self.root.glob('**/request.json')));self.cap=max_calls
  self.blocked=threading.Event()
  self.sent=[f.stat().st_mtime for f in self.root.glob("**/request.json") if time.time()-f.stat().st_mtime<60]
 def call(self,system,user,dest,max_tokens=None,transport_attempt=0):
  max_tokens=max_tokens or int(os.environ.get('ROSS_MAXTOK','49152'))
  dest=Path(dest)
  body={'model':MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
        'temperature':0,'max_tokens':max_tokens,'stream':False,
        'chat_template_kwargs':{'enable_thinking':True}}
  if 'GLM' in MODEL: body['reasoning_effort']=os.environ.get('ROSS_EFFORT','high')
  fp=digest({'endpoint':ENDPOINT,'body':body})
  if (dest/'response.json').exists():
   assert json.loads((dest/'meta.json').read_text())['fingerprint']==fp,'Changed cached input'
   return json.loads((dest/'response.json').read_text())
  while True:
   with self.lock:
    if self.blocked.is_set():raise RuntimeError('Service unavailable; further calls stopped')
    if self.count>=self.cap:raise RuntimeError('Per-sample call cap reached')
    now=time.time();self.sent=[t for t in self.sent if now-t<60]
    if len(self.sent)<RPM:self.sent.append(now);self.count+=1;break
    delay=max(.2,min(60,60-(now-min(self.sent))+.2))
   time.sleep(delay)
  dump(dest/'request.json',body);t=time.time()
  try:
   headers={'Content-Type':'application/json'}
   key=os.environ.get('ROSS_API_KEY','')
   if key:
    name=os.environ.get('ROSS_AUTH_HEADER','Authorization');scheme=os.environ.get('ROSS_AUTH_SCHEME','Bearer')
    headers[name]=(scheme+' '+key).strip()
   req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers=headers)
   with urllib.request.urlopen(req,timeout=1800) as response:raw=json.load(response)
   dump(dest/'response.json',raw)
   dump(dest/'meta.json',{'fingerprint':fp,'seconds':time.time()-t,'response_sha256':digest(raw)})
   return raw
  except urllib.error.HTTPError as e:
   detail=e.read().decode(errors='replace');dump(dest/'error.json',{'status':e.code,'body':detail,'seconds':time.time()-t})
   if e.code in [401,403,404]:self.blocked.set()
   if e.code==429 and transport_attempt<2:
    time.sleep(30);return self.call(system,user,dest/'transport_retry',max_tokens,transport_attempt+1)
   raise RuntimeError('HTTP '+str(e.code)+' '+detail[:300]) from None
  except Exception as e:
   dump(dest/'error.json',{'error':str(e),'seconds':time.time()-t});raise
