#!/usr/bin/env python3
"""QTCPS: deterministic quantum-inspired tensor-constrained program synthesis.
The solver NEVER opens an evaluation-solution file. It selects ten tasks using
training-pair features only, requires pairwise RBF similarity < 0.10, predicts,
and freezes predictions with SHA256. Classical CPU execution; no quantum-speedup claim.
"""
import argparse, hashlib, json, math, statistics
from collections import Counter, deque


def G(x): return tuple(tuple(int(v) for v in r) for r in x)
def L(g): return [list(r) for r in g]
def shp(g): return len(g), len(g[0]) if g else 0
def flat(g): return [v for r in g for v in r]
def mode(g):
    c=Counter(flat(g)); return min(((-n,v) for v,n in c.items()))[1] if c else 0
def r90(g): return tuple(tuple(x) for x in zip(*g[::-1]))
def r180(g): return r90(r90(g))
def r270(g): return r90(r180(g))
def fh(g): return tuple(tuple(reversed(r)) for r in g)
def fv(g): return tuple(reversed(g))
def tr(g):
    h,w=shp(g); return tuple(tuple(g[r][c] for r in range(h)) for c in range(w))
def geo(g,k): return (g,r90(g),r180(g),r270(g),fh(g),fv(g),tr(g),fh(tr(g)))[k]
def agree(a,b):
    if shp(a)!=shp(b): return 0.0
    aa,bb=flat(a),flat(b); return sum(x==y for x,y in zip(aa,bb))/max(1,len(aa))
def dist(a,b):
    if shp(a)==shp(b): return 1-agree(a,b)
    ha,wa=shp(a); hb,wb=shp(b)
    return 1+abs(ha-hb)+abs(wa-wb)+abs(ha*wa-hb*wb)/max(1,ha*wa,hb*wb)

def comps(g,bg=None):
    if bg is None:bg=mode(g)
    h,w=shp(g); seen=set(); out=[]
    for r in range(h):
      for c in range(w):
        if (r,c) in seen or g[r][c]==bg: continue
        q=[(r,c)];seen.add((r,c));cc=[]
        while q:
          x,y=q.pop();cc.append((x,y,g[x][y]))
          for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            u,v=x+dx,y+dy
            if 0<=u<h and 0<=v<w and (u,v) not in seen and g[u][v]!=bg:
              seen.add((u,v));q.append((u,v))
        out.append(cc)
    return out
def bbox(c):
    rr=[x for x,_,_ in c];cc=[y for _,y,_ in c];return min(rr),min(cc),max(rr),max(cc)
def crop(g,bg=None):
    if bg is None:bg=mode(g)
    p=[(r,c,g[r][c]) for r in range(len(g)) for c in range(len(g[0])) if g[r][c]!=bg]
    if not p:return g
    a,b,z,d=bbox(p);return tuple(tuple(g[r][c] for c in range(b,d+1)) for r in range(a,z+1))
def ccrop(g,which,bg=None):
    if bg is None:bg=mode(g)
    cs=comps(g,bg)
    if not cs:return g
    keys={'largest':lambda x:(-len(x),bbox(x)[0],bbox(x)[1]),'smallest':lambda x:(len(x),bbox(x)[0],bbox(x)[1]),
          'top':lambda x:(bbox(x)[0],bbox(x)[1]),'left':lambda x:(bbox(x)[1],bbox(x)[0]),
          'bottom':lambda x:(-bbox(x)[2],-bbox(x)[3]),'right':lambda x:(-bbox(x)[3],-bbox(x)[2])}
    p=min(cs,key=keys[which]);a,b,z,d=bbox(p)
    return tuple(tuple(g[r][c] for c in range(b,d+1)) for r in range(a,z+1))
def compress(g,bg=None):
    if bg is None:bg=mode(g)
    rs=[r for r in range(len(g)) if any(v!=bg for v in g[r])]
    cs=[c for c in range(len(g[0])) if any(g[r][c]!=bg for r in range(len(g)))]
    return tuple(tuple(g[r][c] for c in cs) for r in rs) if rs and cs else g
def scale(g,a,b): return tuple(tuple(v for v in row for _ in range(b)) for row in g for _ in range(a))
def tile(g,a,b):
    h,w=shp(g);return tuple(tuple(g[r%h][c%w] for c in range(w*b)) for r in range(h*a))
def cmap(g,m): return tuple(tuple(m.get(v,v) for v in row) for row in g)
def gravity(g,d,bg=None):
    if bg is None:bg=mode(g)
    h,w=shp(g);o=[[bg]*w for _ in range(h)]
    if d in ('up','down'):
      for c in range(w):
        v=[g[r][c] for r in range(h) if g[r][c]!=bg];s=0 if d=='up' else h-len(v)
        for i,x in enumerate(v):o[s+i][c]=x
    else:
      for r in range(h):
        v=[g[r][c] for c in range(w) if g[r][c]!=bg];s=0 if d=='left' else w-len(v)
        for i,x in enumerate(v):o[r][s+i]=x
    return G(o)
def holes(g,bg=None):
    if bg is None:bg=mode(g)
    h,w=shp(g);outside=set();q=deque()
    for r in range(h):
      for c in (0,w-1):
        if g[r][c]==bg and (r,c) not in outside:outside.add((r,c));q.append((r,c))
    for c in range(w):
      for r in (0,h-1):
        if g[r][c]==bg and (r,c) not in outside:outside.add((r,c));q.append((r,c))
    while q:
      r,c=q.popleft()
      for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        u,v=r+dr,c+dc
        if 0<=u<h and 0<=v<w and g[u][v]==bg and (u,v) not in outside:outside.add((u,v));q.append((u,v))
    fg=[v for v,_ in Counter(flat(g)).most_common() if v!=bg]
    if not fg:return g
    o=L(g)
    for r in range(h):
      for c in range(w):
        if g[r][c]==bg and (r,c) not in outside:o[r][c]=fg[0]
    return G(o)
def boxfill(g,outline=False,bg=None):
    if bg is None:bg=mode(g)
    p=[(r,c,g[r][c]) for r in range(len(g)) for c in range(len(g[0])) if g[r][c]!=bg]
    if not p:return g
    a,b,z,d=bbox(p);col=Counter(v for _,_,v in p).most_common(1)[0][0];o=L(g)
    for r in range(a,z+1):
      for c in range(b,d+1):
        if not outline or r in (a,z) or c in (b,d):o[r][c]=col
    return G(o)
def connect(g,bg=None):
    if bg is None:bg=mode(g)
    o=L(g);by={}
    for r in range(len(g)):
      for c in range(len(g[0])):
        if g[r][c]!=bg:by.setdefault(g[r][c],[]).append((r,c))
    for v,p in by.items():
      for i,(r,c) in enumerate(p):
        for u,z in p[i+1:]:
          if r==u:
            for x in range(min(c,z),max(c,z)+1):o[r][x]=v
          if c==z:
            for x in range(min(r,u),max(r,u)+1):o[x][c]=v
    return G(o)
def mirror(g,axis,dup):
    if axis=='h':return tuple(tuple(list(r)+(list(reversed(r)) if dup else list(reversed(r[:-1])))) for r in g)
    return tuple(g)+(tuple(reversed(g)) if dup else tuple(reversed(g[:-1])))

def infer_map(src,tgt):
    m={}
    for a,b in zip(src,tgt):
      if shp(a)!=shp(b):return None
      for ra,rb in zip(a,b):
        for x,y in zip(ra,rb):
          if x in m and m[x]!=y:return None
          m[x]=y
    return m

def mkops(train):
    src=[G(x['input']) for x in train];tgt=[G(x['output']) for x in train];ops=[];names=set()
    def add(n,c,f):
      if n not in names:names.add(n);ops.append((n,c,f))
    for k,n in enumerate(('id','r90','r180','r270','fh','fv','tr','atr')):add(n,.1 if k==0 else .8,lambda g,k=k:geo(g,k))
    for bgname,bgf in (('mode',mode),('zero',lambda g:0)):
      add('crop_'+bgname,1.2,lambda g,bgf=bgf:crop(g,bgf(g)));add('compress_'+bgname,1.4,lambda g,bgf=bgf:compress(g,bgf(g)))
      for x in ('largest','smallest','top','bottom','left','right'):add('cc_'+x+'_'+bgname,1.7,lambda g,x=x,bgf=bgf:ccrop(g,x,bgf(g)))
      for d in ('up','down','left','right'):add('gravity_'+d+'_'+bgname,1.8,lambda g,d=d,bgf=bgf:gravity(g,d,bgf(g)))
      add('holes_'+bgname,1.8,lambda g,bgf=bgf:holes(g,bgf(g)));add('fillbox_'+bgname,1.7,lambda g,bgf=bgf:boxfill(g,False,bgf(g)))
      add('outline_'+bgname,1.8,lambda g,bgf=bgf:boxfill(g,True,bgf(g)));add('connect_'+bgname,2.0,lambda g,bgf=bgf:connect(g,bgf(g)))
    for a in ('h','v'):
      for d in (0,1):add('mirror_'+a+str(d),1.6,lambda g,a=a,d=d:mirror(g,a,d))
    ratios=[]
    for a,b in zip(src,tgt):
      ha,wa=shp(a);hb,wb=shp(b);ratios.append((hb/ha,wb/wa))
    if ratios and len(set(ratios))==1:
      a,b=ratios[0]
      if a.is_integer() and b.is_integer() and 1<=a<=8 and 1<=b<=8:
        a,b=int(a),int(b);add(f'scale_{a}x{b}',1.5,lambda g,a=a,b=b:scale(g,a,b));add(f'tile_{a}x{b}',1.5,lambda g,a=a,b=b:tile(g,a,b))
    for k,n in enumerate(('id','r90','r180','r270','fh','fv','tr','atr')):
      m=infer_map([geo(x,k) for x in src],tgt)
      if m and any(m[x]!=x for x in m):add('geomap_'+n+str(sorted(m.items())),1.3,lambda g,k=k,m=dict(m):cmap(geo(g,k),m))
    m=infer_map(src,tgt)
    if m and any(m[x]!=x for x in m):add('cmap_'+str(sorted(m.items())),1.0,lambda g,m=dict(m):cmap(g,m))
    return ops

def safe_apply(f,g):
    try:
      x=f(g);h,w=shp(x);return x if 0<h<=60 and 0<w<=60 else g
    except Exception:return g

def synth(train,depth=3,cap=12000):
    ops=mkops(train);src=tuple(G(x['input']) for x in train);tgt=tuple(G(x['output']) for x in train)
    if src==tgt:return [],ops,{'exact':True,'depth':0,'states':1,'evals':0}
    layer={src:(0.0,())};total=1;ev=0
    for d in range(1,depth+1):
      nx={}
      for st,(cost,p) in sorted(layer.items(),key=lambda x:(x[1][0],x[1][1])):
        for i,(n,c,f) in enumerate(ops):
          ns=tuple(safe_apply(f,g) for g in st);ev+=1;v=(cost+c+.02*d,p+(i,))
          if ns not in nx or v<nx[ns]:nx[ns]=v
      if tgt in nx:
        cost,p=nx[tgt];return [ops[i] for i in p],ops,{'exact':True,'depth':d,'states':total+len(nx),'evals':ev,'cost':cost}
      if len(nx)>cap:
        ranked=sorted(nx.items(),key=lambda x:(sum(dist(a,b) for a,b in zip(x[0],tgt)),x[1][0],x[1][1]));nx=dict(ranked[:cap])
      layer=nx;total+=len(layer)
    st,(cost,p)=min(layer.items(),key=lambda x:(sum(dist(a,b) for a,b in zip(x[0],tgt)),x[1][0],x[1][1]))
    return [ops[i] for i in p],ops,{'exact':False,'depth':depth,'states':total,'evals':ev,'cost':cost,'train_dist':sum(dist(a,b) for a,b in zip(st,tgt))}

def greedy(train,depth=3):
    ops=mkops(train);st=tuple(G(x['input']) for x in train);tgt=tuple(G(x['output']) for x in train);p=[]
    for _ in range(depth):
      q=[]
      for i,(n,c,f) in enumerate(ops):
        ns=tuple(safe_apply(f,g) for g in st);q.append((sum(dist(a,b) for a,b in zip(ns,tgt)),c,i,ns))
      _,_,i,st=min(q);p.append(ops[i])
      if st==tgt:break
    return p,st==tgt

def sig(t):
    z=[]
    for p in t['train']:
      a,b=G(p['input']),G(p['output']);ha,wa=shp(a);hb,wb=shp(b);ba,bb=mode(a),mode(b)
      z.append([math.log1p(ha),math.log1p(wa),math.log1p(hb),math.log1p(wb),math.log((hb+1)/(ha+1)),math.log((wb+1)/(wa+1)),len(set(flat(a)))/10,len(set(flat(b)))/10,len(comps(a,ba))/20,len(comps(b,bb))/20,float(shp(a)==shp(b)),agree(a,b) if shp(a)==shp(b) else 0,*[float(a==geo(a,k)) for k in (1,2,4,5,6)],*[float(b==geo(b,k)) for k in (1,2,4,5,6)]])
    return [sum(x[i] for x in z)/len(z) for i in range(len(z[0]))]
def select(tasks,n=10,thr=.10):
    ids=sorted(tasks);f=[sig(tasks[i]) for i in ids];d=len(f[0]);med=[statistics.median(x[j] for x in f) for j in range(d)];sc=[]
    for j in range(d):
      mad=statistics.median(abs(x[j]-med[j]) for x in f);sc.append(max(.05,1.4826*mad))
    Z={i:[(x[j]-med[j])/sc[j] for j in range(d)] for i,x in zip(ids,f)}
    sim=lambda a,b:math.exp(-.5*sum((x-y)**2 for x,y in zip(Z[a],Z[b]))/d)
    first=sorted(ids,key=lambda i:(-sum(x*x for x in Z[i]),i))[0];s=[first]
    while len(s)<n:
      c=[(max(sim(i,j) for j in s),i) for i in ids if i not in s];c=[x for x in c if x[0]<thr]
      if not c:break
      s.append(min(c)[1])
    mx=max((sim(a,b) for k,a in enumerate(s) for b in s[k+1:]),default=0);return s,mx

def run(g,p):
    for _,_,f in p:g=safe_apply(f,g)
    return g

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--challenges',required=True);ap.add_argument('--out',required=True);ap.add_argument('--depth',type=int,default=3);ap.add_argument('--cap',type=int,default=12000);a=ap.parse_args()
    with open(a.challenges) as f:tasks=json.load(f)
    ids,mx=select(tasks,10,.10);print('SELECTED',ids);print('MAX_PAIRWISE_SIMILARITY',mx)
    if len(ids)!=10 or mx>=.10:raise SystemExit('diversity constraint failed')
    pred={};meta={}
    for tid in ids:
      t=tasks[tid];p,ops,st=synth(t['train'],a.depth,a.cap);gp,gexact=greedy(t['train'],a.depth)
      pred[tid]=[L(run(G(x['input']),p)) for x in t['test']]
      meta[tid]={'program':[x[0] for x in p],'tensor':st,'greedy_program':[x[0] for x in gp],'greedy_train_exact':gexact,'n_ops':len(ops)}
      print('TASK',tid,'train_exact',st['exact'],'program',[x[0] for x in p],'states',st['states'],'evals',st['evals'])
    core={'selected':ids,'max_pairwise_similarity':mx,'predictions':pred,'meta':meta,'protocol':{'deterministic':True,'similarity_threshold':.10,'depth':a.depth,'cap':a.cap,'solutions_access':False}}
    raw=json.dumps(core,sort_keys=True,separators=(',',':')).encode();sha=hashlib.sha256(raw).hexdigest();core['freeze_sha256']=sha
    with open(a.out,'w') as f:json.dump(core,f,sort_keys=True,indent=2)
    with open(a.out+'.sha256','w') as f:f.write(sha+'  '+a.out+'\n')
    print('FROZEN_SHA256',sha)
if __name__=='__main__':main()
