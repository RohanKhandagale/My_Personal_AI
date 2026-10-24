"""Python backend for My_personal_AI: vector search, REST API and Ollama RAG."""
import json, math, os, random, time, urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
DIMS = 16
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
app = Flask(__name__, static_folder=None)


def distance(a, b, metric="cosine"):
    if len(a) != len(b): return float("inf")
    if metric == "euclidean": return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    if metric == "manhattan": return sum(abs(x-y) for x,y in zip(a,b))
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return 1.0 if na < 1e-9 or nb < 1e-9 else 1.0-dot/(na*nb)


def chunk_text(text, size=250, overlap=30):
    words=text.split()
    if len(words)<=size: return [text] if words else []
    step=max(1,size-overlap)
    return [' '.join(words[i:i+size]) for i in range(0,len(words),step)]


class KDNode:
    def __init__(self, item, axis, left=None, right=None):
        self.item=item; self.axis=axis; self.left=left; self.right=right


class KDTree:
    """Median-split KD-Tree. Euclidean queries use branch pruning."""
    def __init__(self, items, dims=DIMS):
        self.dims=dims
        self.root=self._build(list(items),0)
    def _build(self, items, depth):
        if not items:return None
        axis=depth%self.dims
        items.sort(key=lambda x:x["embedding"][axis])
        mid=len(items)//2
        return KDNode(items[mid],axis,self._build(items[:mid],depth+1),self._build(items[mid+1:],depth+1))
    def search(self, q, k, metric):
        if metric!="euclidean":
            return sorted((distance(q,x["embedding"],metric),x["id"]) for x in self._all())[:k]
        best=[]
        def visit(node):
            if node is None:return
            item=node.item; axis=node.axis
            d=distance(q,item["embedding"],metric)
            best.append((d,item["id"])); best.sort()
            if len(best)>k:best.pop()
            diff=q[axis]-item["embedding"][axis]
            near,far=(node.left,node.right) if diff<0 else (node.right,node.left)
            visit(near)
            if len(best)<k or abs(diff)<best[-1][0]:visit(far)
        visit(self.root)
        return best
    def _all(self):
        out=[]
        def walk(node):
            if node:
                out.append(node.item);walk(node.left);walk(node.right)
        walk(self.root)
        return out


class HNSW:
    """Pure-Python layered neighbor graph for educational approximate search."""
    def __init__(self,m=8):
        self.m=m;self.nodes={};self.level={};self.links=defaultdict(lambda:defaultdict(set));self.entry=None;self.top=-1
    def insert(self,item):
        idx=item["id"];level=0
        while random.random()<.5 and level<8:level+=1
        self.nodes[idx]=item;self.level[idx]=level
        if self.entry is None:self.entry=idx;self.top=level;return
        for layer in range(level+1):
            near=sorted((distance(item["embedding"],n["embedding"]),other)
                for other,n in self.nodes.items() if other!=idx and self.level[other]>=layer)[:self.m]
            for _,other in near:
                self.links[idx][layer].add(other);self.links[other][layer].add(idx)
                if len(self.links[other][layer])>2*self.m:
                    self.links[other][layer]=set(sorted(self.links[other][layer],
                        key=lambda x:distance(self.nodes[other]["embedding"],self.nodes[x]["embedding"]))[:2*self.m])
        if level>self.top:self.entry=idx;self.top=level
    def remove(self,idx):
        self.nodes.pop(idx,None);self.level.pop(idx,None);self.links.pop(idx,None)
        for layers in self.links.values():
            for neighbors in layers.values():neighbors.discard(idx)
        if self.entry==idx:
            self.entry=max(self.nodes,key=lambda x:self.level[x]) if self.nodes else None
            self.top=self.level[self.entry] if self.entry is not None else -1
    def search(self,q,k,metric):
        if not self.nodes:return []
        current=self.entry
        for layer in range(self.top,0,-1):
            changed=True
            while changed:
                changed=False;best=distance(q,self.nodes[current]["embedding"],metric)
                for other in self.links[current][layer]:
                    d=distance(q,self.nodes[other]["embedding"],metric)
                    if d<best:current=other;best=d;changed=True
        seen={current};queue=[current]
        while queue and len(seen)<max(50,k*8):
            node=queue.pop(0)
            for other in self.links[node][0]:
                if other not in seen:seen.add(other);queue.append(other)
        return sorted((distance(q,self.nodes[i]["embedding"],metric),i) for i in seen if i in self.nodes)[:k]


class VectorDB:
    def __init__(self):
        self.items={};self.next_id=1;self.hnsw=HNSW();self.kdtree=KDTree([])
        rows=json.loads((ROOT/"demo_vectors.json").read_text(encoding="utf-8"))
        for row in rows:self.insert(row["metadata"],row["category"],row["embedding"])
    def insert(self,meta,category,emb):
        idx=self.next_id;self.next_id+=1
        item={"id":idx,"metadata":meta,"category":category,"embedding":[float(x) for x in emb]}
        self.items[idx]=item;self.hnsw.insert(item);self.kdtree=KDTree(self.items.values());return idx
    def remove(self,idx):
        if idx not in self.items:return False
        del self.items[idx];self.hnsw.remove(idx);self.kdtree=KDTree(self.items.values());return True
    def search(self,q,k,metric,algo):
        start=time.perf_counter_ns()
        if algo=="hnsw":pairs=self.hnsw.search(q,k,metric)
        elif algo=="kdtree":pairs=self.kdtree.search(q,k,metric)
        else:pairs=sorted((distance(q,x["embedding"],metric),idx) for idx,x in self.items.items())[:k]
        return {"results":[dict(self.items[i],distance=round(d,6)) for d,i in pairs if i in self.items],
                "latencyUs":(time.perf_counter_ns()-start)//1000,"algo":algo,"metric":metric}


class DocumentDB:
    """Document chunks indexed by a separate HNSW graph (supports 768D embeddings)."""
    def __init__(self):
        self.docs={};self.next_id=1;self.dims=0;self.hnsw=HNSW(m=16)
    def insert(self,title,text,embedding):
        idx=self.next_id;self.next_id+=1;self.dims=len(embedding)
        item={"id":idx,"title":title,"text":text,"embedding":[float(x) for x in embedding]}
        self.docs[idx]=item
        self.hnsw.insert(item)
        return idx
    def remove(self,idx):
        if idx not in self.docs:return False
        self.docs.pop(idx,None);self.hnsw.remove(idx)
        return True
    def search(self,q,k):
        if not self.docs or len(q)!=self.dims:return []
        pairs=self.hnsw.search(q,k,"cosine")
        return [(dist,self.docs[idx]) for dist,idx in pairs if idx in self.docs]


db=VectorDB();docdb=DocumentDB()


def ollama_post(path,payload,timeout=120):
    req=urllib.request.Request(OLLAMA_URL+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.loads(response.read().decode())
    except (urllib.error.URLError,TimeoutError,ValueError):return None

def ollama_available():
    try:
        with urllib.request.urlopen(OLLAMA_URL+"/api/tags",timeout=2) as response:return response.status==200
    except Exception:return False

def embed(text):
    """Create an embedding with the current Ollama API, with legacy fallback."""
    result=ollama_post("/api/embed",{"model":EMBED_MODEL,"input":text})
    if result:
        vectors=result.get("embeddings")
        if isinstance(vectors,list) and vectors and isinstance(vectors[0],list):
            return vectors[0]
        # Be tolerant of compatible Ollama versions returning a singular vector.
        if isinstance(result.get("embedding"),list):
            return result["embedding"]
    result=ollama_post("/api/embeddings",{"model":EMBED_MODEL,"prompt":text})
    return result.get("embedding",[]) if result else []

def generate(prompt):
    result=ollama_post("/api/generate",{"model":GEN_MODEL,"prompt":prompt,"stream":False})
    return result.get("response","") if result else "Ollama is unavailable or the generation model is not installed."


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"]="*"
    response.headers["Access-Control-Allow-Methods"]="GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"]="Content-Type"
    return response

@app.route("/",methods=["GET"])
def home():return send_from_directory(ROOT,"index.html")

@app.route("/<path:path>",methods=["OPTIONS"])
def preflight(path):return ("",204)

@app.get("/items")
def items():return jsonify(list(db.items.values()))

@app.get("/search")
def search():
    try:q=[float(x) for x in request.args.get("v","").split(",") if x.strip()]
    except ValueError:return jsonify(error="invalid vector"),400
    if len(q)!=DIMS:return jsonify(error=f"need {DIMS}D vector"),400
    k=max(1,min(100,int(request.args.get("k",5))));metric=request.args.get("metric","cosine");algo=request.args.get("algo","hnsw")
    if metric not in ("cosine","euclidean","manhattan"):metric="cosine"
    if algo not in ("hnsw","kdtree","bruteforce"):algo="hnsw"
    return jsonify(db.search(q,k,metric,algo))

@app.post("/insert")
def insert():
    data=request.get_json(silent=True) or {};meta=str(data.get("metadata","")).strip();cat=str(data.get("category","")).strip();emb=data.get("embedding")
    if not meta or not isinstance(emb,list) or len(emb)!=DIMS:return jsonify(error="invalid body"),400
    try:emb=[float(x) for x in emb]
    except (ValueError,TypeError):return jsonify(error="invalid embedding"),400
    return jsonify(id=db.insert(meta,cat,emb))

@app.delete("/delete/<int:idx>")
def delete(idx):return jsonify(ok=db.remove(idx))

@app.get("/benchmark")
def benchmark():
    try:q=[float(x) for x in request.args.get("v","").split(",") if x.strip()]
    except ValueError:return jsonify(error="invalid vector"),400
    if len(q)!=DIMS:return jsonify(error=f"need {DIMS}D vector"),400
    k=max(1,min(100,int(request.args.get("k",5))));metric=request.args.get("metric","cosine");result={}
    for algo,key in (("bruteforce","bruteforceUs"),("kdtree","kdtreeUs"),("hnsw","hnswUs")):result[key]=db.search(q,k,metric,algo)["latencyUs"]
    result["itemCount"]=len(db.items);return jsonify(result)

@app.get("/hnsw-info")
def hnsw_info():
    h=db.hnsw;top=max(h.top,0);nodes_per=[sum(h.level[i]>=layer for i in h.nodes) for layer in range(top+1)];edges=[]
    for idx,layers in h.links.items():
        for layer,neighbors in layers.items():
            for other in neighbors:
                if idx<other:edges.append({"src":idx,"dst":other,"lyr":layer})
    return jsonify(topLayer=h.top,nodeCount=len(h.nodes),nodesPerLayer=nodes_per,
        edgesPerLayer=[sum(e["lyr"]==layer for e in edges) for layer in range(top+1)],
        nodes=[{"id":i,"metadata":x["metadata"],"category":x["category"],"maxLyr":h.level[i]} for i,x in h.nodes.items()],edges=edges)

@app.get("/stats")
def stats():return jsonify(count=len(db.items),dims=DIMS,algorithms=["bruteforce","kdtree","hnsw"],metrics=["euclidean","cosine","manhattan"])

@app.post("/doc/insert")
def doc_insert():
    data=request.get_json(silent=True) or {};title=str(data.get("title","")).strip();text=str(data.get("text","")).strip()
    if not title or not text:return jsonify(error="need title and text"),400
    chunks=chunk_text(text)
    if not chunks:return jsonify(error="document text is empty"),400
    # Embed every chunk before mutating the store. A model/network failure must
    # not leave a partially inserted document behind.
    vectors=[]
    for chunk in chunks:
        vector=embed(chunk)
        if not vector:
            return jsonify(error="Ollama embedding failed. Start Ollama and run: ollama pull nomic-embed-text"),503
        if vectors and len(vector)!=len(vectors[0]):
            return jsonify(error="Ollama returned inconsistent embedding dimensions"),502
        vectors.append(vector)
    ids=[]
    for i,(chunk,vector) in enumerate(zip(chunks,vectors)):
        label=f"{title} [{i+1}/{len(chunks)}]" if len(chunks)>1 else title
        ids.append(docdb.insert(label,chunk,vector))
    return jsonify(ids=ids,chunks=len(chunks),dims=docdb.dims)

@app.delete("/doc/delete/<int:idx>")
def doc_delete(idx):return jsonify(ok=docdb.remove(idx))

@app.get("/doc/list")
def doc_list():
    return jsonify([{"id":d["id"],"title":d["title"],"preview":d["text"][:120]+("…" if len(d["text"])>120 else ""),"words":len(d["text"].split())} for d in docdb.docs.values()])

def retrieve(question,k):
    vector=embed(question)
    return (None,None) if not vector else (vector,docdb.search(vector,k))

@app.post("/doc/search")
def doc_search():
    data=request.get_json(silent=True) or {};question=str(data.get("question","")).strip();k=max(1,min(20,int(data.get("k",3))))
    if not question:return jsonify(error="need question"),400
    vector,hits=retrieve(question,k)
    if vector is None:return jsonify(error="Ollama unavailable"),503
    return jsonify(contexts=[{"id":d["id"],"title":d["title"],"distance":round(dist,4)} for dist,d in hits])

@app.post("/doc/ask")
def doc_ask():
    data=request.get_json(silent=True) or {};question=str(data.get("question","")).strip();k=max(1,min(20,int(data.get("k",3))))
    if not question:return jsonify(error="need question"),400
    vector,hits=retrieve(question,k)
    if vector is None:return jsonify(error="Ollama unavailable"),503
    context="".join(f"[{i+1}] {d['title']}:\n{d['text']}\n\n" for i,(_,d) in enumerate(hits))
    prompt=("You are a helpful assistant. Answer directly. Use the retrieved context if relevant; otherwise use general knowledge. Do not mention the context or provided text.\n\nContext:\n"+context+f"Question: {question}\n\nAnswer:")
    return jsonify(answer=generate(prompt),model=GEN_MODEL,contexts=[{"id":d["id"],"title":d["title"],"text":d["text"],"distance":round(dist,4)} for dist,d in hits],docCount=len(docdb.docs))

@app.get("/status")
def status():
    return jsonify(ollamaAvailable=ollama_available(),embedModel=EMBED_MODEL,genModel=GEN_MODEL,docCount=len(docdb.docs),docDims=docdb.dims,demoDims=DIMS,demoCount=len(db.items))

if __name__=="__main__":
    print("=== My_personal_AI Engine ===\nhttp://localhost:8080")
    print(f"{len(db.items)} demo vectors | {DIMS} dims | HNSW + KD-Tree + Brute Force")
    print("Ollama:","ONLINE" if ollama_available() else "OFFLINE (install from ollama.com)")
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","8080")),debug=False)
