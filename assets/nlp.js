(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // Simplified browser tokenization demo. Python examples use the real jieba package.
  const dictionary = [
    '传智教育','上市公司','黑马程序员','人工智能','自然语言处理','机器学习','深度学习',
    '词嵌入','循环神经网络','文本生成','搜索引擎','品牌','旗下','学习','一家','这里','服务','拉面','好吃'
  ].sort((a,b) => b.length - a.length);
  const punctuation = new Set('，。！？、；：,.!?;:()（）'.split(''));
  function preciseSegment(text){
    const out=[]; let i=0;
    while(i<text.length){
      const ch=text[i];
      if(/\s/.test(ch)){i+=1;continue;}
      if(punctuation.has(ch)){out.push(ch);i+=1;continue;}
      const match=dictionary.find(w=>text.startsWith(w,i));
      if(match){out.push(match);i+=match.length;}else{out.push(ch);i+=1;}
    }
    return out;
  }
  function fullSegment(text){
    const out=[];
    for(let i=0;i<text.length;i+=1){
      if(/\s/.test(text[i])||punctuation.has(text[i])) continue;
      const matches=dictionary.filter(w=>text.startsWith(w,i));
      if(matches.length) out.push(...matches); else out.push(text[i]);
    }
    return out;
  }
  function searchSegment(text){
    const base=preciseSegment(text), out=[];
    base.forEach(token=>{
      if(token.length>=4){for(let n=2;n<=3;n+=1)for(let i=0;i<=token.length-n;i+=1)out.push(token.slice(i,i+n));}
      out.push(token);
    });
    return out;
  }
  if($('tokenizeInput')){
    const render=()=>{
      const mode=$('tokenizeMode').value, text=$('tokenizeInput').value;
      const tokens=mode==='full'?fullSegment(text):mode==='search'?searchSegment(text):preciseSegment(text);
      $('tokenizeOutput').innerHTML=tokens.map((t,i)=>`<span class="token-chip ${i%3===1?'alt':''}">${t.replace(/[<&]/g,s=>s==='<'?'&lt;':'&amp;')}</span>`).join('');
      $('tokenizeCount').textContent=String(tokens.length);
      $('tokenizeExplain').textContent=mode==='precise'?'精确模式尽量得到最合理的切分，适合一般文本分析。':mode==='full'?'全模式列出尽可能多的候选词，召回高但会有重叠。':'搜索模式在精确切分上补充较短子词，便于建立倒排索引。';
    };
    $('tokenizeMode').addEventListener('change',render);$('tokenizeInput').addEventListener('input',render);$('tokenizeRun').addEventListener('click',render);render();
  }

  // Corpus analysis chart.
  const corpus=[
    {label:'正向',text:'这家 拉面 很 好吃 服务 热情'},
    {label:'正向',text:'价格 实惠 环境 干净 还会 再来'},
    {label:'负向',text:'等待 时间 太 长 服务 一般'},
    {label:'负向',text:'味道 偏 咸 分量 也 少'},
    {label:'正向',text:'店员 很 耐心 推荐 合适'},
    {label:'负向',text:'上菜 太 慢 体验 不好'},
    {label:'正向',text:'位置 方便 整体 满意'},
    {label:'负向',text:'价格 偏 高 环境 嘈杂'}
  ];
  if($('corpusCanvas')){
    const c=$('corpusCanvas'),ctx=c.getContext('2d'); let mode='label';
    const tokenized=corpus.map(x=>({...x,tokens:x.text.split(' ')}));
    function drawBars(items,max,title){
      ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);
      ctx.fillStyle='#202633';ctx.font='bold 15px sans-serif';ctx.fillText(title,18,24);
      const left=55,bottom=35,top=42,w=c.width-left-25,h=c.height-top-bottom;
      ctx.strokeStyle='#d8e0ec';ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,c.height-bottom);ctx.lineTo(c.width-20,c.height-bottom);ctx.stroke();
      const bw=Math.min(90,w/(items.length*1.6));
      items.forEach((item,i)=>{const x=left+(i+.5)*w/items.length-bw/2,bh=(item.value/max)*h;ctx.fillStyle=i%2?'#d93b3b':'#3169ff';ctx.fillRect(x,c.height-bottom-bh,bw,bh);ctx.fillStyle='#445068';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText(item.name,x+bw/2,c.height-bottom+18);ctx.fillText(String(item.value),x+bw/2,c.height-bottom-bh-7);});ctx.textAlign='left';
    }
    function render(){
      const labels=tokenized.reduce((a,x)=>(a[x.label]=(a[x.label]||0)+1,a),{}),lengths=tokenized.map(x=>x.tokens.length),freq={};tokenized.forEach(x=>x.tokens.forEach(t=>freq[t]=(freq[t]||0)+1));
      const vocab=Object.keys(freq).length,avg=lengths.reduce((a,b)=>a+b,0)/lengths.length,top=Object.entries(freq).sort((a,b)=>b[1]-a[1])[0];
      $('corpusBalance').textContent=`${labels['正向']}:${labels['负向']}`;$('corpusAvgLen').textContent=avg.toFixed(1);$('corpusVocab').textContent=String(vocab);$('corpusTopWord').textContent=`${top[0]} × ${top[1]}`;
      if(mode==='label')drawBars(Object.entries(labels).map(([name,value])=>({name,value})),Math.max(...Object.values(labels)),'标签数量分布');
      else{const counts={};lengths.forEach(n=>counts[n]=(counts[n]||0)+1);drawBars(Object.entries(counts).map(([name,value])=>({name:`${name}词`,value})),Math.max(...Object.values(counts)),'句子长度分布');}
      $('corpusLabelBtn').classList.toggle('active',mode==='label');$('corpusLengthBtn').classList.toggle('active',mode==='length');
    }
    $('corpusLabelBtn').addEventListener('click',()=>{mode='label';render();});$('corpusLengthBtn').addEventListener('click',()=>{mode='length';render();});render();
  }

  // n-gram demo.
  if($('ngramTokens')){
    const tokens=['自然','语言','处理','让','机器','理解','文本'];let n=2;
    function render(){const grams=[];for(let i=0;i<=tokens.length-n;i+=1)grams.push(tokens.slice(i,i+n).join(' · '));$('ngramTokens').innerHTML=tokens.map(t=>`<span class="token-chip">${t}</span>`).join('');$('ngramOutput').innerHTML=grams.map(g=>`<span class="token-chip alt">${g}</span>`).join('');$('ngramCount').textContent=String(grams.length);document.querySelectorAll('[data-ngram]').forEach(b=>b.classList.toggle('active',+b.dataset.ngram===n));}
    document.querySelectorAll('[data-ngram]').forEach(b=>b.addEventListener('click',()=>{n=+b.dataset.ngram;render();}));render();
  }

  // Padding and truncation demo.
  if($('paddingMax')){
    const samples=[['我','喜欢','自然','语言','处理'],['循环','神经','网络','可以','建模','序列','上下文'],['短句']];
    function render(){const maxLen=+$('paddingMax').value,side=$('paddingSide').value;$('paddingMaxVal').textContent=String(maxLen);$('paddingRows').innerHTML=samples.map((seq,idx)=>{let kept=seq.slice(0,maxLen),removed=seq.slice(maxLen),pads=Array(Math.max(0,maxLen-kept.length)).fill('<PAD>');const final=side==='pre'?[...pads,...kept]:[...kept,...pads],mask=final.map(t=>t==='<PAD>'?0:1);return `<div class="sequence-row"><strong>样本 ${idx+1}</strong><div><div class="sequence-cells">${final.map(t=>`<span class="token-chip ${t==='<PAD>'?'pad':''}">${t}</span>`).join('')}${removed.length?`<span class="token-chip truncated">截断 ${removed.length} 词</span>`:''}</div><div class="sequence-cells mask-cells" style="margin-top:7px">${mask.map(v=>`<span class="token-chip ${v?'alt':'pad'}">${v}</span>`).join('')}</div></div></div>`;}).join('');}
    $('paddingMax').addEventListener('input',render);$('paddingSide').addEventListener('change',render);render();
  }

  // Sparse encoding demo.
  if($('encodingMode')){
    const docs=[['拉面','好吃','服务'],['服务','一般','等待'],['拉面','实惠','环境']],vocab=['拉面','好吃','服务','一般','等待','实惠','环境'];
    function render(){const mode=$('encodingMode').value,N=docs.length,df=Object.fromEntries(vocab.map(t=>[t,docs.filter(d=>d.includes(t)).length]));const matrix=docs.map(d=>vocab.map(t=>{const count=d.filter(x=>x===t).length;if(mode==='binary')return count?1:0;if(mode==='count')return count;const tf=count;const idf=Math.log((N+1)/(df[t]+1))+1;return tf*idf;}));let out='<table class="encoding-matrix"><thead><tr><th>文档</th>'+vocab.map(t=>`<th>${t}</th>`).join('')+'</tr></thead><tbody>';matrix.forEach((row,i)=>{out+=`<tr><th>D${i+1}</th>`+row.map(v=>`<td class="${v>0?'hot':''}">${mode==='tfidf'?v.toFixed(2):v}</td>`).join('')+'</tr>';});out+='</tbody></table>';$('encodingMatrix').innerHTML=out;$('encodingExplain').textContent=mode==='binary'?'二值词袋只记录“是否出现”，矩阵通常高度稀疏。':mode==='count'?'词频编码记录每个词在文档中的出现次数，但高频常用词可能占据优势。':'TF-IDF 同时考虑文档内词频和跨文档稀有度，降低普遍词的权重。';}
    $('encodingMode').addEventListener('change',render);render();
  }

  // Embedding lookup and cosine similarity.
  if($('embeddingToken')){
    const vectors={拉面:[.55,.35,.08,-.12,.22,.05],面条:[.50,.31,.04,-.08,.18,.09],好吃:[.12,.08,.72,.45,-.10,.22],美味:[.10,.05,.76,.41,-.08,.25],服务:[-.20,.42,.12,.18,.52,-.16],等待:[-.35,.20,-.28,.12,.15,.55]};
    const cosine=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0)/(Math.hypot(...a)*Math.hypot(...b));
    function render(){const token=$('embeddingToken').value,v=vectors[token],max=Math.max(...v.map(Math.abs),.01);$('embeddingBars').innerHTML=v.map((x,i)=>{const width=Math.abs(x)/max*50,left=x>=0?50:50-width;return `<div class="vector-row"><span>d${i+1}</span><div class="vector-track"><i class="vector-fill" style="left:${left}%;width:${width}%"></i></div><b>${x.toFixed(2)}</b></div>`;}).join('');const near=Object.entries(vectors).filter(([t])=>t!==token).map(([t,w])=>[t,cosine(v,w)]).sort((a,b)=>b[1]-a[1])[0];$('embeddingNearest').textContent=`${near[0]} (${near[1].toFixed(3)})`;$('embeddingShape').textContent=`[1, ${v.length}]`;}
    $('embeddingToken').innerHTML=Object.keys(vectors).map(t=>`<option>${t}</option>`).join('');$('embeddingToken').addEventListener('change',render);render();
  }

  // Vanilla RNN hidden-state animation.
  if($('rnnUnroll')){
    const tokens=['春','风','轻','轻','吹'],inputs=[.25,.7,-.15,.45,.2];let states=[],step=0,timer=null;
    function calculate(){states=[];let h=0;inputs.forEach(x=>{h=Math.tanh(.72*h+1.1*x-.05);states.push(h);});}
    function render(){calculate();$('rnnUnroll').innerHTML=tokens.map((t,i)=>`<div class="rnn-step-node ${i===step?'current':''}"><span>x${i+1}=${t}</span><b>h${i+1}=${states[i].toFixed(3)}</b></div>${i<tokens.length-1?'<span class="rnn-arrow">→</span>':''}`).join('');$('rnnCurrentToken').textContent=tokens[step];$('rnnCurrentH').textContent=states[step].toFixed(4);$('rnnPrevH').textContent=(step?states[step-1]:0).toFixed(4);$('rnnStepVal').textContent=String(step+1);}
    $('rnnStepBtn').addEventListener('click',()=>{step=(step+1)%tokens.length;render();});$('rnnResetBtn').addEventListener('click',()=>{step=0;render();});$('rnnPlayBtn').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;$('rnnPlayBtn').textContent='播放';return;}$('rnnPlayBtn').textContent='暂停';timer=setInterval(()=>{step=(step+1)%tokens.length;render();},650);});render();
  }

  // LSTM gate demo.
  if($('lstmForget')){
    const ids=['lstmForget','lstmInput','lstmOutput','lstmPrev','lstmCandidate'];
    function render(){const f=+$('lstmForget').value/100,i=+$('lstmInput').value/100,o=+$('lstmOutput').value/100,cPrev=+$('lstmPrev').value/100,g=+$('lstmCandidate').value/100,c=f*cPrev+i*g,h=o*Math.tanh(c);$('lstmFVal').textContent=f.toFixed(2);$('lstmIVal').textContent=i.toFixed(2);$('lstmOVal').textContent=o.toFixed(2);$('lstmPrevVal').textContent=cPrev.toFixed(2);$('lstmCandVal').textContent=g.toFixed(2);$('lstmCellVal').textContent=c.toFixed(3);$('lstmHiddenVal').textContent=h.toFixed(3);$('lstmForgetMeter').style.width=`${f*100}%`;$('lstmInputMeter').style.width=`${i*100}%`;$('lstmOutputMeter').style.width=`${o*100}%`;}
    ids.forEach(id=>$(id).addEventListener('input',render));render();
  }

  // PyTorch RNN shape calculator.
  if($('rnnShapeCell')){
    const ids=['rnnShapeCell','rnnShapeBatch','rnnShapeSeq','rnnShapeInput','rnnShapeHidden','rnnShapeLayers','rnnShapeBi'];
    function render(){const cell=$('rnnShapeCell').value,B=+$('rnnShapeBatch').value,L=+$('rnnShapeSeq').value,I=+$('rnnShapeInput').value,H=+$('rnnShapeHidden').value,N=+$('rnnShapeLayers').value,D=$('rnnShapeBi').checked?2:1,gates=cell==='lstm'?4:cell==='gru'?3:1;let params=0,layerIn=I;for(let layer=0;layer<N;layer+=1){params+=D*gates*(H*layerIn+H*H+2*H);layerIn=D*H;}$('rnnOutputShape').textContent=`[${B}, ${L}, ${D*H}]`;$('rnnHiddenShape').textContent=`[${D*N}, ${B}, ${H}]`;$('rnnCellShape').textContent=cell==='lstm'?`[${D*N}, ${B}, ${H}]`:'—';$('rnnParamCount').textContent=params.toLocaleString();}
    ids.forEach(id=>$(id).addEventListener(id==='rnnShapeBi'?'change':'input',render));$('rnnShapeCell').addEventListener('change',render);render();
  }

  // Toy text generation with temperature.
  if($('generatorStart')){
    const graph={
      '春':[['风',3.2],['天',1.8],['光',1.2]],
      '风':[['轻',3.1],['吹',2.0],['来',1.0]],
      '轻':[['轻',2.4],['落',1.8],['过',1.4]],
      '吹':[['过',2.7],['来',1.8],['向',1.1]],
      '过':[['湖',2.8],['山',1.9],['夜',1.2]],
      '湖':[['面',3.0],['边',1.5],['光',1.1]],
      '面':[['泛',2.6],['上',2.0],['静',1.0]],
      '星':[['光',3.1],['河',1.6],['落',1.4]],
      '光':[['落',2.7],['照',2.1],['闪',1.1]],
      '落':[['在',3.2],['下',1.3],['满',1.0]],
      '在':[['安',2.4],['湖',1.8],['夜',1.5]],
      '安':[['静',3.2],['然',1.2]],
      '静':[['的',3.0],['夜',1.7]],
      '的':[['夜',2.9],['风',1.8],['光',1.4]],
      '夜':[['里',3.0],['色',1.7],['风',1.2]],
      '里':[['。',3.0],['星',1.4],['风',1.0]],
      '天':[['空',2.8],['边',1.5]],
      '空':[['很',2.0],['有',1.4]],
      '很':[['蓝',2.2],['静',1.6]],
      '蓝':[['。',2.8]],
      '河':[['缓',2.2],['亮',1.4]],
      '缓':[['缓',2.3],['流',1.7]],
      '流':[['动',2.5]],
      '动':[['。',2.8]],
      '山':[['谷',2.3],['间',1.8]],
      '谷':[['回',2.2]],
      '回':[['响',2.6]],
      '响':[['。',2.8]],
      '。':[['春',2.0],['星',1.7]]
    };
    function probs(items,temp){const ex=items.map(x=>Math.exp(x[1]/temp)),sum=ex.reduce((a,b)=>a+b,0);return items.map((x,i)=>[x[0],ex[i]/sum]);}
    function sample(items){let r=Math.random();for(const item of items){r-=item[1];if(r<=0)return item[0];}return items.at(-1)[0];}
    function renderProbs(items){$('generatorProbs').innerHTML=items.map(([t,p])=>`<div class="generation-prob"><b>${t}</b><div class="prob-bar"><i style="width:${p*100}%"></i></div><span>${(p*100).toFixed(1)}%</span></div>`).join('');}
    function generate(){const temp=+$('generatorTemp').value/100,len=+$('generatorLength').value;let token=$('generatorStart').value,text=token,first=[];for(let i=1;i<len;i+=1){const items=probs(graph[token]||graph['。'],temp);if(i===1)first=items;token=sample(items);text+=token;if(token==='。'&&i>8)break;}$('generatorTempVal').textContent=temp.toFixed(2);$('generatorLengthVal').textContent=String(len);$('generatedText').textContent=text;renderProbs(first);}
    $('generatorStart').addEventListener('change',generate);$('generatorTemp').addEventListener('input',generate);$('generatorLength').addEventListener('input',generate);$('generatorRun').addEventListener('click',generate);generate();
  }
})();
