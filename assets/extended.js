(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // ---------- Decision tree split demo ----------
  const treeCanvas = $('treeCanvas');
  if (treeCanvas) {
    const ctx = treeCanvas.getContext('2d');
    const slider = $('treeThreshold');
    const samples = [
      {x:1.0,y:1},{x:1.5,y:1},{x:2.1,y:1},{x:2.7,y:0},
      {x:3.1,y:1},{x:3.6,y:0},{x:4.0,y:0},{x:4.4,y:1},
      {x:4.9,y:0},{x:5.4,y:0},{x:5.8,y:0}
    ];
    const gini = (arr) => {
      if (!arr.length) return 0;
      const p = arr.reduce((s, v) => s + v.y, 0) / arr.length;
      return 1 - p*p - (1-p)*(1-p);
    };
    const parentGini = gini(samples);
    const px = (x) => 60 + (x / 6.4) * (treeCanvas.width - 110);
    function renderTreeSplit() {
      const t = Number(slider.value) / 10;
      const left = samples.filter(s => s.x <= t);
      const right = samples.filter(s => s.x > t);
      const weighted = (left.length * gini(left) + right.length * gini(right)) / samples.length;
      const gain = parentGini - weighted;
      ctx.clearRect(0,0,treeCanvas.width,treeCanvas.height);
      ctx.fillStyle = '#fff'; ctx.fillRect(0,0,treeCanvas.width,treeCanvas.height);
      ctx.strokeStyle = '#d8e0ec'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(50, 195); ctx.lineTo(treeCanvas.width-35,195); ctx.stroke();
      for (let i=1;i<=6;i++) {
        const x=px(i); ctx.beginPath(); ctx.moveTo(x,190); ctx.lineTo(x,201); ctx.stroke();
        ctx.fillStyle='#667188'; ctx.font='12px sans-serif'; ctx.fillText(String(i),x-3,218);
      }
      samples.forEach(s => {
        const y = s.y ? 72 : 145;
        ctx.beginPath(); ctx.arc(px(s.x),y,10,0,Math.PI*2);
        ctx.fillStyle = s.y ? '#3169ff' : '#d93b3b'; ctx.fill();
        ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
      });
      const tx = px(t);
      ctx.save(); ctx.setLineDash([7,5]); ctx.strokeStyle='#f0a500'; ctx.lineWidth=3;
      ctx.beginPath(); ctx.moveTo(tx,35); ctx.lineTo(tx,190); ctx.stroke(); ctx.restore();
      ctx.fillStyle='#202633'; ctx.font='13px sans-serif';
      ctx.fillText('类别 1',10,76); ctx.fillText('类别 0',10,149);
      ctx.fillText(`threshold = ${t.toFixed(1)}`, clamp(tx-45,55,treeCanvas.width-150),28);
      $('treeThresholdVal').textContent=t.toFixed(1);
      $('treeLeftCount').textContent=left.length;
      $('treeRightCount').textContent=right.length;
      $('treeGini').textContent=weighted.toFixed(3);
      $('treeGain').textContent=gain.toFixed(3);
    }
    slider.addEventListener('input',renderTreeSplit); renderTreeSplit();
  }

  // ---------- Bagging / Boosting switch ----------
  const ensembleDiagram = $('ensembleDiagram');
  if (ensembleDiagram) {
    const bagBtn=$('ensembleBag'), boostBtn=$('ensembleBoost'), desc=$('ensembleDescription');
    function renderEnsemble(mode) {
      bagBtn.classList.toggle('active',mode==='bag'); boostBtn.classList.toggle('active',mode==='boost');
      if (mode==='bag') {
        ensembleDiagram.innerHTML = '<div class="learner">抽样集 A<br>Tree 1</div><div class="learner">抽样集 B<br>Tree 2</div><div class="learner">抽样集 C<br>Tree 3</div><div class="arrow">→</div><div class="vote">多数投票 / 平均</div>';
        desc.textContent = 'Bagging：各模型使用有放回抽样得到的训练集，彼此独立，可并行训练。分类任务平权投票，回归任务取平均。核心目标是降低方差。';
      } else {
        ensembleDiagram.innerHTML = '<div class="learner focus">Model 1<br>初始预测</div><div class="arrow">→</div><div class="learner focus">Model 2<br>修正误差</div><div class="arrow">→</div><div class="learner focus">Model 3<br>继续修正</div><div class="arrow">→</div><div class="vote">加权累加</div>';
        desc.textContent = 'Boosting：模型串行训练，后一个学习器针对前一轮不足继续学习。最终结果是逐轮加权累加，核心目标是持续降低偏差。';
      }
    }
    bagBtn.addEventListener('click',()=>renderEnsemble('bag'));
    boostBtn.addEventListener('click',()=>renderEnsemble('boost'));
    renderEnsemble('bag');
  }

  // ---------- KMeans demo and elbow chart ----------
  const kmCanvas = $('kmeansCanvas');
  if (kmCanvas) {
    const ctx=kmCanvas.getContext('2d'), kSlider=$('kmeansK');
    const pts=[
      [1.0,1.2],[1.2,1.7],[1.4,1.1],[1.7,1.5],[2.0,1.2],[1.8,2.0],[2.2,1.8],[1.1,2.2],
      [3.5,4.2],[3.8,4.7],[4.1,4.1],[4.4,4.8],[4.6,4.2],[4.0,5.2],[4.8,5.0],[3.4,5.0],
      [6.5,1.1],[6.8,1.5],[7.1,1.0],[7.4,1.7],[7.7,1.3],[6.9,2.1],[7.5,2.2],[6.3,1.9],
      [6.0,5.5],[6.5,5.9],[7.0,5.4],[7.5,5.8],[7.8,5.2],[6.8,6.4]
    ];
    const colors=['#3169ff','#d93b3b','#14865d','#7d57ff','#f0a500','#159bb6'];
    let centers=[], labels=[], phase='assign', iter=0, timer=null;
    const sx=x=>55+(x/8.5)*(kmCanvas.width-100), sy=y=>kmCanvas.height-45-(y/7)*(kmCanvas.height-85);
    function initCenters(k) {
      const indexes=[0,8,16,24,28];
      centers=indexes.slice(0,k).map(i=>pts[i].slice());
      labels=new Array(pts.length).fill(-1); phase='assign'; iter=0;
    }
    const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
    function assign() { labels=pts.map(p=>{let best=0,bd=Infinity;centers.forEach((c,i)=>{const d=distance(p,c);if(d<bd){bd=d;best=i;}});return best;}); phase='update'; }
    function update() {
      const next=centers.map((c,j)=>{const members=pts.filter((_,i)=>labels[i]===j);if(!members.length)return c.slice();return [members.reduce((s,p)=>s+p[0],0)/members.length,members.reduce((s,p)=>s+p[1],0)/members.length];});
      centers=next; iter+=1; phase='assign';
    }
    function sse() { return pts.reduce((sum,p,i)=>labels[i]<0?sum:sum+distance(p,centers[labels[i]])**2,0); }
    function drawKM() {
      ctx.clearRect(0,0,kmCanvas.width,kmCanvas.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,kmCanvas.width,kmCanvas.height);
      ctx.strokeStyle='#e0e6ef';ctx.lineWidth=1;
      for(let i=1;i<=8;i++){ctx.beginPath();ctx.moveTo(sx(i),25);ctx.lineTo(sx(i),kmCanvas.height-45);ctx.stroke();}
      for(let i=1;i<=6;i++){ctx.beginPath();ctx.moveTo(55,sy(i));ctx.lineTo(kmCanvas.width-45,sy(i));ctx.stroke();}
      pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),7,0,Math.PI*2);ctx.fillStyle=labels[i]<0?'#8f9aad':colors[labels[i]%colors.length];ctx.fill();});
      centers.forEach((c,i)=>{const x=sx(c[0]),y=sy(c[1]);ctx.strokeStyle=colors[i];ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(x-10,y-10);ctx.lineTo(x+10,y+10);ctx.moveTo(x+10,y-10);ctx.lineTo(x-10,y+10);ctx.stroke();});
      $('kmeansKVal').textContent=kSlider.value;$('kmeansPhase').textContent=phase==='assign'?'分配样本':'更新质心';$('kmeansIter').textContent=iter;$('kmeansSSE').textContent=sse().toFixed(2);
    }
    function kmStep(){phase==='assign'?assign():update();drawKM();}
    function kmReset(){if(timer){clearInterval(timer);timer=null;$('kmeansPlay').textContent='播放';}initCenters(+kSlider.value);drawKM();}
    $('kmeansStep').addEventListener('click',kmStep);$('kmeansReset').addEventListener('click',kmReset);kSlider.addEventListener('input',kmReset);
    $('kmeansPlay').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;$('kmeansPlay').textContent='播放';}else{timer=setInterval(kmStep,650);$('kmeansPlay').textContent='暂停';}});
    initCenters(+kSlider.value);drawKM();

    function runK(k){let cs=[0,8,16,24,28,4].slice(0,k).map(i=>pts[i].slice()), ls=[];for(let t=0;t<20;t++){ls=pts.map(p=>{let b=0,bd=Infinity;cs.forEach((c,i)=>{const d=distance(p,c);if(d<bd){bd=d;b=i;}});return b;});const next=cs.map((c,j)=>{const m=pts.filter((_,i)=>ls[i]===j);return m.length?[m.reduce((s,p)=>s+p[0],0)/m.length,m.reduce((s,p)=>s+p[1],0)/m.length]:c;});const delta=next.reduce((s,c,i)=>s+distance(c,cs[i]),0);cs=next;if(delta<1e-5)break;}return pts.reduce((s,p,i)=>s+distance(p,cs[ls[i]])**2,0);}
    const elbow=$('elbowCanvas');
    if(elbow){const ec=elbow.getContext('2d'),vals=[1,2,3,4,5,6].map(k=>runK(k));const max=Math.max(...vals);ec.fillStyle='#fff';ec.fillRect(0,0,elbow.width,elbow.height);ec.strokeStyle='#d8e0ec';ec.beginPath();ec.moveTo(55,20);ec.lineTo(55,elbow.height-45);ec.lineTo(elbow.width-35,elbow.height-45);ec.stroke();const ex=k=>65+(k-1)/5*(elbow.width-120),ey=v=>elbow.height-55-(v/max)*(elbow.height-90);ec.beginPath();vals.forEach((v,i)=>i?ec.lineTo(ex(i+1),ey(v)):ec.moveTo(ex(i+1),ey(v)));ec.strokeStyle='#3169ff';ec.lineWidth=4;ec.stroke();vals.forEach((v,i)=>{ec.beginPath();ec.arc(ex(i+1),ey(v),6,0,Math.PI*2);ec.fillStyle=i===2?'#d93b3b':'#3169ff';ec.fill();ec.fillStyle='#202633';ec.font='12px sans-serif';ec.fillText(`K=${i+1}`,ex(i+1)-13,elbow.height-24);ec.fillText(v.toFixed(1),ex(i+1)-13,ey(v)-10);});}
  }

  // ---------- Convolution demo ----------
  const convInputGrid=$('convInputGrid');
  if(convInputGrid){
    const input=[[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]];
    const kernels={edge:[[1,0,-1],[1,0,-1],[1,0,-1]],sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]],blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]};
    const step=$('convStep'), select=$('convKernel');
    function makeGrid(el,matrix,hotSet,outHot){el.innerHTML='';el.style.gridTemplateColumns=`repeat(${matrix[0].length},42px)`;matrix.flat().forEach((v,idx)=>{const c=document.createElement('div');c.className='gcell'+(hotSet&&hotSet.has(idx)?' hot':'')+(outHot===idx?' out-hot':'');c.textContent=Number.isInteger(v)?v:v.toFixed(2);el.appendChild(c);});}
    function convOutput(kernel){const out=[];for(let r=0;r<3;r++){const row=[];for(let c=0;c<3;c++){let s=0;for(let i=0;i<3;i++)for(let j=0;j<3;j++)s+=input[r+i][c+j]*kernel[i][j];row.push(s);}out.push(row);}return out;}
    function renderConv(){const idx=+step.value,r=Math.floor(idx/3),c=idx%3,k=kernels[select.value],out=convOutput(k),hot=new Set();for(let i=0;i<3;i++)for(let j=0;j<3;j++)hot.add((r+i)*5+c+j);makeGrid(convInputGrid,input,hot,null);makeGrid($('convKernelGrid'),k,null,null);makeGrid($('convOutputGrid'),out,null,idx);let parts=[],sum=0;for(let i=0;i<3;i++)for(let j=0;j<3;j++){const val=input[r+i][c+j]*k[i][j];sum+=val;parts.push(`${input[r+i][c+j]}×${Number(k[i][j].toFixed(2))}`);}$('convCalc').textContent=`输出[${r},${c}] = ${parts.join(' + ')} = ${sum.toFixed(3)}`;$('convStepVal').textContent=`${idx+1} / 9`;}
    step.addEventListener('input',renderConv);select.addEventListener('change',renderConv);renderConv();
  }

  // ---------- Pooling demo ----------
  if($('poolInputGrid')){
    const x=[[1,3,2,0],[4,6,5,1],[2,1,7,3],[0,2,4,8]],mode=$('poolMode');
    function makePoolGrid(el,m){el.innerHTML='';el.style.gridTemplateColumns=`repeat(${m[0].length},42px)`;m.flat().forEach(v=>{const c=document.createElement('div');c.className='gcell';c.textContent=Number.isInteger(v)?v:v.toFixed(2);el.appendChild(c);});}
    function renderPool(){const out=[];for(let r=0;r<2;r++){const row=[];for(let c=0;c<2;c++){const vals=[];for(let i=0;i<2;i++)for(let j=0;j<2;j++)vals.push(x[r*2+i][c*2+j]);row.push(mode.value==='max'?Math.max(...vals):vals.reduce((a,b)=>a+b,0)/vals.length);}out.push(row);}makePoolGrid($('poolInputGrid'),x);makePoolGrid($('poolOutputGrid'),out);}
    mode.addEventListener('change',renderPool);renderPool();
  }

  // ---------- Attention matrix stepper ----------
  if($('attnMatrixStage')){
    const X=[[1,0,1,0],[0,1,0,1],[1,1,0,0]];
    const transpose=m=>m[0].map((_,i)=>m.map(r=>r[i]));
    const matmul=(a,b)=>a.map(r=>transpose(b).map(c=>r.reduce((s,v,i)=>s+v*c[i],0)));
    const scale=(m,s)=>m.map(r=>r.map(v=>v/s));
    const softmax=m=>m.map(r=>{const mx=Math.max(...r),e=r.map(v=>Math.exp(v-mx)),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z);});
    const scores=matmul(X,transpose(X)),scaled=scale(scores,2),weights=softmax(scaled),out=matmul(weights,X);
    function matrixHTML(name,m,note){return `<div class="matrix-card"><h5>${name} · ${m.length}×${m[0].length}</h5><div class="matrix-box" style="grid-template-columns:repeat(${m[0].length},minmax(42px,1fr))">${m.flat().map(v=>`<div class="mcell">${Number(v).toFixed(3)}</div>`).join('')}</div><p>${note||''}</p></div>`;}
    function renderAttnStep(){const s=+$('attnStep').value,stage=$('attnMatrixStage');let h='',label='';if(s===0){label='输入 X';h=matrixHTML('X',X,'L=3，d_model=4');}else if(s===1){label='生成 Q / K / V';h=matrixHTML('Q',X,'示例令 Q=X')+matrixHTML('K',X,'示例令 K=X')+matrixHTML('V',X,'示例令 V=X');}else if(s===2){label='QKᵀ 分数';h=matrixHTML('Scores',scores,'每行表示一个 Query 对所有 Key 的分数');}else if(s===3){label='缩放与 Softmax';h=matrixHTML('Scaled',scaled,'除以 √d_k = 2')+matrixHTML('Weights',weights,'每行权重和为 1');}else{label='Weights × V';h=matrixHTML('Output',out,'得到上下文相关表示');}stage.innerHTML=h;$('attnStepLabel').textContent=label;}
    $('attnStep').addEventListener('input',renderAttnStep);renderAttnStep();
  }

  // ---------- Multi-head attention demo ----------
  if($('headBars')){
    const tokens=['我','喜欢','这家','拉面','因为','很香'];
    const heads=[
      {title:'Head 1：局部搭配',text:'当前 Query“拉面”主要关注相邻的“这家”和“喜欢”，表现局部词组关系。',w:[.05,.16,.24,.32,.13,.10]},
      {title:'Head 2：情感修饰',text:'当前 Query“拉面”更关注“喜欢”和“很香”，捕捉情感或属性修饰。',w:[.04,.28,.08,.18,.08,.34]},
      {title:'Head 3：长距离关联',text:'该 Head 把较远的“我”和“因为”纳入关系，示意长距离依赖。',w:[.25,.08,.08,.18,.31,.10]}
    ];
    function renderHead(i){document.querySelectorAll('.head-focus').forEach((b,j)=>b.classList.toggle('active',j===i));$('headTitle').textContent=heads[i].title;$('headText').textContent=heads[i].text;$('headBars').innerHTML=heads[i].w.map((w,j)=>`<div class="bar-row"><span>${tokens[j]}</span><div class="bar"><i style="width:${w*100}%"></i></div><strong>${w.toFixed(2)}</strong></div>`).join('');}
    document.querySelectorAll('.head-focus').forEach((b,i)=>b.addEventListener('click',()=>renderHead(i)));renderHead(0);
  }

  // ---------- Positional encoding demo ----------
  const peCanvas=$('peCanvas');
  if(peCanvas){
    const ctx=peCanvas.getContext('2d'),slider=$('peDim'),dModel=16;
    const pe=(pos,dim)=>{const i=Math.floor(dim/2),angle=pos/Math.pow(10000,2*i/dModel);return dim%2===0?Math.sin(angle):Math.cos(angle);};
    function renderPE(){const pair=+slider.value;ctx.clearRect(0,0,peCanvas.width,peCanvas.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,peCanvas.width,peCanvas.height);const left=50,right=25,top=20,bottom=35,mid=(peCanvas.height-bottom+top)/2;ctx.strokeStyle='#d9e1ec';ctx.beginPath();ctx.moveTo(left,mid);ctx.lineTo(peCanvas.width-right,mid);ctx.stroke();const draw=(dim,color)=>{ctx.beginPath();for(let p=0;p<=60;p++){const x=left+p/60*(peCanvas.width-left-right),y=mid-pe(p,dim)*(peCanvas.height-top-bottom)/2*.85;if(p===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke();};draw(pair*2,'#d93b3b');draw(pair*2+1,'#3169ff');ctx.fillStyle='#d93b3b';ctx.fillText(`sin dim ${pair*2}`,60,18);ctx.fillStyle='#3169ff';ctx.fillText(`cos dim ${pair*2+1}`,170,18);$('peDimVal').textContent=pair;const heat=$('peHeatmap');heat.style.gridTemplateColumns='42px repeat(16,29px)';let h='<div class="pe-cell"></div>'+Array.from({length:16},(_,d)=>`<div class="pe-cell" style="font-weight:800">d${d}</div>`).join('');for(let p=0;p<12;p++){h+=`<div class="pe-cell" style="font-weight:800">p${p}</div>`;for(let d=0;d<16;d++){const v=pe(p,d),alpha=.15+.75*Math.abs(v),bg=v>=0?`rgba(49,105,255,${alpha})`:`rgba(217,59,59,${alpha})`;h+=`<div class="pe-cell" title="${v.toFixed(3)}" style="background:${bg};color:${Math.abs(v)>.6?'#fff':'#1d2b44'}"></div>`;}}heat.innerHTML=h;}
    slider.addEventListener('input',renderPE);renderPE();
  }

  // ---------- Causal mask demo ----------
  if($('maskGrid')){
    const slider=$('maskRow'),n=6,grid=$('maskGrid');grid.style.gridTemplateColumns=`repeat(${n},minmax(42px,1fr))`;
    function renderMask(){const row=+slider.value;let h='';for(let r=0;r<n;r++)for(let c=0;c<n;c++){const allow=c<=r;h+=`<div class="mask-cell ${allow?'allow':'deny'} ${r===row?'current':''}">${allow?'0':'−∞'}</div>`;}grid.innerHTML=h;$('maskRowVal').textContent=`第 ${row+1} 个 token：可看 1~${row+1}`;}
    slider.addEventListener('input',renderMask);renderMask();
  }

  // ---------- Temperature demo ----------
  if($('tempBars')){
    const tokens=['继续','学习','模型','今天','结束'],logits=[2.4,1.8,1.0,.5,-.3],slider=$('tempSlider');
    function renderTemp(){const t=+slider.value/100,scaled=logits.map(x=>x/t),mx=Math.max(...scaled),ex=scaled.map(x=>Math.exp(x-mx)),z=ex.reduce((a,b)=>a+b,0),p=ex.map(x=>x/z);$('tempVal').textContent=t.toFixed(2);$('tempBars').innerHTML=p.map((v,i)=>`<div class="prob-row"><span>${tokens[i]}</span><div class="prob-bar"><i style="width:${v*100}%"></i></div><strong>${(v*100).toFixed(1)}%</strong></div>`).join('');}
    slider.addEventListener('input',renderTemp);renderTemp();
  }

  // ---------- LoRA calculator ----------
  if($('loraDim')){
    function fmt(n){return n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n);}
    function renderLora(){const n=+$('loraDim').value,r=+$('loraRank').value,full=n*n,lora=2*n*r,ratio=lora/full;$('loraDimVal').textContent=n;$('loraRankVal').textContent=r;$('loraFull').textContent=fmt(full);$('loraParams').textContent=fmt(lora);$('loraRatio').textContent=(ratio*100).toFixed(3)+'%';$('loraSaving').textContent=(full/lora).toFixed(1)+'×';}
    $('loraDim').addEventListener('input',renderLora);$('loraRank').addEventListener('input',renderLora);renderLora();
  }
})();
