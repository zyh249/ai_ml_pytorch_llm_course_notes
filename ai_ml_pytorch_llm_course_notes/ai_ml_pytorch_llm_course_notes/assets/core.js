(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);

  // ---------- Site navigation / utilities ----------
  const links = [...document.querySelectorAll('#toc a')];
  const sections = links
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  const progress = $('progress');
  const search = $('search');
  const mobileToggle = $('mobileToggle');

  const updateProgress = () => {
    if (!progress) return;
    const root = document.documentElement;
    const max = root.scrollHeight - root.clientHeight;
    progress.style.width = `${max > 0 ? (root.scrollTop / max) * 100 : 0}%`;
  };
  window.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = `#${entry.target.id}`;
        links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === id));
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0.01 });
    sections.forEach((section) => observer.observe(section));
  }

  document.querySelectorAll('.copy').forEach((button) => {
    button.addEventListener('click', async () => {
      const code = button.parentElement?.querySelector('code')?.innerText || '';
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = '已复制';
        button.classList.add('ok');
        window.setTimeout(() => {
          button.textContent = '复制';
          button.classList.remove('ok');
        }, 1200);
      } catch (_) {
        // file:// pages may not receive clipboard permission; provide a selection fallback.
        const range = document.createRange();
        const node = button.parentElement?.querySelector('code');
        if (node) {
          range.selectNodeContents(node);
          const selection = window.getSelection();
          selection?.removeAllRanges();
          selection?.addRange(range);
          button.textContent = '已选中';
          window.setTimeout(() => { button.textContent = '复制'; }, 1200);
        } else {
          button.textContent = '复制失败';
        }
      }
    });
  });

  $('expandAll')?.addEventListener('click', () => {
    document.querySelectorAll('.nav-group').forEach((detail) => { detail.open = true; });
  });
  $('collapseAll')?.addEventListener('click', () => {
    document.querySelectorAll('.nav-group').forEach((detail) => { detail.open = false; });
  });
  $('printBtn')?.addEventListener('click', () => window.print());

  search?.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    document.querySelectorAll('.nav-group').forEach((group) => {
      let hasMatch = false;
      group.querySelectorAll('a').forEach((a) => {
        const text = `${a.dataset.title || ''} ${a.dataset.module || ''} ${a.innerText}`.toLowerCase();
        const show = !query || text.includes(query);
        a.classList.toggle('hidden', !show);
        hasMatch ||= show;
      });
      group.classList.toggle('hidden', !hasMatch);
      if (query && hasMatch) group.open = true;
    });
  });

  mobileToggle?.addEventListener('click', () => document.body.classList.toggle('nav-open'));
  links.forEach((a) => a.addEventListener('click', () => document.body.classList.remove('nav-open')));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') document.body.classList.remove('nav-open');
  });

  // ---------- KNN interactive demo ----------
  const knnCanvas = $('knnCanvas');
  if (knnCanvas) {
    const ctx = knnCanvas.getContext('2d');
    const kInput = $('knnK');
    const points = [
      { name: 'A', x: 1.0, y: 2.0, label: '蓝色类' },
      { name: 'B', x: 2.0, y: 1.5, label: '蓝色类' },
      { name: 'C', x: 3.0, y: 3.5, label: '红色类' },
      { name: 'D', x: 3.6, y: 2.8, label: '红色类' },
      { name: 'E', x: 1.8, y: 3.2, label: '蓝色类' },
      { name: 'F', x: 4.2, y: 1.2, label: '红色类' }
    ];
    let query = { x: 2.8, y: 2.4 };
    const mapX = (value) => 80 + (value / 5) * (knnCanvas.width - 140);
    const mapY = (value) => knnCanvas.height - 60 - (value / 5) * (knnCanvas.height - 110);
    const inverseX = (pixel) => Math.max(0.3, Math.min(4.7, ((pixel - 80) / (knnCanvas.width - 140)) * 5));
    const inverseY = (pixel) => Math.max(0.3, Math.min(4.7, ((knnCanvas.height - 60 - pixel) / (knnCanvas.height - 110)) * 5));
    const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

    const drawStar = (x, y, radius) => {
      ctx.beginPath();
      for (let i = 0; i < 5; i += 1) {
        let angle = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
        ctx.lineTo(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius);
        angle += Math.PI / 5;
        ctx.lineTo(x + Math.cos(angle) * radius * 0.45, y + Math.sin(angle) * radius * 0.45);
      }
      ctx.closePath();
    };

    const renderKnn = () => {
      const k = Number(kInput?.value || 3);
      const ordered = points
        .map((point) => ({ ...point, distance: distance(point, query) }))
        .sort((a, b) => a.distance - b.distance);
      const neighbors = ordered.slice(0, k);
      const votes = neighbors.reduce((acc, point) => {
        acc[point.label] = (acc[point.label] || 0) + 1;
        return acc;
      }, {});
      const prediction = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];

      ctx.clearRect(0, 0, knnCanvas.width, knnCanvas.height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, knnCanvas.width, knnCanvas.height);
      ctx.strokeStyle = '#d9e1ee';
      ctx.lineWidth = 1;
      ctx.font = '12px sans-serif';
      for (let i = 0; i <= 5; i += 1) {
        const x = mapX(i);
        const y = mapY(i);
        ctx.beginPath(); ctx.moveTo(x, 25); ctx.lineTo(x, knnCanvas.height - 60); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(80, y); ctx.lineTo(knnCanvas.width - 60, y); ctx.stroke();
        if (i < 5) {
          ctx.fillStyle = '#667188';
          ctx.fillText(String(i), x + 2, knnCanvas.height - 40);
          ctx.fillText(String(i), 55, y + 4);
        }
      }

      neighbors.forEach((point) => {
        ctx.beginPath();
        ctx.strokeStyle = '#f0a500';
        ctx.lineWidth = 2;
        ctx.moveTo(mapX(query.x), mapY(query.y));
        ctx.lineTo(mapX(point.x), mapY(point.y));
        ctx.stroke();
      });

      ordered.forEach((point) => {
        ctx.beginPath();
        ctx.fillStyle = point.label === '蓝色类' ? '#3169ff' : '#d93b3b';
        ctx.arc(mapX(point.x), mapY(point.y), 9, 0, Math.PI * 2);
        ctx.fill();
        if (neighbors.some((neighbor) => neighbor.name === point.name)) {
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#f0a500';
          ctx.stroke();
        }
        ctx.fillStyle = '#202633';
        ctx.fillText(point.name, mapX(point.x) + 12, mapY(point.y) - 8);
      });

      ctx.fillStyle = '#111';
      drawStar(mapX(query.x), mapY(query.y), 13);
      ctx.fill();
      ctx.fillStyle = '#202633';
      ctx.fillText('Q', mapX(query.x) + 14, mapY(query.y) - 10);

      if ($('knnKVal')) $('knnKVal').textContent = String(k);
      if ($('knnPred')) $('knnPred').textContent = prediction;
      if ($('knnNearest')) $('knnNearest').textContent = neighbors.map((point) => point.name).join(' / ');
      if ($('knnVote')) $('knnVote').textContent = `蓝 ${votes['蓝色类'] || 0} : 红 ${votes['红色类'] || 0}`;
      if ($('knnQuery')) $('knnQuery').textContent = `(${query.x.toFixed(2)}, ${query.y.toFixed(2)})`;
    };

    kInput?.addEventListener('input', renderKnn);
    $('knnReset')?.addEventListener('click', () => { query = { x: 2.8, y: 2.4 }; renderKnn(); });
    knnCanvas.addEventListener('click', (event) => {
      const rect = knnCanvas.getBoundingClientRect();
      const px = (event.clientX - rect.left) * (knnCanvas.width / rect.width);
      const py = (event.clientY - rect.top) * (knnCanvas.height / rect.height);
      query = { x: inverseX(px), y: inverseY(py) };
      renderKnn();
    });
    renderKnn();
  }

  // ---------- Gradient descent interactive demo ----------
  const gdCanvas = $('gdCanvas');
  if (gdCanvas) {
    const ctx = gdCanvas.getContext('2d');
    let weight = -4;
    let steps = 0;
    let timer = null;
    const loss = (w) => (w - 2) ** 2 + 1;
    const gradient = (w) => 2 * (w - 2);
    const learningRate = () => Number($('gdLr')?.value || 20) / 100;
    const mapX = (w) => 70 + ((w + 5) / 10) * (gdCanvas.width - 130);
    const mapY = (value) => gdCanvas.height - 40 - (value / 40) * (gdCanvas.height - 80);

    const renderGd = () => {
      ctx.clearRect(0, 0, gdCanvas.width, gdCanvas.height);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, gdCanvas.width, gdCanvas.height);
      ctx.strokeStyle = '#d9e1ee'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(55, 20); ctx.lineTo(55, gdCanvas.height - 40); ctx.lineTo(gdCanvas.width - 35, gdCanvas.height - 40); ctx.stroke();
      ctx.beginPath();
      for (let i = 0; i <= 200; i += 1) {
        const w = -5 + (i / 200) * 10;
        if (i === 0) ctx.moveTo(mapX(w), mapY(loss(w)));
        else ctx.lineTo(mapX(w), mapY(loss(w)));
      }
      ctx.strokeStyle = '#3169ff'; ctx.lineWidth = 4; ctx.stroke();
      ctx.beginPath(); ctx.fillStyle = '#d93b3b'; ctx.arc(mapX(weight), mapY(loss(weight)), 9, 0, Math.PI * 2); ctx.fill();
      const nextWeight = weight - learningRate() * gradient(weight);
      ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = '#f0a500'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(mapX(weight), mapY(loss(weight))); ctx.lineTo(mapX(nextWeight), mapY(loss(nextWeight))); ctx.stroke(); ctx.restore();

      if ($('gdLrVal')) $('gdLrVal').textContent = learningRate().toFixed(2);
      if ($('gdW')) $('gdW').textContent = weight.toFixed(3);
      if ($('gdGrad')) $('gdGrad').textContent = gradient(weight).toFixed(3);
      if ($('gdLoss')) $('gdLoss').textContent = loss(weight).toFixed(3);
      if ($('gdStepCount')) $('gdStepCount').textContent = String(steps);
    };

    const singleStep = () => {
      weight -= learningRate() * gradient(weight);
      steps += 1;
      renderGd();
    };
    const stop = () => {
      if (timer) window.clearInterval(timer);
      timer = null;
      if ($('gdPlay')) $('gdPlay').textContent = '播放';
    };

    $('gdLr')?.addEventListener('input', renderGd);
    $('gdStep')?.addEventListener('click', singleStep);
    $('gdReset')?.addEventListener('click', () => { stop(); weight = -4; steps = 0; renderGd(); });
    $('gdPlay')?.addEventListener('click', () => {
      if (timer) { stop(); return; }
      $('gdPlay').textContent = '暂停';
      timer = window.setInterval(() => {
        singleStep();
        if (Math.abs(gradient(weight)) < 0.02 || steps > 45) stop();
      }, 450);
    });
    renderGd();
  }

  // ---------- Classification metric threshold demo ----------
  const metricThreshold = $('metricThreshold');
  if (metricThreshold) {
    const samples = [
      { y: 1, p: 0.95 }, { y: 1, p: 0.82 }, { y: 0, p: 0.77 }, { y: 1, p: 0.66 },
      { y: 0, p: 0.61 }, { y: 1, p: 0.58 }, { y: 0, p: 0.49 }, { y: 0, p: 0.31 },
      { y: 1, p: 0.27 }, { y: 0, p: 0.14 }
    ];
    const renderMetrics = () => {
      const threshold = Number(metricThreshold.value) / 100;
      let tp = 0; let fp = 0; let tn = 0; let fn = 0;
      samples.forEach((sample) => {
        const prediction = sample.p >= threshold ? 1 : 0;
        if (sample.y === 1 && prediction === 1) tp += 1;
        else if (sample.y === 0 && prediction === 1) fp += 1;
        else if (sample.y === 0 && prediction === 0) tn += 1;
        else fn += 1;
      });
      const accuracy = (tp + tn) / samples.length;
      const precision = tp + fp ? tp / (tp + fp) : 0;
      const recall = tp + fn ? tp / (tp + fn) : 0;
      const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
      const values = {
        metricThresholdVal: threshold.toFixed(2), mTP: tp, mFP: fp, mTN: tn, mFN: fn,
        mAcc: accuracy.toFixed(2), mPre: precision.toFixed(2), mRec: recall.toFixed(2), mF1: f1.toFixed(2)
      };
      Object.entries(values).forEach(([id, value]) => { if ($(id)) $(id).textContent = String(value); });
    };
    metricThreshold.addEventListener('input', renderMetrics);
    renderMetrics();
  }

  // ---------- Attention-context heatmap demo ----------
  const attentionSentence = $('attnSentenceSel');
  if (attentionSentence && $('attnBars') && $('attnGrid')) {
    const data = {
      fruit: {
        tokens: ['这', '家', '苹果', '很', '甜'],
        matrix: [
          [0.28, 0.28, 0.20, 0.12, 0.12], [0.16, 0.31, 0.23, 0.14, 0.16],
          [0.09, 0.12, 0.22, 0.23, 0.34], [0.08, 0.10, 0.25, 0.21, 0.36],
          [0.07, 0.08, 0.18, 0.26, 0.41]
        ]
      },
      phone: {
        tokens: ['苹果', '发布', '新', '手机', '了'],
        matrix: [
          [0.36, 0.22, 0.12, 0.20, 0.10], [0.18, 0.28, 0.16, 0.28, 0.10],
          [0.10, 0.18, 0.20, 0.42, 0.10], [0.14, 0.20, 0.18, 0.38, 0.10],
          [0.10, 0.18, 0.14, 0.30, 0.28]
        ]
      }
    };
    let focus = 2;
    const renderAttention = () => {
      const current = data[attentionSentence.value] || data.fruit;
      focus = Math.min(focus, current.tokens.length - 1);
      $('attnBars').innerHTML = current.matrix[focus].map((weight, index) =>
        `<div class="bar-row"><span>${current.tokens[index]}</span><div class="bar"><i style="width:${(weight * 100).toFixed(1)}%"></i></div><strong>${weight.toFixed(2)}</strong></div>`
      ).join('');

      const grid = $('attnGrid');
      grid.style.gridTemplateColumns = `repeat(${current.tokens.length + 1}, minmax(0, 1fr))`;
      let html = '<div class="cell head">Query\\Key</div>';
      html += current.tokens.map((token) => `<div class="cell head">${token}</div>`).join('');
      current.tokens.forEach((token, row) => {
        html += `<div class="cell head">${token}${row === focus ? ' ★' : ''}</div>`;
        current.tokens.forEach((_, column) => {
          const value = current.matrix[row][column];
          const alpha = 0.08 + value * 0.85;
          html += `<div class="cell" style="background:rgba(49,105,255,${alpha});color:${value > 0.30 ? '#fff' : '#1d2b44'}">${value.toFixed(2)}</div>`;
        });
      });
      grid.innerHTML = html;
      document.querySelectorAll('.attn-focus').forEach((button, index) => button.classList.toggle('active', index === focus));
    };

    attentionSentence.addEventListener('change', renderAttention);
    document.querySelectorAll('.attn-focus').forEach((button) => {
      button.addEventListener('click', () => { focus = Number(button.dataset.idx || 0); renderAttention(); });
    });
    renderAttention();
  }
})();
