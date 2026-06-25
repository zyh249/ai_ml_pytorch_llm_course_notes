
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);

  const stages = {
    business: ['业务定义', '把新闻短文本分到 10 个频道。先明确标签边界、拒识规则、延迟与准确率目标。'],
    data: ['数据与质检', '读取 train/dev/test 与 class.txt，检查编码、空行、重复、标签越界、类别比例和文本长度。'],
    feature: ['预处理与特征', '随机森林使用分词与 TF-IDF；FastText 构造字符/词级监督格式；BERT 使用 tokenizer、padding、truncation 与 mask。'],
    train: ['训练与选择', '先做低成本基线，再比较 FastText、BERT 与 LLM。所有方案必须使用相同切分、指标和硬件记录。'],
    deploy: ['服务化', '模型进程启动时加载一次，提供版本化预测接口、健康检查、日志、超时、限流和监控。'],
    compress: ['压缩与迭代', '量化、蒸馏和剪枝都要重新测精度、真实文件大小、CPU/GPU 延迟、吞吐和峰值内存。'],
  };
  const stageDetail = $('projectStageDetail');
  document.querySelectorAll('.project-stage').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.project-stage').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      const [title, text] = stages[button.dataset.stage] || ['', ''];
      if (stageDetail) stageDetail.innerHTML = `<h4>${title}</h4><p>${text}</p>`;
    });
  });
  document.querySelector('.project-stage')?.click();

  const modelData = {
    rf: { name: 'TF-IDF + RandomForest', score: '82.48%', latency: '低到中', cost: '低', use: '快速基线、流程验证', note: '高维稀疏文本中也应同时尝试 LinearSVC / LogisticRegression。' },
    ft: { name: 'FastText 字符级自动调参', score: '91.65%', latency: '很低', cost: '低', use: 'CPU 在线服务、边缘部署', note: '文档快照中是性价比最突出的方案。' },
    bert: { name: 'BERT 分类器', score: '约 93.64%', latency: '中到高', cost: '中到高', use: '追求较高闭集分类质量', note: '需要固定数据切分并记录 tokenizer、checkpoint 与硬件。' },
    llm: { name: 'LLM API 提示分类', score: '约 69.08%', latency: '很高', cost: '按调用计费', use: '冷启动、开放集、弱标注、兜底', note: '对固定 10 类任务不宜默认替代监督模型。' },
    distill: { name: '蒸馏 BiLSTM', score: '约 89.89%~91.25%', latency: '低到中', cost: '训练中等', use: '压缩 BERT、CPU 服务', note: '文档有两组不一致结果，复现实验时需统一 checkpoint 与划分。' },
  };
  const modelButtons = document.querySelectorAll('[data-project-model]');
  const renderModel = (key) => {
    const data = modelData[key];
    if (!data) return;
    modelButtons.forEach((button) => button.classList.toggle('active', button.dataset.projectModel === key));
    if ($('projectModelName')) $('projectModelName').textContent = data.name;
    if ($('projectModelScore')) $('projectModelScore').textContent = data.score;
    if ($('projectModelLatency')) $('projectModelLatency').textContent = data.latency;
    if ($('projectModelCost')) $('projectModelCost').textContent = data.cost;
    if ($('projectModelUse')) $('projectModelUse').textContent = data.use;
    if ($('projectModelNote')) $('projectModelNote').textContent = data.note;
  };
  modelButtons.forEach((button) => button.addEventListener('click', () => renderModel(button.dataset.projectModel)));
  renderModel('ft');

  document.querySelectorAll('.project-score-row').forEach((row) => {
    const score = Number(row.dataset.score || 0);
    const fill = row.querySelector('.fill');
    if (fill) requestAnimationFrame(() => { fill.style.width = `${Math.max(0, Math.min(100, score))}%`; });
  });

  const quantBits = $('projectQuantBits');
  const renderQuant = () => {
    if (!quantBits) return;
    const bits = Number(quantBits.value);
    const ratio = bits / 32;
    const estimated = 390 * ratio;
    if ($('projectQuantBitsVal')) $('projectQuantBitsVal').textContent = `${bits}-bit`;
    if ($('projectQuantRatio')) $('projectQuantRatio').textContent = `${(ratio * 100).toFixed(1)}%`;
    if ($('projectQuantEstimate')) $('projectQuantEstimate').textContent = `${estimated.toFixed(1)} MB`;
    const meter = $('projectQuantMeterFill');
    if (meter) meter.style.width = `${Math.max(4, ratio * 100)}%`;
  };
  quantBits?.addEventListener('input', renderQuant);
  renderQuant();

  const logits = [4, 2, 0];
  const softmax = (values, temperature) => {
    const scaled = values.map((value) => value / temperature);
    const maxValue = Math.max(...scaled);
    const exp = scaled.map((value) => Math.exp(value - maxValue));
    const sum = exp.reduce((acc, value) => acc + value, 0);
    return exp.map((value) => value / sum);
  };
  const tempInput = $('projectDistillTemp');
  const renderTemp = () => {
    if (!tempInput) return;
    const temperature = Number(tempInput.value) / 10;
    const probabilities = softmax(logits, temperature);
    if ($('projectDistillTempVal')) $('projectDistillTempVal').textContent = temperature.toFixed(1);
    probabilities.forEach((probability, index) => {
      const fill = $(`projectTempFill${index}`);
      const value = $(`projectTempValue${index}`);
      if (fill) fill.style.width = `${probability * 100}%`;
      if (value) value.textContent = probability.toFixed(3);
    });
  };
  tempInput?.addEventListener('input', renderTemp);
  renderTemp();
})();
