# AI / 机器学习 / PyTorch / NLP / Transformer / 项目实战课程笔记

这是一个完全离线的单页课程网站，由上传课件和项目讲义持续整理。

## 本次新增：NewsCompass 投满分项目实战

- 仓库审计：说明静态 MkDocs 讲义与缺失的 TMFCode 源码边界。
- 业务与架构：新闻短文本 10 分类、项目分层与完整建模流水线。
- 数据模块：18 万训练集、1 万测试集、1 万验证集、10 类标签与 EDA。
- 模型模块：RandomForest、FastText、BERT、LLM API 的原理、代码、结果和选型。
- 工程模块：Flask/Streamlit 服务化、接口契约、监控与生产改造。
- 压缩模块：动态量化、知识蒸馏、模型剪枝及真实工程边界。
- 代码审查：修正数据路径覆盖、DataLoader、padding、LLM 解析、KL 散度与稀疏加速等问题。
- 新增 10 个示例脚本（55–64）和 2 个离线动态演示。

## 文件

- `index.html`：课程主页面
- `assets/`：样式、交互脚本和项目图例
- `examples/`：64 个配套 Python 示例
- `PROJECT_STUDY_GUIDE.md`：项目学习文档摘要
- `requirements.txt`：基础课程依赖
- `requirements-project.txt`：项目实战可选依赖

解压后直接打开 `index.html`。所有动态图例均使用本地 JavaScript，不依赖网络。

## DrawIO 串讲图补充
- `DRAWIO_EXTRACTED_NOTES.md`：从两个 draw.io 文件整理出的学习清单。
- `assets/drawio/`：提取出的 16 张串讲图和原始 `.drawio` 文件。

本次更新：DrawIO 串讲内容不再单独成章，已分别补充到投满分项目、TF-IDF、随机森林、FastText、部署、PyTorch、CNN、RNN、BERT 等对应章节。


## 本次补充：深度学习细节强化
- 激活函数：Sigmoid、Tanh、ReLU、LeakyReLU、GELU、Softmax 的作用、解决问题与局限。
- 初始化：全 0、随机、Xavier、Kaiming、Orthogonal 的适用场景。
- 损失函数：MSE、MAE、BCE、BCEWithLogits、CrossEntropy、类别不平衡加权。
- 优化器：SGD、Momentum、Nesterov、AdaGrad、RMSProp、Adam、AdamW 的核心解释、公式、优缺点和 PyTorch 写法。
- 学习率：固定 LR、StepLR、MultiStepLR、ExponentialLR、Cosine、ReduceLROnPlateau、Warmup 的选择建议。
- 正则化：L1/L2、weight_decay、Dropout、BatchNorm、LayerNorm、EarlyStopping、数据增强。
- 额外加入 CNN、RNN、Transformer 小点细化表格。
