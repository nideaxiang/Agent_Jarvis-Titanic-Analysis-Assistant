# 🤖 Jarvis-Titanic-Analysis-Assistant

**Jarvis** 是一个基于大语言模型（LLM）驱动的智能数据分析与预测系统。它不仅能以“贾维斯”的语气与你交谈，还能通过自动调用工具完成泰坦尼克号数据集的**统计分析**、**数据可视化**、**实时信息检索**以及**生存预测**。

---

## ✨ 核心特性

* **🎙 贾维斯人格设定**：沉浸式的对话体验，模仿《钢铁侠》中高科技助手的语气。
* **🔍 智能函数调用 (Function Calling)**：AI 能根据用户意图自动识别并调用以下工具：
* 以下的function纯手写：
* `analyze_data`: 自动识别数值或分类变量，输出统计摘要。
* `draw`: 动态生成数据图表（直方图或饼图）并返回给前端展示。
* `predict`: 基于内置的 **Random Forest（随机森林）** 模型进行实时生存预测。
* `web_search`: 集成 **Tavily Search**，获取最新的实时资讯或技术文档。


* **📊 机器学习集成**：启动时自动训练随机森林模型，提供可靠的预测能力。
* **🌐 Web 交互界面**：基于 Flask 开发，支持前后端分离的异步对话体验。

---

## 🛠 技术栈

* **Backend**: Python / Flask
* **AI SDK**: OpenAI API (Qwen-plus)
* **Data Science**: Pandas, NumPy, Scikit-learn, Scipy
* **Visualization**: Matplotlib
* **Search Engine**: Tavily API
* **Environment**: Dotenv (环境变量管理)

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/你的用户名/Jarvis-Titanic-Analysis.git
cd Jarvis-Titanic-Analysis

```

### 2. 安装依赖

```bash
pip install -r requirements.tx

```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并填入你的 API 密钥：

```env
OPENAI_API_KEY=你的OpenAI密钥
TAVILY_API_KEY=你的Tavily密钥
# 如果需要邮箱通知等其他功能可扩展

```

### 4. 准备数据

请确保项目目录下存在以下文件：
`10agentforanalysis/titanic_cleaned.csv`

### 5. 启动程序

```bash
python app.py

```

启动后，访问 `http://127.0.0.1:8080` 即可开始与贾维斯对话。

---

## 💡 使用示例

> **丹尼先生**: "帮我分析一下 Pclass 这一列的数据。"
> **Jarvis**: "正在为您接入数据核心... 先生。Pclass 的分析结果已出炉，数据显示大部分乘客集中在三等舱。"

> **丹尼先生**: "如果一个22岁的男性，在3等舱，他能活下来吗？"
> **Jarvis**: "正在模拟计算生存概率... 根据我的算法模型，结果为：Not Survived。抱歉，先生。"

---

## 📂 项目结构

```text
.
├── app.py                 # Flask 主程序及 AI 逻辑
├── .env                   # 环境变量配置文件
├── templates/             # HTML 模板文件夹
│   └── index.html         # 聊天界面
├── 10agentforanalysis/    # 数据文件夹
│   └── titanic_cleaned.csv # 处理后的泰坦尼克号数据
└── README.md              # 项目文档

```

---

## ⚖️ 开源协议

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源。

---

### 👨‍💻 作者

**丹尼**
*如有疑问或想进一步升级“贾维斯”的系统，欢迎提交 Issue 或 Pull Request！*
