# 三维相似性搜索平台 · 项目全景网站

航海学院「三维相似性搜索」课程项目的全景网站源码。用于向全组成员展示项目的**介绍、资料、执行计划、模块分工与在线认领**。

> 线上预览地址：https://cad-search-project.app.workbuddy.host/?sharecode=N5BBZLXLAW

## 项目是做什么的

基于 **BRepNet**（边界表征神经网络）对三维 CAD 模型（STEP 格式）做**相似性检索**：把零件三角面片图喂给 BRepNet 得到嵌入向量，再用 **FAISS** 做向量检索，输出「和当前零件最相似的 Top-K 零件」。评测指标为 **Recall@k / mAP**。

## 目录结构

```
.
├── index.html              # 单页应用（项目介绍/资料/执行计划/分工/认领）
├── docs-data.js            # 14 篇资料的全文数据（供站内阅读）
├── server.js               # 零依赖 Node 后端：静态托管 + /api/claims 认领接口
├── package.json            # npm start 启动脚本
├── assets/
│   ├── code/               # 核心代码：brepnet_retrieval.py、train.py、测试脚本
│   ├── docs/               # 项目文档（讲解/分工方案/操作手册/跑通报告）
│   ├── reports/            # 阶段报告（可行性报告 pptx / 实验报告 docx / 计划报告 docx）
│   └── results/            # 检索实验结果（per_query.csv / top_k.csv / summary.json）
└── data/claims.json        # 在线认领数据（运行时生成，默认不上传）
```

## 本地运行

依赖仅需 [Node.js](https://nodejs.org/)（版本 ≥ 14 即可，无第三方依赖）。

```bash
npm start
# 或
node server.js
```

浏览器打开 http://localhost:3000 即可查看完整网站。

> 只做静态预览（不启用在线认领）时，也可以直接双击 `index.html` 打开，但认领功能需要走 `server.js` 的后端接口。

## 功能说明

- **项目介绍**：背景、技术路线（BRepNet + FAISS）、数据来源（A 阶段）。
- **资料目录**：站内全文阅读 14 篇项目资料。
- **执行计划**：按阶段（A1–A9）拆解的步骤，带锚点可跳转。
- **模块分工**：10 个分工角色，每个角色在各阶段要做的具体步骤。
- **在线认领**：输入姓名认领分工（一人认领一个，自动去重）。

## 在线认领接口

- `GET /api/claims` → 返回当前认领数据 `{ "0": "张三", ... }`
- `POST /api/claims` → 认领/释放，body 形如 `{ "role": "3", "name": "张三", "action": "claim" }`
  - `role` 取 0–9（对应 10 个分工）；`action` 为 `claim`（认领）或 `release`（释放）。
  - 一个姓名只能认领一个分工，会自动清除该姓名在其他分工上的认领。

## 技术栈

- 前端：原生 HTML / CSS / JavaScript（单页，hash 路由，无框架）
- 后端：Node.js 原生 `http` 模块（零依赖）
