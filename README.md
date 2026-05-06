# Astro_Agent

Astro_Agent 是面向天文文献调研的 RAG + 知识图谱工具箱。本项目当前重点是白矮星相关文献，把通用论文知识点改造成源中心图谱：围绕具体天体源整理“具有什么特征、用什么方法测量、测得哪些物理参数、证据来自哪些文献切片”。

## 当前知识图谱逻辑

- 图谱模式：`source_feature_literature_graph`
- 核心节点：`AstronomicalSource`、`SourceFeature`、`MeasurementMethod`、`PhysicalParameter`
- 核心关系：`具有特征`、`用方法测量`、`测得参数`
- 重点特征：磁性白矮星、短周期白矮星双星、激变变星/CV、大质量/超大质量白矮星、ELM 白矮星、脉动白矮星、金属污染/行星残骸白矮星、食白矮星双星、X 射线/高能白矮星系统、双简并/并合候选体

## 重新生成图谱

```powershell
cd C:\Users\Administrator\Desktop\rag
python prompt2graph_for_astronomy\build_white_dwarf_kg.py --run-name production_full
```

生成文件位于：

```text
prompt2graph_for_astronomy/output/white_dwarf_kg/production_full/
```

该目录包含完整图谱和源画像，体积较大，默认不提交到 Git。

## 启动前端

```powershell
cd C:\Users\Administrator\Desktop\rag\prompt2graph_for_astronomy
python launch_frontend_detached.py
```

默认访问：

```text
http://127.0.0.1:5011
```

前端已改为自定义的天体源特性调研面板，使用 `source_profiles.json` 的轻量源画像驱动图谱展示，避免直接加载完整 5 万条关系导致页面卡顿。
