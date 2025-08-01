#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
framework_balanced2.py ── AutoML 框架示意图（Dataset 居中 + Dynamic Tuning 底部）
"""

import os
from pathlib import Path
from graphviz import Digraph

# ---------- 0. 字体路径 ----------
EN_PATH = r"C:\Windows\Fonts\times.ttf"     # Times New Roman
CN_PATH = r"C:\Windows\Fonts\simsun.ttc"    # SimSun
os.environ["GDFONTPATH"] = r"C:\Windows\Fonts"

# ---------- 1. 新建图 ----------
g = Digraph(name="framework_balanced2", format="svg")
g.attr(fontsize="11", fontname=EN_PATH)
g.node_attr.update(fontname=EN_PATH)
g.edge_attr.update(fontname=EN_PATH)

# ========== 2. Feature Enhancement (Offline) ==========
with g.subgraph(name="cluster_feat") as c:
    c.attr(label="特征增强（离线）", color="#007acc",
           style="rounded,filled", fillcolor="#e6f2ff",
           fontname=CN_PATH)

    c.node("hist",    "历史\n⟨q,a,f⟩",        fontname=CN_PATH,
           shape="folder", style="filled", fillcolor="#ffffff")
    c.node("extract", "特征\n提取 z",        fontname=CN_PATH,
           shape="box",    style="filled", fillcolor="#c6e2ff")
    c.node("ranker",  "LightGBM 排序器",     fontname=CN_PATH,
           shape="box",    style="filled", fillcolor="#c6e2ff")
    c.node("qrf",     "分位随机森林",        fontname=CN_PATH,
           shape="box",    style="filled", fillcolor="#c6e2ff")
    c.node("phi",     "预测器 Φ",           fontname=CN_PATH,
           shape="component", style="filled", fillcolor="#ffffff")

    c.edge("hist", "extract")
    c.edge("extract", "ranker")
    c.edge("ranker", "qrf", label="残差", fontsize="9", fontname=CN_PATH)
    c.edge("ranker", "phi")
    c.edge("qrf", "phi")

# ========== 3. Confidence‑UCB Search ==========
with g.subgraph(name="cluster_search") as s:
    s.attr(label="置信-UCB 搜索", color="#3ab54a",
           style="rounded,filled", fillcolor="#e8ffe8",
           fontname=CN_PATH)

    # 为右移留空位
    g.node("right_dummy", "", width="0.01", style="invis")

    s.node("space",  "完整空间 Π",     fontname=CN_PATH,
           shape="cylinder", style="filled", fillcolor="#ffffff")
    s.node("ucb",    "UCB 选择\nπ⋆",   fontname=CN_PATH,
           shape="box",      style="filled", fillcolor="#cdf1d1")
    s.node("prune",  "剪枝\n(UCB < best-ε)", fontname=CN_PATH,
           shape="box",      style="filled", fillcolor="#cdf1d1")
    s.node("eval",   "评估\nf(π⋆)",    fontname=CN_PATH,
           shape="box",      style="filled", fillcolor="#cdf1d1")
    s.node("expand", "扩展 &\nTop‑k",  fontname=CN_PATH,
           shape="box",      style="filled", fillcolor="#cdf1d1")
    s.node("cand",   "候选集合 S",     fontname=CN_PATH,
           shape="folder",   style="filled", fillcolor="#ffffff")

    with s.subgraph() as same:
        same.attr(rank="same")
        same.node("space")
        same.node("right_dummy")
        g.edge("right_dummy", "space", style="invis")

    # 绿色模块内部连线
    g.edge("space", "ucb")
    g.edge("ucb", "prune", label="fail")
    g.edge("ucb", "eval",  label="pass")
    g.edge("eval", "expand")
    g.edge("expand", "space", style="dotted")
    g.edge("eval", "cand")

# ========== 4. Dataset D⋆（中心） ==========
g.node("data", "数据集 D⋆", fontname=CN_PATH,
       shape="cylinder", style="filled", fillcolor="#ffffff")

# Dataset 与 UCB 同层对齐
with g.subgraph() as center:
    center.attr(rank="same")
    center.node("data")
    center.node("ucb")

# ========== 5. Dynamic Tuning (Online) ==========
with g.subgraph(name="cluster_tune") as t:
    t.attr(label="动态调优（在线）", color="#d87a00",
           style="rounded,filled", fillcolor="#fff2e6",
           fontname=CN_PATH)

    t.node("monitor", "监控\n{Sil, DB}", fontname=CN_PATH,
           shape="box",     style="filled", fillcolor="#ffe5cc")
    t.node("trigger", "触发？",           fontname=CN_PATH,
           shape="diamond", style="filled", fillcolor="#ffe5cc")
    t.node("local",   "局部 Θ",          fontname=CN_PATH,
           shape="box",     style="filled", fillcolor="#ffe5cc")
    t.node("probe",   "探测 10%",        fontname=CN_PATH,
           shape="box",     style="filled", fillcolor="#ffe5cc")
    t.node("update",  "更新 π̂",         fontname=CN_PATH,
           shape="box",     style="filled", fillcolor="#ffe5cc")

    t.edge("monitor", "trigger")
    t.edge("trigger", "local", label="yes", fontsize="9", fontname=CN_PATH)
    t.edge("local", "probe")
    t.edge("probe", "update")
    t.edge("update", "monitor", style="dashed")

# **关键：让 Dynamic Tuning 位于最底部**
# 建立一条不可见边：cand (绿色最后) → monitor (橙色起点)
g.edge("cand", "monitor", style="invis")

# ========== 6. 跨模块连线 ==========
g.edge("data", "extract", style="dashed", label="q")                   # Dataset → Feature
g.edge("data", "ucb", style="dashed", label="q,a_mask")                # Dataset → UCB (垂直虚线)
g.edge("phi",  "ucb", label="Φ", color="#007acc", constraint="false", fontname=CN_PATH)
g.edge("cand", "monitor", label="π̂", color="#3ab54a", fontname=CN_PATH)  # 绿色 → 橙色 (功能连线)
g.edge("update","ucb", style="dashed", label="ΔΦ", color="#d87a00", fontname=CN_PATH)

# ========== 7. 输出 ==========
out_dir  = Path(r"D:\algorithm paper\AutoMLClustering_full\task_progress\figures\4graph")
out_path = out_dir / "method_framework"
g.render(str(out_path), cleanup=True)
print("✅ SVG 已生成：", str(out_path) + ".svg")
