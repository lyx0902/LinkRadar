# LinkRadar 本地交互报告

这是静态HTML/CSS/JS报告，真实数据由脚本从已验证结果导出。无需npm、CDN、数据库或云服务；直接用浏览器打开 `dist/index.html` 也能使用（通过data.js加载内嵌数据，不依赖file://的fetch权限）。

从项目根目录本地预览：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory web/linkradar/dist
```

浏览器打开 http://127.0.0.1:8765/ 。停止时在终端按Ctrl+C。

重新导出：

```powershell
.\.venv\Scripts\python.exe scripts/export_node_insights.py --reference-run results/20261023T054258_364875Z --paired-run results/20261023T061912_984947Z_facebook_paired --paired-run results/20261023T090440_116909Z_ca_grqc_paired
```

每张图可各指定一次 `--paired-run results/<实际配对结果目录>`，该参数可以重复。任务03会按当前网络匹配同图指纹的复测表格与图，不混用Facebook和ca-GrQc。默认排名为全图精确介数，不是局部示意子图介数。页面内所有ID均为匿名原始编号。原始输入改变后必须重新计算及验证，导出器会检查图指纹和节点映射。

支持网络切换、原始ID筛选、介数/度数排序、全节点分页、CSV下载、局部网络点选/缩放/平移、失效曲线与性能诊断。此页面不向服务器提交任务；若未来需要实时计算，再加后端队列和API。

浏览器验收说明：应用内浏览器仅允许HTTP/HTTPS，不允许直接导航file协议；本轮通过127.0.0.1本地HTTP验证了页面、切换、搜索、排序、分页和节点点选。离线文件入口使用相邻data.js而非fetch，不需联网；直接文件打开未在该内置浏览器中实测。

## 发布到GitHub Pages

已提供[部署工作流](../../.github/workflows/pages.yml)。它只发布`web/linkradar/dist/`内的导出结果，不安装Python依赖或重新运行实验。

1. 在仓库Settings → Pages → Build and deployment中，将Source设为GitHub Actions。
2. 提交并推送工作流到`master`。此后推送网页目录或工作流本身的修改，会自动发布。
3. 也可在Actions中选择“Deploy LinkRadar to GitHub Pages” → Run workflow，分支选择`master`。
4. 工作流成功后，使用部署记录中的地址访问，核验网络切换、节点搜索、图片和CSV下载。

按当前仓库名，默认项目地址预计为 https://lyx0902.github.io/LinkRadar/ 。仅创建此工作流不代表已经上线；最终地址与状态以GitHub Pages部署记录为准。修改Python算法后，应先在本地计算、验证并重新导出网页，再提交新的展示结果。

## 原理与术语页

导航“04 原理与术语”提供介数与归一化、Brandes路径计数及依赖递推、串行/静态/动态分批、LCC与固定排名失效、加速比/效率及容差公式。五节点教学图支持切换端点、点选观察节点，查看最短路份额与完整手算结果；另有固定S为起点的六步BFS/反向累计演示，逐步显示d、σ、前驱与δ；术语支持搜索。真实图公式示例随网络选择切换。

公式使用浏览器原生MathML，交互代码为本地`dist/theory.js`，无需在线公式库或CDN。教学图只计算五个节点，不重新运行真实网络算法；现有导出数据与正式实验结果保持原样。

## 失效页与配对复测展示

任务02只保留当前网络的失效曲线和铺满内容区的“度数与中转作用”散点图，切换网络后同步替换。双网络总览导出图仍属于实验附件，不再重复占用网页面板。

配对实验使用同一完整图、B=128、固定4个可用逻辑CPU；一次串行热身后，serial、dynamic(p=1)、dynamic(p=4)按3×3轮换顺序各测三次。可分别启动：

```powershell
.\.venv\Scripts\python.exe -B scripts/paired_baseline.py --dataset facebook --budget 1200
.\.venv\Scripts\python.exe -B scripts/paired_baseline.py --dataset ca_grqc --budget 600
```

这是较长的真实计算，不是打开网页的前置步骤。只导入完整成功且与当前图指纹匹配的运行；柱形显示中位数、散点显示重复实测、误差线显示min–max。单工作进程基线检查仅为启发式诊断，未通过时不把四进程比值解释为可靠多核加速。
