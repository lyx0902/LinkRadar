# LinkRadar 本地交互报告

这是静态HTML/CSS/JS报告，真实数据由脚本从已验证结果导出。无需npm、CDN、数据库或云服务；直接用浏览器打开 `dist/index.html` 也能使用（通过data.js加载内嵌数据，不依赖file://的fetch权限）。

从项目根目录本地预览：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory web/linkradar/dist
```

浏览器打开 http://127.0.0.1:8765/ 。停止时在终端按Ctrl+C。

重新导出：

```powershell
.\.venv\Scripts\python.exe scripts/export_node_insights.py --reference-run results/20261023T054258_364875Z
```

配对复测完成后可添加 `--paired-run results/<实际配对结果目录>`。默认排名为全图精确介数，不是局部示意子图介数。页面内所有ID均为匿名原始编号。原始输入改变后必须重新计算及验证，导出器会检查图指纹和节点映射。

支持网络切换、原始ID筛选、介数/度数排序、全节点分页、CSV下载、局部网络点选/缩放/平移、失效曲线与性能诊断。此页面不向服务器提交任务；若未来需要实时计算，再加后端队列和API。

浏览器验收说明：应用内浏览器仅允许HTTP/HTTPS，不允许直接导航file协议；本轮通过127.0.0.1本地HTTP验证了页面、切换、搜索、排序、分页和节点点选。离线文件入口使用相邻data.js而非fetch，不需联网；直接文件打开未在该内置浏览器中实测。

## 发布到GitHub Pages

已提供[部署工作流](../../.github/workflows/pages.yml)。它只发布`web/linkradar/dist/`内的导出结果，不安装Python依赖或重新运行实验。

1. 在仓库Settings → Pages → Build and deployment中，将Source设为GitHub Actions。
2. 提交并推送工作流到`master`。此后推送网页目录或工作流本身的修改，会自动发布。
3. 也可在Actions中选择“Deploy LinkRadar to GitHub Pages” → Run workflow，分支选择`master`。
4. 工作流成功后，使用部署记录中的地址访问，核验网络切换、节点搜索、图片和CSV下载。

按当前仓库名，默认项目地址预计为 https://lyx0902.github.io/LinkRadar/ 。仅创建此工作流不代表已经上线；最终地址与状态以GitHub Pages部署记录为准。修改Python算法后，应先在本地计算、验证并重新导出网页，再提交新的展示结果。
