# 线程接手检查单（60秒）

## A. 基线核验（20秒）
1. `git rev-parse --short HEAD`
2. `git branch --show-current`
3. `git rev-list -n 1 v1.1.0`
4. 确认 `HEAD == v1.1.0` 对应提交。

## B. 工作区核验（10秒）
1. `git status --porcelain=v1`
2. 允许存在未跟踪产物: `build/`、`dist/`、`*.spec`。
3. 若发现代码文件脏改动，先记录再继续。

## C. 代码健康检查（10秒）
1. `.\.venv\Scripts\python.exe -m py_compile .\csv_wave_viewer.py`
2. 失败则先修复语法/导入错误，不进入需求开发。

## D. GUI 冒烟（20秒，桌面环境）
1. 启动: `.\.venv\Scripts\python.exe .\csv_wave_viewer.py`
2. 打开样例 CSV，勾选 1~3 个信号。
3. 校验交互:
   - 单击记录标记点（槽位轮转）
   - 左键拖拽框选时间
   - 滚轮缩放（主图/总览都能触发）
   - `×` 删除标记
4. 异常时记录: 操作步骤、预期、实际、日志路径 `logs/`。

## E. 交接输出模板
1. 本次改动: `<一句话>`
2. 基线位置: `<branch/tag/commit>`
3. 验证结果: `<py_compile + 冒烟结论>`
4. 回退点: `<tag 或 commit>`

