@echo off
REM 删除错误文件
del "tatus \357\200\276 git_status.txt"
del update_status.txt

REM 清理 Git 状态
git reset HEAD *

REM 添加需要的文件和目录
git add "预处理测试/"
git add tempCodeRunnerFile.bat

REM 提交更改
git commit -m "清理并更新文件结构"

REM 推送到远程
git push origin main

REM 验证最终状态
git status > final_status.txt
type final_status.txt