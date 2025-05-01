@echo off
cd /d "%~dp0"
echo Compiling LaTeX document...

:: 运行 XeLaTeX 编译 LaTeX 文件
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex

:: 获取当前日期（格式：YYYYMMDD）
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set today=%datetime:~0,8%

:: 目标文件名
set TARGET_NAME=%today%数据清洗与下游聚类的自动化协同优化及其影响机理研究.pdf

:: 复制并重命名 PDF 文件
copy /Y paper.pdf "%TARGET_NAME%"

echo Compilation complete. PDF saved as "%TARGET_NAME%"

:: 自动退出，不等待用户按键
exit
