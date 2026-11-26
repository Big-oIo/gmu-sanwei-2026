# 一键推送脚本：把本目录源码推送到 GitHub 仓库 gmu-sanwei-2026
# 使用方法：在 PowerShell 中运行  .\push.ps1
# 首次推送会弹出浏览器要求登录 GitHub（账号 Big-oIo），登录一次即可。

$ErrorActionPreference = 'Stop'

$git = "C:\Users\31356\.workbuddy\binaries\PortableGit\versions\1.2.0\cmd\git.exe"
$env:PATH = "C:\Users\31356\.workbuddy\binaries\PortableGit\versions\1.2.0\cmd;C:\Users\31356\.workbuddy\binaries\PortableGit\versions\1.2.0\mingw64\bin;" + $env:PATH

$repoUrl = "https://github.com/Big-oIo/gmu-sanwei-2026.git"

Write-Host "== 1/4 检查远程仓库 ==" -ForegroundColor Cyan
$remote = & $git remote
if ($remote -notcontains "origin") {
    & $git remote add origin $repoUrl
    Write-Host "已添加远程 origin"
} else {
    & $git remote set-url origin $repoUrl
    Write-Host "已更新远程 origin"
}

Write-Host "== 2/4 配置提交者信息（如缺失） ==" -ForegroundColor Cyan
$name = & $git config user.name
$email = & $git config user.email
if (-not $name) {
    & $git config user.name "Big-oIo"
}
if (-not $email) {
    & $git config user.email "315270633+Big-oIo@users.noreply.github.com"
}

Write-Host "== 3/4 暂存并提交 ==" -ForegroundColor Cyan
& $git add -A
$committed = & $git rev-parse --verify HEAD 2>$null
if (-not $committed) {
    & $git commit -m "init: 三维相似性搜索项目全景网站源码"
} else {
    & $git commit -m "update: 项目网站源码" --allow-empty
}

Write-Host "== 4/4 推送到 GitHub ==" -ForegroundColor Cyan
Write-Host "首次推送会弹出浏览器登录 GitHub（账号 Big-oIo），请按提示完成授权。" -ForegroundColor Yellow
& $git push -u origin main

Write-Host ""
Write-Host "完成！仓库地址：https://github.com/Big-oIo/gmu-sanwei-2026" -ForegroundColor Green
