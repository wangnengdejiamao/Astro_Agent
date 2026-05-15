#!/bin/bash
# Astro_Agent 服务停止脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_ROOT/.service_pids"

echo "=========================================="
echo "  Astro_Agent 服务停止"
echo "=========================================="

if [ ! -f "$PID_FILE" ]; then
    echo "[WARN] 未找到 PID 文件 ($PID_FILE)，尝试按端口查找进程..."

    # 按端口强制清理
    for port in 8765 6777 5010; do
        pid=$(lsof -t -Pi :"$port" -sTCP:LISTEN 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "[STOP] 杀死端口 $port 的进程 PID=$pid"
            kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        fi
    done

    echo "[OK] 清理完成"
    exit 0
fi

# 按 PID 文件优雅停止
while IFS=: read -r name pid port; do
    if [ -z "$pid" ]; then
        continue
    fi

    if kill -0 "$pid" >/dev/null 2>&1; then
        echo "[STOP] 停止 $name (PID=$pid, 端口=$port)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "[WARN] $name 未响应，强制终止..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    else
        echo "[SKIP] $name (PID=$pid) 已不在运行"
    fi
done < "$PID_FILE"

# 额外清理可能残留的 uvicorn/python 进程
echo "[INFO] 检查残留进程..."
for port in 8765 6777 5010; do
    pid=$(lsof -t -Pi :"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[STOP] 清理残留端口 $port PID=$pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

rm -f "$PID_FILE"
echo "=========================================="
echo "  所有服务已停止"
echo "=========================================="
