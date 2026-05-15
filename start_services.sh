#!/bin/bash
# Astro_Agent 服务启动脚本
# 启动: Agent 网页版 (8765) + 图谱可视化 (6777)
# 可选: 图谱统一接口 (5010)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 加载环境变量
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[INFO] 加载环境变量: $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "[WARN] 未找到 .env 文件"
fi

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"

PID_FILE="$PROJECT_ROOT/.service_pids"
PYTHON="python3"

# 检查端口占用
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[ERROR] 端口 $port 已被占用，无法启动 $name"
        exit 1
    fi
}

# 启动服务
start_service() {
    local name=$1
    local dir=$2
    local cmd=$3
    local port=$4
    local logfile=$5

    echo "[START] 启动 $name (端口 $port)..."
    cd "$dir"
    nohup $cmd > "$logfile" 2>&1 &
    local pid=$!
    cd "$PROJECT_ROOT"
    echo "$name:$pid:$port" >> "$PID_FILE"
    echo "[OK] $name PID=$pid, 日志=$logfile"
}

# 清理旧 PID 文件
if [ -f "$PID_FILE" ]; then
    echo "[INFO] 发现旧 PID 文件，先执行停止..."
    bash "$PROJECT_ROOT/stop_services.sh" >/dev/null 2>&1 || true
fi

echo "=========================================="
echo "  Astro_Agent 服务启动"
echo "=========================================="

# 1. Agent 网页版
# ------------------------------------------------------------------
check_port 8765 "Agent 网页版"
start_service \
    "Agent网页版" \
    "$PROJECT_ROOT/Astro_Agent" \
    "$PYTHON -m uvicorn analysis_agent.server:app --host 0.0.0.0 --port 8765 --reload" \
    8765 \
    "$PROJECT_ROOT/logs/agent_server.log"

# 2. 图谱可视化
# ------------------------------------------------------------------
check_port 6777 "图谱可视化"
start_service \
    "图谱可视化" \
    "$PROJECT_ROOT/graph_for_astronomy" \
    "$PYTHON vis_graph.py" \
    6777 \
    "$PROJECT_ROOT/logs/vis_graph.log"

# 3. 【可选】图谱统一接口 (如需 Neo4j 查询服务，取消下面注释)
# ------------------------------------------------------------------
# check_port 5010 "图谱统一接口"
# start_service \
#     "图谱统一接口" \
#     "$PROJECT_ROOT/graph_for_astronomy" \
#     "$PYTHON graph_api.py" \
#     5010 \
#     "$PROJECT_ROOT/logs/graph_api.log"

echo "=========================================="
echo "  所有服务已启动"
echo "=========================================="
echo ""
echo "🌐 访问地址:"
echo "   Agent 网页版:  http://localhost:8765"
echo "   图谱可视化:    http://localhost:6777"
echo ""
echo "📄 日志目录: $PROJECT_ROOT/logs/"
echo "🔧 停止命令: bash stop_services.sh"
echo ""
