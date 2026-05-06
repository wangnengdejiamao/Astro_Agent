#!/bin/bash
# DESI光谱数据服务器访问脚本
# 用于在可以访问 /data1/DESI_DR1 的服务器上使用

# 配置
SERVER_PATH="/data1/DESI_DR1"
SPECTRA_PATH="$SERVER_PATH/coadd/spectra"
DUPLICATE_PATH="$SERVER_PATH/coadd/duplicate"
CSV_FILE="mws_gaia.csv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "DESI DR1 光谱数据服务器访问工具"
echo "========================================"
echo ""

# 检查路径是否存在
if [ ! -d "$SERVER_PATH" ]; then
    echo -e "${RED}错误: 无法访问服务器路径 $SERVER_PATH${NC}"
    echo "请确保您在正确的机器上运行此脚本"
    echo ""
    echo "如果您在本地机器上，请使用:"
    echo "  python download_desi_spectra.py --test"
    exit 1
fi

echo -e "${GREEN}成功连接到服务器!${NC}"
echo "服务器路径: $SERVER_PATH"
echo ""

# 显示统计数据
echo "正在统计光谱数据..."
if [ -d "$SPECTRA_PATH" ]; then
    SPECTRA_COUNT=$(ls $SPECTRA_PATH/*.fits 2>/dev/null | wc -l)
    echo "  主光谱文件: $SPECTRA_COUNT 个"
fi

if [ -d "$DUPLICATE_PATH" ]; then
    DUPLICATE_COUNT=$(ls $DUPLICATE_PATH/*.fits 2>/dev/null | wc -l)
    echo "  重复观测: $DUPLICATE_COUNT 个"
fi
echo ""

# 菜单
while true; do
    echo "请选择操作:"
    echo "  1. 检查mws_gaia.csv中有多少源有光谱"
    echo "  2. 复制指定数量的光谱到本地目录"
    echo "  3. 提取光谱数据为numpy格式"
    echo "  4. 查找特定TARGETID的光谱"
    echo "  5. 显示光谱目录结构"
    echo "  0. 退出"
    echo ""
    read -p "请输入选项 [0-5]: " choice
    
    case $choice in
        1)
            read -p "检查样本数 (默认1000): " sample_size
            sample_size=${sample_size:-1000}
            python read_local_spectra.py --check --check-sample $sample_size
            ;;
        2)
            read -p "输出目录 (默认./local_spectra): " output_dir
            output_dir=${output_dir:-./local_spectra}
            read -p "最大源数 (默认100): " max_sources
            max_sources=${max_sources:-100}
            python read_local_spectra.py --output $output_dir --max-sources $max_sources
            ;;
        3)
            read -p "输出目录 (默认./spectrum_data): " output_dir
            output_dir=${output_dir:-./spectrum_data}
            read -p "最大源数 (默认全部): " max_sources
            if [ -z "$max_sources" ]; then
                python read_local_spectra.py --extract --output $output_dir
            else
                python read_local_spectra.py --extract --output $output_dir --max-sources $max_sources
            fi
            ;;
        4)
            read -p "请输入TARGETID: " targetid
            echo "搜索光谱文件..."
            
            found=0
            if [ -f "$SPECTRA_PATH/$targetid.fits" ]; then
                echo -e "${GREEN}找到主光谱: $SPECTRA_PATH/$targetid.fits${NC}"
                ls -lh "$SPECTRA_PATH/$targetid.fits"
                found=1
            fi
            
            if [ -f "$DUPLICATE_PATH/$targetid.fits" ]; then
                echo -e "${GREEN}找到重复观测: $DUPLICATE_PATH/$targetid.fits${NC}"
                ls -lh "$DUPLICATE_PATH/$targetid.fits"
                found=1
            fi
            
            # 检查多次观测
            for i in {2..10}; do
                if [ -f "$DUPLICATE_PATH/${targetid}_${i}.fits" ]; then
                    echo -e "${GREEN}找到第$((i+1))次观测: $DUPLICATE_PATH/${targetid}_${i}.fits${NC}"
                    ls -lh "$DUPLICATE_PATH/${targetid}_${i}.fits"
                    found=1
                fi
            done
            
            if [ $found -eq 0 ]; then
                echo -e "${RED}未找到TARGETID $targetid 的光谱文件${NC}"
            fi
            ;;
        5)
            echo ""
            echo "服务器目录结构:"
            echo "$SERVER_PATH/"
            find "$SERVER_PATH" -maxdepth 2 -type d | head -20 | sed 's|[^/]*/|  |g'
            ;;
        0)
            echo "退出"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            ;;
    esac
    
    echo ""
    read -p "按回车键继续..."
    echo ""
done
