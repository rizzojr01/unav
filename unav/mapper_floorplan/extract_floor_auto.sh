#!/bin/bash

# 自动化地板点云提取流程
# 支持两种方法: pointcloud (SLAM稀疏点云) 或 depth (DA²深度重投影)
#
# 用法: ./extract_floor_auto.sh [OPTIONS] [PLACE] [BUILDING] [FLOOR]
# 选项:
#   -m, --method METHOD  提取方法: pointcloud 或 depth (默认: depth)
#   -h, --help           显示帮助
#
# 示例:
#   ./extract_floor_auto.sh                              # 使用默认配置和depth方法
#   ./extract_floor_auto.sh -m pointcloud                # 使用pointcloud方法
#   ./extract_floor_auto.sh -m depth NYC LOH 9_floor     # 指定位置和方法

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# 默认方法
METHOD=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: ./extract_floor_auto.sh [OPTIONS] [PLACE] [BUILDING] [FLOOR]"
            echo ""
            echo "选项:"
            echo "  -m, --method METHOD  提取方法: pointcloud 或 depth (默认从config读取)"
            echo "  -h, --help           显示帮助"
            echo ""
            echo "示例:"
            echo "  ./extract_floor_auto.sh                              # 使用默认配置"
            echo "  ./extract_floor_auto.sh -m pointcloud                # 使用SLAM点云方法"
            echo "  ./extract_floor_auto.sh -m depth NYC LOH 9_floor     # 使用DA²深度方法"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 使用 Python 读取 YAML 配置
read_yaml() {
    python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
keys = '$1'.split('.')
value = config
for key in keys:
    if value is None:
        break
    value = value.get(key)
print(value if value is not None else '')
"
}

# 从配置文件读取默认值
DATA_ROOT=$(read_yaml "data_root")
DEFAULT_PLACE=$(read_yaml "default_place")
DEFAULT_BUILDING=$(read_yaml "default_building")
DEFAULT_FLOOR=$(read_yaml "default_floor")

# 目录名称
SLAM_DIR_NAME=$(read_yaml "dir_names.slam")
KEYFRAMES_DIR_NAME=$(read_yaml "dir_names.keyframes")
FLOOR_MAP_DIR_NAME=$(read_yaml "dir_names.floor_map")
MASK_DIR_NAME=$(read_yaml "dir_names.mask")
DOOR_MASK_DIR_NAME=$(read_yaml "dir_names.door_mask")
DEPTH_DIR_NAME=$(read_yaml "dir_names.depth")
KEYFRAME_FLOORS_DIR_NAME=$(read_yaml "dir_names.keyframe_floors")
KEYFRAME_VIS_DIR_NAME=$(read_yaml "dir_names.keyframe_vis")

# 文件名
DB_FILENAME=$(read_yaml "file_names.database")
OUTPUT_PLY_FILENAME=$(read_yaml "file_names.output_ply")
OUTPUT_PLY_DEPTH_FILENAME=$(read_yaml "file_names.output_ply_depth")

# SAM3 参数
FLOOR_PROMPT=$(read_yaml "sam3.floor_prompt")
DOOR_PROMPT=$(read_yaml "sam3.door_prompt")

# DA² 参数
DA2_CONDA_ENV=$(read_yaml "da2.conda_env")
DA2_PATH=$(read_yaml "da2.da2_path")
DEPTH_SCALE=$(read_yaml "da2.depth_scale")
DEPTH_PATTERN=$(read_yaml "da2.depth_pattern")

# 提取参数
if [ -z "$METHOD" ]; then
    METHOD=$(read_yaml "extraction.method")
fi
MASK_PATTERN=$(read_yaml "extraction.mask_pattern")
GRID_RESOLUTION=$(read_yaml "extraction.grid_resolution")
DEPTH_SUBSAMPLE=$(read_yaml "extraction.depth_subsample")

# 从命令行参数获取或使用默认值
PLACE="${1:-$DEFAULT_PLACE}"
BUILDING="${2:-$DEFAULT_BUILDING}"
FLOOR="${3:-$DEFAULT_FLOOR}"

# 构建路径
SLAM_DIR="${DATA_ROOT}/${PLACE}/${BUILDING}/${FLOOR}/${SLAM_DIR_NAME}"
FLOOR_MAP_DIR="${DATA_ROOT}/${PLACE}/${BUILDING}/${FLOOR}/${FLOOR_MAP_DIR_NAME}"
KEYFRAMES_DIR="${SLAM_DIR}/${KEYFRAMES_DIR_NAME}"
SQLITE_DB="${SLAM_DIR}/${DB_FILENAME}"

# 所有输出都放在 floor_map 目录下
MASK_DIR="${FLOOR_MAP_DIR}/${MASK_DIR_NAME}"
DEPTH_DIR="${FLOOR_MAP_DIR}/${DEPTH_DIR_NAME}"
KEYFRAME_FLOORS_DIR="${FLOOR_MAP_DIR}/${KEYFRAME_FLOORS_DIR_NAME}"
KEYFRAME_VIS_DIR="${FLOOR_MAP_DIR}/${KEYFRAME_VIS_DIR_NAME}"
OUTPUT_PLY="${FLOOR_MAP_DIR}/${OUTPUT_PLY_FILENAME}"
OUTPUT_PLY_DEPTH="${FLOOR_MAP_DIR}/${OUTPUT_PLY_DEPTH_FILENAME}"
DOOR_MASK_DIR="${FLOOR_MAP_DIR}/${DOOR_MASK_DIR_NAME}"
WEB_DIR="${FLOOR_MAP_DIR}/web"
WEB_POINTS="${WEB_DIR}/keyframe_points_world.json"
WEB_META="${WEB_DIR}/web_meta.json"

# 验证方法
if [ "$METHOD" != "pointcloud" ] && [ "$METHOD" != "depth" ]; then
    echo "✗ 错误: 无效的方法 '$METHOD'. 请使用 'pointcloud' 或 'depth'"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          自动化地板点云提取流程                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "位置:       $PLACE"
echo "建筑:       $BUILDING"
echo "楼层:       $FLOOR"
echo "方法:       $METHOD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查关键帧目录
if [ ! -d "$KEYFRAMES_DIR" ]; then
    echo "✗ 错误: 关键帧目录不存在: $KEYFRAMES_DIR"
    exit 1
fi

# 检查数据库文件
if [ ! -f "$SQLITE_DB" ]; then
    echo "✗ 错误: 数据库文件不存在: $SQLITE_DB"
    exit 1
fi

# 统计关键帧数量
KEYFRAME_COUNT=$(ls -1 "$KEYFRAMES_DIR"/image*.png 2>/dev/null | wc -l)
echo "✓ 找到 $KEYFRAME_COUNT 个关键帧"

# ═══════════════════════════════════════════════════════════════════════════════
# 步骤 1: 生成地板掩码 (两种方法都需要)
# ═══════════════════════════════════════════════════════════════════════════════

SKIP_MASK_GEN=false
if [ -d "$MASK_DIR" ]; then
    MASK_COUNT=$(ls -1 "$MASK_DIR"/*floor_mask.png 2>/dev/null | wc -l)
    if [ $MASK_COUNT -gt 0 ]; then
        echo "✓ 已存在 $MASK_COUNT 个掩码文件"
        read -p "是否重新生成掩码? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过掩码生成步骤"
            SKIP_MASK_GEN=true
        fi
    fi
fi

if [ "$SKIP_MASK_GEN" != "true" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "步骤 1: 使用 SAM3 生成地板掩码"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 使用 sam3 conda 环境
    source /home/unav/miniconda3/etc/profile.d/conda.sh
    conda activate sam3

    python "${SCRIPT_DIR}/generate_floor_masks_sam3.py" \
        "$KEYFRAMES_DIR" \
        "$MASK_DIR" \
        --prompt "$FLOOR_PROMPT"

    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ SAM3 掩码生成失败"
        exit 1
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 方法分支: pointcloud vs depth
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$METHOD" == "depth" ]; then
    # ═══════════════════════════════════════════════════════════════════════════
    # DEPTH 方法: DA² 深度估计 + 重投影
    # ═══════════════════════════════════════════════════════════════════════════

    # 步骤 2: 生成深度图
    SKIP_DEPTH_GEN=false
    if [ -d "$DEPTH_DIR" ]; then
        DEPTH_COUNT=$(ls -1 "$DEPTH_DIR"/*.npy 2>/dev/null | wc -l)
        if [ $DEPTH_COUNT -gt 0 ]; then
            echo ""
            echo "✓ 已存在 $DEPTH_COUNT 个深度图"
            read -p "是否重新生成深度图? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "跳过深度图生成步骤"
                SKIP_DEPTH_GEN=true
            fi
        fi
    fi

    if [ "$SKIP_DEPTH_GEN" != "true" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "步骤 2: 使用 DA² 生成深度图"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "注意: DA² 需要 Python 3.12 环境 ($DA2_CONDA_ENV)"
        echo ""

        # 调用 DA² wrapper 脚本
        bash "${SCRIPT_DIR}/run_da2_inference.sh" "$KEYFRAMES_DIR" "$DEPTH_DIR"

        if [ $? -ne 0 ]; then
            echo ""
            echo "✗ DA² 深度估计失败"
            exit 1
        fi
    fi

    # 步骤 3: 从深度图提取地板点云
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "步骤 3: 从深度图提取地板点云"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 使用 unav 环境 (包含 matplotlib)
    source /home/unav/miniconda3/etc/profile.d/conda.sh
    conda activate unav

    python "${SCRIPT_DIR}/extract_floor_from_depth.py" \
        "$SQLITE_DB" \
        "$DEPTH_DIR" \
        "$MASK_DIR" \
        "${FLOOR_MAP_DIR}" \
        --depth-pattern "$DEPTH_PATTERN" \
        --mask-pattern "$MASK_PATTERN" \
        --subsample "$DEPTH_SUBSAMPLE" \
        --scale "$DEPTH_SCALE"

    OUTPUT_FILE="$OUTPUT_PLY_DEPTH"

    # 步骤 4: 生成 web 预计算文件
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "步骤 4: 生成 Web 预计算文件"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    NEED_WEB=true
    if [ -f "$WEB_POINTS" ] && [ -f "$WEB_META" ]; then
        KEYFRAME_JSON_COUNT=$(ls -1 "${KEYFRAME_FLOORS_DIR}"/keyframe_*.json 2>/dev/null | wc -l | tr -d ' ')
        META_COUNT=$(python3 - <<PY
import json
try:
    with open("${WEB_META}","r") as f:
        print(json.load(f).get("source_keyframe_json", -1))
except Exception:
    print(-1)
PY
)
        if [ "$META_COUNT" = "$KEYFRAME_JSON_COUNT" ] && [ "$KEYFRAME_JSON_COUNT" != "0" ]; then
            echo "✓ Web 预计算文件已存在且数量匹配 (${KEYFRAME_JSON_COUNT})"
            NEED_WEB=false
        fi
    fi

    if [ "$NEED_WEB" = true ]; then
        python3 "${SCRIPT_DIR}/visualization/prepare_web_assets.py" "${FLOOR_MAP_DIR}" --max-points 200
    else
        echo "跳过 Web 预计算文件生成"
    fi

else
    # ═══════════════════════════════════════════════════════════════════════════
    # POINTCLOUD 方法: SLAM 稀疏点云过滤
    # ═══════════════════════════════════════════════════════════════════════════

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "步骤 2: 从SLAM点云提取地板点"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python3 "${SCRIPT_DIR}/extract_floor_points.py" \
        "$SQLITE_DB" \
        "$MASK_DIR" \
        "$OUTPUT_PLY" \
        --mask-pattern "$MASK_PATTERN" \
        --grid-resolution "$GRID_RESOLUTION" \
        --save-views \
        --save-map

    OUTPUT_FILE="$OUTPUT_PLY"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 地板点云提取失败"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ 全部完成！                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "使用方法: $METHOD"
echo ""
echo "输出文件:"
echo "  掩码目录:   $MASK_DIR"
if [ "$METHOD" == "depth" ]; then
    echo "  深度目录:   $DEPTH_DIR"
fi
echo "  点云文件:   $OUTPUT_FILE"
if [ -f "${OUTPUT_FILE%.ply}_views.png" ]; then
    echo "  三视图:     ${OUTPUT_FILE%.ply}_views.png"
fi
if [ -f "${OUTPUT_FILE%.ply}_map.png" ]; then
    echo "  网格地图:   ${OUTPUT_FILE%.ply}_map.png"
fi
echo ""

# 显示文件信息
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    POINT_COUNT=$(grep "element vertex" "$OUTPUT_FILE" 2>/dev/null | awk '{print $3}')
    echo "点云统计:"
    echo "  文件大小:   $FILE_SIZE"
    echo "  点数量:     $POINT_COUNT"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
