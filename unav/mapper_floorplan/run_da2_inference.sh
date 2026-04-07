#!/bin/bash
# DA² 深度估计 wrapper 脚本 (支持批处理避免OOM)
# 用法: ./run_da2_inference.sh <keyframes_dir> <output_dir> [batch_size]

KEYFRAMES_DIR="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-50}"  # 默认每批50张图片

if [ -z "$KEYFRAMES_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "用法: ./run_da2_inference.sh <keyframes_dir> <output_dir> [batch_size]"
    exit 1
fi

# DA² 路径
DA2_DIR="/home/unav/Desktop/unav/unav/tmp/pano_depth_methods/DA-2"
DA2_ENV="da2_py312"
TEMP_BATCH_DIR="/tmp/da2_batch_$$"

mkdir -p "$OUTPUT_DIR"

# 激活 DA² 环境
cd "$DA2_DIR"
source /home/unav/miniconda3/etc/profile.d/conda.sh
conda activate "$DA2_ENV"

# 获取所有图片
ALL_IMAGES=($(ls -1 "$KEYFRAMES_DIR"/image*.png 2>/dev/null | sort -V))
TOTAL_IMAGES=${#ALL_IMAGES[@]}

if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "✗ 错误: 没有找到图片"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DA² 深度估计 (批处理模式)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "输入目录: $KEYFRAMES_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "总图片数: $TOTAL_IMAGES"
echo "批大小:   $BATCH_SIZE"
echo ""

# 计算批次数
NUM_BATCHES=$(( (TOTAL_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "总批次数: $NUM_BATCHES"
echo ""

# 处理每个批次
for (( batch=0; batch<NUM_BATCHES; batch++ )); do
    START_IDX=$((batch * BATCH_SIZE))
    END_IDX=$((START_IDX + BATCH_SIZE))
    if [ $END_IDX -gt $TOTAL_IMAGES ]; then
        END_IDX=$TOTAL_IMAGES
    fi

    BATCH_NUM=$((batch + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "处理批次 $BATCH_NUM / $NUM_BATCHES (图片 $((START_IDX+1)) - $END_IDX)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 创建批次临时目录
    rm -rf "$TEMP_BATCH_DIR"
    mkdir -p "$TEMP_BATCH_DIR"

    # 复制本批次图片 (使用符号链接节省空间)
    for (( i=START_IDX; i<END_IDX; i++ )); do
        ln -s "${ALL_IMAGES[$i]}" "$TEMP_BATCH_DIR/"
    done

    # 创建配置
    TEMP_CONFIG="${TEMP_BATCH_DIR}/infer_config.json"
    cat > "$TEMP_CONFIG" << EOF
{
    "env": {
        "seed": 42,
        "verbose": true
    },
    "accelerator": {
        "report_to": ["tensorboard"],
        "mixed_precision": "fp16",
        "accumulation_nsteps": 4,
        "timeout": 36000
    },
    "inference": {
        "images": "${TEMP_BATCH_DIR}",
        "masks": "${DA2_DIR}/assets/masks",
        "min_pixels": 580000,
        "max_pixels": 620000
    },
    "spherevit": {
        "vit_w_esphere": {
            "input_dims": [1024, 1024, 1024, 1024],
            "hidden_dim": 512,
            "num_heads": 8,
            "expansion": 4,
            "num_layers_head": [2, 2, 2],
            "dropout": 0.0,
            "layer_scale": 0.0001,
            "out_dim": 64,
            "kernel_size": 3,
            "num_prompt_blocks": 1,
            "use_norm": false
        },
        "sphere": {
            "width": 1092,
            "height": 546,
            "hfov": 6.2832,
            "vfov": 3.1416
        }
    }
}
EOF

    # 运行 DA² 推理 (使用简化版脚本，只保存深度)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    accelerate launch \
        --config_file=configs/accelerate/0.yaml \
        --mixed_precision="fp16" \
        --main_process_port="12347" \
        "${SCRIPT_DIR}/da2_infer_depth_only.py" \
        --config_path="$TEMP_CONFIG"

    if [ $? -ne 0 ]; then
        echo "✗ 批次 $BATCH_NUM 处理失败"
        rm -rf "$TEMP_BATCH_DIR"
        exit 1
    fi

    # 移动输出到最终目录
    LATEST_OUTPUT=$(ls -td output/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_OUTPUT" ] && [ -d "${LATEST_OUTPUT}depth" ]; then
        cp "${LATEST_OUTPUT}depth/"*.npy "$OUTPUT_DIR/" 2>/dev/null
        BATCH_COUNT=$(ls -1 "${LATEST_OUTPUT}depth/"*.npy 2>/dev/null | wc -l)
        echo "✓ 批次 $BATCH_NUM 完成: $BATCH_COUNT 个深度图"
        # 清理DA²输出目录
        rm -rf "$LATEST_OUTPUT"
    fi

    # 清理临时目录
    rm -rf "$TEMP_BATCH_DIR"
    echo ""
done

# 最终统计
FINAL_COUNT=$(ls -1 "$OUTPUT_DIR"/*.npy 2>/dev/null | wc -l)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ DA² 深度估计完成"
echo "  总深度图数: $FINAL_COUNT / $TOTAL_IMAGES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
