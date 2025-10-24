#!/bin/bash

# RunPod Quick Launcher for LTX Video Training
# Provides interactive menu for common tasks

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

WORKSPACE="/workspace/LTX_video_training"

# Check if we're in the right directory
if [ ! -d "$WORKSPACE" ]; then
    echo -e "${RED}Error: $WORKSPACE not found!${NC}"
    echo "Please run runpod_setup.sh first."
    exit 1
fi

cd $WORKSPACE

# Function to display header
show_header() {
    clear
    echo -e "${CYAN}"
    echo "=============================================="
    echo "  LTX Video Trajectory Training - RunPod"
    echo "=============================================="
    echo -e "${NC}"
    echo ""
}

# Function to check GPU status
check_gpu() {
    echo -e "${BLUE}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx name temp util mem_used mem_total; do
        echo "  GPU $idx: $name"
        echo "    Temperature: ${temp}°C"
        echo "    Utilization: ${util}%"
        echo "    Memory: ${mem_used}MB / ${mem_total}MB"
    done
    echo ""
}

# Function to check disk space
check_disk() {
    echo -e "${BLUE}Disk Space:${NC}"
    df -h /workspace | tail -1 | awk '{print "  Used: "$3" / "$2" ("$5")"}'
    echo ""
}

# Function to count dataset
check_dataset() {
    if [ -f "./dataset/train.csv" ]; then
        count=$(wc -l < ./dataset/train.csv)
        count=$((count - 1))  # Subtract header
        echo -e "${BLUE}Dataset:${NC}"
        echo "  Training samples: $count"
        echo ""
    fi
}

# Main menu
main_menu() {
    while true; do
        show_header
        check_gpu
        check_disk
        check_dataset

        echo -e "${GREEN}What would you like to do?${NC}"
        echo ""
        echo "  ${YELLOW}1)${NC} Prepare Dataset"
        echo "  ${YELLOW}2)${NC} Start Training"
        echo "  ${YELLOW}3)${NC} Monitor Training"
        echo "  ${YELLOW}4)${NC} Run Inference"
        echo "  ${YELLOW}5)${NC} Open JupyterLab"
        echo "  ${YELLOW}6)${NC} System Status"
        echo "  ${YELLOW}7)${NC} Cleanup"
        echo "  ${YELLOW}8)${NC} Help"
        echo "  ${YELLOW}9)${NC} Exit"
        echo ""
        read -p "Enter choice [1-9]: " choice

        case $choice in
            1) prepare_dataset ;;
            2) start_training ;;
            3) monitor_training ;;
            4) run_inference ;;
            5) open_jupyter ;;
            6) system_status ;;
            7) cleanup ;;
            8) show_help ;;
            9) exit 0 ;;
            *) echo -e "${RED}Invalid choice${NC}" ; sleep 2 ;;
        esac
    done
}

# Prepare dataset
prepare_dataset() {
    show_header
    echo -e "${GREEN}Dataset Preparation${NC}"
    echo ""

    # Check for videos
    video_count=$(find /workspace/LTX_video_training/raw_videos -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | wc -l)
    echo "Found $video_count videos in /workspace/LTX_video_training/raw_videos/"

    if [ $video_count -eq 0 ]; then
        echo -e "${RED}No videos found!${NC}"
        echo "Please upload videos to /workspace/LTX_video_training/raw_videos/ first."
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    read -p "Prepare dataset from these videos? (y/n): " confirm

    if [ "$confirm" = "y" ]; then
        echo ""
        echo "Starting dataset preparation..."
        python scripts/prepare_dataset_single_source.py \
            --input_dir /workspace/LTX_video_training/raw_videos \
            --output_dir /workspace/LTX_video_training/dataset \
            --num_workers 4 \
            --visualization_type multi

        echo ""
        echo -e "${GREEN}Dataset preparation complete!${NC}"
        read -p "Press Enter to continue..."
    fi
}

# Start training
start_training() {
    show_header
    echo -e "${GREEN}Start Training${NC}"
    echo ""

    # Check if dataset exists
    if [ ! -f "./dataset/train.csv" ]; then
        echo -e "${RED}No dataset found!${NC}"
        echo "Please prepare dataset first (option 1)."
        read -p "Press Enter to continue..."
        return
    fi

    echo "Available configurations:"
    echo "  1) High Quality (A100/A6000, 10K steps)"
    echo "  2) Medium Quality (RTX 4090, 5K steps)"
    echo "  3) Quick Test (Any GPU, 1K steps)"
    echo "  4) Custom config"
    echo ""
    read -p "Select config [1-4]: " config_choice

    case $config_choice in
        1) config_file="configs/trajectory_iclora_high_quality.yaml" ;;
        2) config_file="configs/trajectory_iclora_medium.yaml" ;;
        3) config_file="configs/trajectory_iclora_quick_test.yaml" ;;
        4)
            read -p "Enter config path: " config_file
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            sleep 2
            return
            ;;
    esac

    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Config file not found: $config_file${NC}"
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    echo "Starting training with config: $config_file"
    echo ""
    read -p "Continue? (y/n): " confirm

    if [ "$confirm" = "y" ]; then
        ./scripts/train.sh "$config_file" 1

        echo ""
        echo -e "${GREEN}Training started!${NC}"
        echo "Monitor progress with option 3."
        read -p "Press Enter to continue..."
    fi
}

# Monitor training
monitor_training() {
    show_header
    echo -e "${GREEN}Training Monitor${NC}"
    echo ""
    echo "  1) View training logs"
    echo "  2) Monitor GPU usage"
    echo "  3) View TensorBoard"
    echo "  4) Check latest checkpoint"
    echo "  5) Back"
    echo ""
    read -p "Choose option [1-5]: " monitor_choice

    case $monitor_choice in
        1)
            if [ -d "./logs" ]; then
                echo "Showing latest log file..."
                tail -f ./logs/*.log 2>/dev/null || echo "No log files found"
            else
                echo "No logs directory found"
            fi
            ;;
        2)
            watch -n 1 nvidia-smi
            ;;
        3)
            echo "Starting TensorBoard on port 6006..."
            tensorboard --logdir ./logs --port 6006 --bind_all &
            echo "Access at: http://your-runpod-url:6006"
            read -p "Press Enter to continue..."
            ;;
        4)
            if [ -d "./checkpoints" ]; then
                ls -lht ./checkpoints | head -10
            else
                echo "No checkpoints found"
            fi
            read -p "Press Enter to continue..."
            ;;
        5)
            return
            ;;
    esac
}

# Run inference
run_inference() {
    show_header
    echo -e "${GREEN}Run Inference${NC}"
    echo ""

    echo "  1) Use Jupyter notebook (recommended)"
    echo "  2) Command line interface"
    echo "  3) Back"
    echo ""
    read -p "Choose option [1-3]: " infer_choice

    case $infer_choice in
        1)
            echo "Opening notebook..."
            echo "Navigate to: notebooks/04_Inference.ipynb"
            read -p "Press Enter to continue..."
            ;;
        2)
            echo "Inference CLI"
            echo ""
            read -p "Input image path: " input_image
            read -p "Reference video path: " ref_video
            read -p "Prompt: " prompt
            read -p "Output path: " output_path

            python src/inference/trajectory_guided_inference.py \
                --model_path ./models/ltx-video-13b \
                --lora_path ./output/trajectory_iclora/final \
                --mode reference \
                --input_image "$input_image" \
                --reference_video "$ref_video" \
                --prompt "$prompt" \
                --output "$output_path"

            echo ""
            echo -e "${GREEN}Inference complete!${NC}"
            echo "Output: $output_path"
            read -p "Press Enter to continue..."
            ;;
        3)
            return
            ;;
    esac
}

# Open JupyterLab
open_jupyter() {
    show_header
    echo -e "${GREEN}JupyterLab${NC}"
    echo ""
    echo "JupyterLab should already be running."
    echo ""
    echo "Access at: http://your-runpod-url:8888"
    echo ""
    echo "Notebooks location: /workspace/LTX_video_training/notebooks/"
    echo ""
    read -p "Press Enter to continue..."
}

# System status
system_status() {
    show_header
    echo -e "${GREEN}System Status${NC}"
    echo ""

    # GPU
    echo -e "${BLUE}GPU Information:${NC}"
    nvidia-smi
    echo ""

    # Disk
    echo -e "${BLUE}Disk Usage:${NC}"
    df -h /workspace
    echo ""

    # Python packages
    echo -e "${BLUE}Key Packages:${NC}"
    pip show torch transformers diffusers | grep -E "Name|Version"
    echo ""

    # Processes
    echo -e "${BLUE}Training Processes:${NC}"
    ps aux | grep python | grep -v grep || echo "No training processes running"
    echo ""

    read -p "Press Enter to continue..."
}

# Cleanup
cleanup() {
    show_header
    echo -e "${YELLOW}Cleanup${NC}"
    echo ""
    echo "This will remove:"
    echo "  - Temporary files"
    echo "  - Pip cache"
    echo "  - Old checkpoints (keeps latest 3)"
    echo ""
    read -p "Continue? (y/n): " confirm

    if [ "$confirm" = "y" ]; then
        echo "Cleaning up..."

        # Remove temp files
        rm -rf ./tmp/*
        rm -rf /tmp/*
        rm -rf ~/.cache/pip

        # Keep only latest 3 checkpoints
        cd ./checkpoints 2>/dev/null && ls -t checkpoint-* 2>/dev/null | tail -n +4 | xargs rm -rf {} 2>/dev/null
        cd $WORKSPACE

        echo -e "${GREEN}Cleanup complete!${NC}"
        df -h /workspace
        read -p "Press Enter to continue..."
    fi
}

# Help
show_help() {
    show_header
    echo -e "${GREEN}Help & Documentation${NC}"
    echo ""
    echo "Documentation files:"
    echo "  - README.md: Main documentation"
    echo "  - RUNPOD_DEPLOYMENT.md: RunPod-specific guide"
    echo "  - docs/BEST_PRACTICES.md: Training best practices"
    echo ""
    echo "Notebooks:"
    echo "  - 01_Quick_Start.ipynb: Quick start guide"
    echo "  - 02_Dataset_Preparation.ipynb: Dataset prep"
    echo "  - 03_Training.ipynb: Training guide"
    echo "  - 04_Inference.ipynb: Inference guide"
    echo ""
    echo "Useful commands:"
    echo "  - Monitor GPU: watch -n 1 nvidia-smi"
    echo "  - View logs: tail -f logs/*.log"
    echo "  - Check disk: df -h /workspace"
    echo ""
    read -p "Press Enter to continue..."
}

# Run main menu
main_menu
