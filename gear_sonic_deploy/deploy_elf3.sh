#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

get_network_interfaces() {
    if [[ "$(uname)" == "Darwin" ]]; then
        ifconfig | awk '/^[a-z]/ { iface=$1; gsub(/:$/, "", iface) } /inet / { print iface ":" $2 }'
    else
        ip -4 addr show 2>/dev/null | awk '/^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 } /inet / { split($2,a,"/"); print iface ":" a[1] }' 2>/dev/null || ifconfig 2>/dev/null | awk '/^[a-z]/ { iface=$1; gsub(/:$/, "", iface) } /inet / { for(i=1;i<=NF;i++){ if($i=="inet"){ print iface ":" $(i+1); break } if($i~/^addr:/){ split($i,a,":"); print iface ":" a[2]; break } } }'
    fi
}

find_interface_by_ip() {
    local target_ip="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$target_ip" ]]; then
            echo "$iface"
            return 0
        fi
    done
}

find_interface_by_ip_prefix() {
    local prefix="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$prefix"* ]]; then
            echo "$iface"
            return 0
        fi
    done
}

is_ip_address() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

interface_has_ip() {
    local iface="$1"
    local target_ip="$2"
    get_network_interfaces | while IFS=: read -r name ip; do
        if [[ "$name" == "$iface" && "$ip" == "$target_ip" ]]; then
            echo "yes"
            return 0
        fi
    done
}

resolve_interface() {
    local interface="$1"
    local os_type
    os_type="$(uname)"
    if is_ip_address "$interface"; then
        TARGET="$interface"
        [[ "$interface" == "127.0.0.1" ]] && ENV_TYPE="sim" || ENV_TYPE="real"
        return 0
    fi
    if [[ "$interface" == "sim" ]]; then
        local lo_interface
        lo_interface=$(find_interface_by_ip "127.0.0.1")
        if [[ -n "$lo_interface" ]]; then
            if [[ "$os_type" == "Darwin" && "$lo_interface" == "lo" ]]; then TARGET="lo0"; else TARGET="$lo_interface"; fi
        else
            if [[ "$os_type" == "Darwin" ]]; then TARGET="lo0"; else TARGET="lo"; fi
        fi
        ENV_TYPE="sim"
        return 0
    fi
    if [[ "$interface" == "real" ]]; then
        local real_interface
        real_interface=$(find_interface_by_ip_prefix "192.168.123.")
        if [[ -n "$real_interface" ]]; then
            TARGET="$real_interface"
        else
            local fallback_interface
            fallback_interface=$(get_network_interfaces | grep -v "127.0.0.1" | head -1 | cut -d: -f1)
            if [[ -n "$fallback_interface" ]]; then
                TARGET="$fallback_interface"
                echo -e "${YELLOW}WARNING: ELF3 real interface not auto-detected, using $TARGET${NC}"
            else
                echo -e "${RED}ERROR: No usable real network interface found.${NC}"
                exit 1
            fi
        fi
        ENV_TYPE="real"
        return 0
    fi
    local has_loopback
    has_loopback=$(interface_has_ip "$interface" "127.0.0.1")
    if [[ "$has_loopback" == "yes" ]]; then
        TARGET="$interface"
        ENV_TYPE="sim"
        return 0
    fi
    if [[ "$os_type" == "Darwin" && "$interface" == "lo" ]]; then
        TARGET="lo0"
        ENV_TYPE="sim"
        return 0
    fi
    TARGET="$interface"
    ENV_TYPE="real"
}

show_usage() {
    echo "Usage: $0 [OPTIONS] [sim|real|<interface>]"
    echo ""
    echo "Options:"
    echo "  -h, --help"
    echo "  --cp, --checkpoint PATH     Model prefix, without _encoder.onnx/_decoder.onnx"
    echo "  --obs-config PATH           ELF3 observation config"
    echo "  --planner PATH              Optional planner ONNX"
    echo "  --no-planner                Disable planner"
    echo "  --motion-data PATH          ELF3 deployment reference directory"
    echo "  --input-type TYPE           Default: keyboard"
    echo "  --output-type TYPE          Default: all"
    echo "  --zmq-host HOST             Default: localhost"
    echo "  --motor-kp-scale SPEC"
    echo "  --motor-kd-scale SPEC"
    echo "  --reference-blend-duration SECONDS  Normal-mode playback fade, default: 1.0; 0 disables"
    echo ""
    echo "Examples:"
    echo "  $0 sim"
    echo "  $0 --motion-data reference/elf3 sim"
    echo "  $0 --cp policy/elf3/model_step_002350 sim"
    echo "  $0 --reference-blend-duration 1.0 --no-planner sim"
    echo "  $0 --planner planner/custom.onnx --input-type manager sim"
}

INTERFACE_MODE="sim"
CHECKPOINT_DEFAULT="policy/elf3/model"
OBS_CONFIG_DEFAULT="policy/elf3/observation_config.yaml"
PLANNER_DEFAULT=""
MOTION_DATA_DEFAULT="reference/elf3"
INPUT_TYPE_DEFAULT="keyboard"
OUTPUT_TYPE_DEFAULT="all"
ZMQ_HOST_DEFAULT="localhost"

CHECKPOINT="$CHECKPOINT_DEFAULT"
OBS_CONFIG="$OBS_CONFIG_DEFAULT"
PLANNER="$PLANNER_DEFAULT"
MOTION_DATA="$MOTION_DATA_DEFAULT"
INPUT_TYPE="$INPUT_TYPE_DEFAULT"
OUTPUT_TYPE="$OUTPUT_TYPE_DEFAULT"
ZMQ_HOST="$ZMQ_HOST_DEFAULT"
MOTOR_KP_SCALES=()
MOTOR_KD_SCALES=()
REFERENCE_BLEND_DURATION="1.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --cp|--checkpoint)
            [[ -z "${2:-}" ]] && { echo "ERROR: $1 requires a path"; exit 1; }
            CHECKPOINT="$2"
            shift 2
            ;;
        --obs-config)
            [[ -z "${2:-}" ]] && { echo "ERROR: --obs-config requires a path"; exit 1; }
            OBS_CONFIG="$2"
            shift 2
            ;;
        --planner)
            [[ -z "${2:-}" ]] && { echo "ERROR: --planner requires a path"; exit 1; }
            PLANNER="$2"
            shift 2
            ;;
        --no-planner)
            PLANNER=""
            shift
            ;;
        --motion-data)
            [[ -z "${2:-}" ]] && { echo "ERROR: --motion-data requires a path"; exit 1; }
            MOTION_DATA="$2"
            shift 2
            ;;
        --input-type)
            [[ -z "${2:-}" ]] && { echo "ERROR: --input-type requires a type"; exit 1; }
            INPUT_TYPE="$2"
            shift 2
            ;;
        --output-type)
            [[ -z "${2:-}" ]] && { echo "ERROR: --output-type requires a type"; exit 1; }
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --zmq-host)
            [[ -z "${2:-}" ]] && { echo "ERROR: --zmq-host requires a host"; exit 1; }
            ZMQ_HOST="$2"
            shift 2
            ;;
        --motor-kp-scale)
            [[ -z "${2:-}" ]] && { echo "ERROR: --motor-kp-scale requires <motor-list>=<factor>"; exit 1; }
            MOTOR_KP_SCALES+=("$2")
            shift 2
            ;;
        --motor-kd-scale)
            [[ -z "${2:-}" ]] && { echo "ERROR: --motor-kd-scale requires <motor-list>=<factor>"; exit 1; }
            MOTOR_KD_SCALES+=("$2")
            shift 2
            ;;
        --reference-blend-duration)
            [[ -z "${2:-}" ]] && { echo "ERROR: --reference-blend-duration requires seconds"; exit 1; }
            REFERENCE_BLEND_DURATION="$2"
            # The C++ parser also rejects non-finite values and numeric overflow.
            if [[ ! "$REFERENCE_BLEND_DURATION" =~ ^\+?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
                echo "ERROR: --reference-blend-duration must be a non-negative number"
                exit 1
            fi
            shift 2
            ;;
        sim|real)
            INTERFACE_MODE="$1"
            shift
            ;;
        *)
            INTERFACE_MODE="$1"
            shift
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        ELF3 DEPLOY LAUNCHER                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

resolve_interface "$INTERFACE_MODE"

CHECKPOINT_DECODER="${CHECKPOINT}_decoder.onnx"
CHECKPOINT_ENCODER="${CHECKPOINT}_encoder.onnx"

EXTRA_ARGS=()
if [[ "$ENV_TYPE" == "sim" ]]; then
    EXTRA_ARGS+=("--disable-crc-check")
fi
for scale in "${MOTOR_KP_SCALES[@]}"; do EXTRA_ARGS+=("--motor-kp-scale" "$scale"); done
for scale in "${MOTOR_KD_SCALES[@]}"; do EXTRA_ARGS+=("--motor-kd-scale" "$scale"); done

if [[ -z "$PLANNER" ]]; then
    case "$INPUT_TYPE" in
        ros2|gamepad_manager|zmq_manager)
            echo -e "${RED}ERROR: input type '$INPUT_TYPE' requires a planner model.${NC}"
            exit 1
            ;;
    esac
fi

check_file() {
    if [[ ! -f "$1" ]]; then
        echo -e "${RED}Missing file: $1${NC}"
        return 1
    fi
    echo -e "${GREEN}Found: $1${NC}"
}

echo -e "${BLUE}[1/4] Checking ELF3 deployment files...${NC}"
FAILED=0
check_file "$CHECKPOINT_DECODER" || FAILED=1
check_file "$CHECKPOINT_ENCODER" || FAILED=1
check_file "$OBS_CONFIG" || FAILED=1
if [[ -n "$PLANNER" ]]; then check_file "$PLANNER" || FAILED=1; fi
if [[ ! -d "$MOTION_DATA" ]]; then
    echo -e "${RED}Missing motion directory: $MOTION_DATA${NC}"
    FAILED=1
else
    echo -e "${GREEN}Found: $MOTION_DATA${NC}"
fi
if [[ "$FAILED" -ne 0 ]]; then
    echo -e "${RED}ELF3 deployment prerequisites failed.${NC}"
    exit 1
fi

if [[ -z "${TensorRT_ROOT:-}" ]]; then
    if [[ -d "$HOME/TensorRT" ]]; then
        export TensorRT_ROOT="$HOME/TensorRT"
        echo -e "${GREEN}TensorRT_ROOT=$TensorRT_ROOT${NC}"
    else
        echo -e "${YELLOW}WARNING: TensorRT_ROOT is not set.${NC}"
    fi
fi

echo -e "${BLUE}[2/4] Checking build dependencies...${NC}"
if ! command -v just >/dev/null 2>&1; then
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
fi
for cmd in cmake clang git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        chmod +x scripts/install_deps.sh
        ./scripts/install_deps.sh
        break
    fi
done

echo -e "${BLUE}[3/4] Building ELF3 deployment target...${NC}"
set +e
source scripts/setup_env.sh
SETUP_RET=$?
set -e
if [[ "$SETUP_RET" -ne 0 ]]; then echo -e "${YELLOW}setup_env.sh returned $SETUP_RET; continuing.${NC}"; fi
just build

echo -e "${BLUE}[4/4] Deployment configuration${NC}"
echo "Environment:       $ENV_TYPE"
echo "Interface:         $TARGET"
echo "Decoder:           $CHECKPOINT_DECODER"
echo "Encoder:           $CHECKPOINT_ENCODER"
echo "Observation config:$OBS_CONFIG"
echo "Motion data:       $MOTION_DATA"
echo "Planner:           ${PLANNER:-disabled}"
echo "Input type:        $INPUT_TYPE"
echo "Output type:       $OUTPUT_TYPE"
echo "ZMQ host:          $ZMQ_HOST"
echo "Reference blend:   ${REFERENCE_BLEND_DURATION}s (Normal mode; 0 disables)"
echo ""

RUN_ARGS=("$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" "--obs-config" "$OBS_CONFIG" "--encoder-file" "$CHECKPOINT_ENCODER" "--input-type" "$INPUT_TYPE" "--output-type" "$OUTPUT_TYPE" "--zmq-host" "$ZMQ_HOST")
if [[ -n "$PLANNER" ]]; then RUN_ARGS+=("--planner-file" "$PLANNER"); fi
RUN_ARGS+=("--reference-blend-duration" "$REFERENCE_BLEND_DURATION")
RUN_ARGS+=("${EXTRA_ARGS[@]}")

echo -e "${CYAN}Command:${NC}"
printf "just run elf3_deploy_onnx_ref"
printf " %q" "${RUN_ARGS[@]}"
echo ""
echo ""

if [[ "$ENV_TYPE" == "real" ]]; then
    echo -e "${RED}WARNING: ELF3 REAL mode selected.${NC}"
    echo -e "${YELLOW}Only continue after the ELF3 real hardware communication layer has been ported and verified.${NC}"
else
    echo -e "${GREEN}ELF3 MuJoCo sim2sim mode selected.${NC}"
fi

read -r -p "Proceed with deployment? [Y/n]: " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

just run elf3_deploy_onnx_ref "${RUN_ARGS[@]}"
