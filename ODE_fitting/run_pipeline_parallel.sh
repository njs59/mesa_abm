#!/usr/bin/env bash
set -euo pipefail

# ============================================
# CONFIGURATION
# ============================================

CONFIG_PATH="pipeline/pipeline_config.yaml"
PACKAGE="pipeline"      # python package name containing pipeline_mle_only.py

# Colours for nice output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}>>> 1. Preparing sweep...${NC}"

# ============================================
# 1. PREPARE SCENARIOS
# ============================================

PREP_OUTPUT=$(python -m ${PACKAGE}.pipeline_mle_only prepare --config ${CONFIG_PATH})

# Extract ONLY the real RUN_ROOT (ignore the instructional export lines)
RUN_ROOT=$(echo "$PREP_OUTPUT" \
  | grep "^RUN_ROOT=" \
  | head -n 1 \
  | sed 's/RUN_ROOT=//')

if [ -z "$RUN_ROOT" ]; then
    echo "ERROR: Could not extract RUN_ROOT from prepare output!"
    echo "Prepare output was:"
    echo "$PREP_OUTPUT"
    exit 1
fi

echo -e "${GREEN}>>> RUN_ROOT detected:${NC} $RUN_ROOT"


# ============================================
# 2. COUNT SCENARIOS
# ============================================

SCENARIOS_JSON="$RUN_ROOT/scenarios.json"

if [ ! -f "$SCENARIOS_JSON" ]; then
    echo "ERROR: scenarios.json not found at:"
    echo "  $SCENARIOS_JSON"
    exit 1
fi

# Requires jq (JSON parser)
NUM_SCENARIOS=$(jq '. | length' "$SCENARIOS_JSON")
LAST_INDEX=$((NUM_SCENARIOS - 1))

echo -e "${GREEN}>>> Found $NUM_SCENARIOS scenarios (indices 0 .. $LAST_INDEX)${NC}"


# ============================================
# 3. RUN SCENARIOS IN PARALLEL
# ============================================

echo -e "${GREEN}>>> Running all scenarios in parallel ...${NC}"

parallel --jobs 0 "
  python -m ${PACKAGE}.pipeline_mle_only run-scenario \
      --config ${CONFIG_PATH} \
      --run-root ${RUN_ROOT} \
      --scenario-index {}
" ::: $(seq 0 $LAST_INDEX)


# ============================================
# 4. FINALIZE
# ============================================

echo -e "${GREEN}>>> Finalizing ...${NC}"

python -m ${PACKAGE}.pipeline_mle_only finalize \
    --config ${CONFIG_PATH} \
    --run-root ${RUN_ROOT}


# ============================================
# 5. DONE
# ============================================

echo -e "${GREEN}>>> ALL DONE.${NC}"
echo "  Run folder:           $RUN_ROOT"
echo "  Consolidated results: $RUN_ROOT/mle_results.csv"
echo "  Plots:                $RUN_ROOT/plots"