#!/bin/bash
# ====================================================
#       Cross-Dataset Training & Evaluation Script
# ====================================================

DATA=/data
TRAINER=CoDA
CFG=xd
SHOTS=16
FOLDER=output_xd
GPU=5

for DATASET in imagenet; do
  for SEED in 1 2 3; do
    DIR=${FOLDER}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    [ -d "$DIR" ] && echo "[Skip] ${DIR} exists" && continue
    echo "[Run] Training ${DATASET}  seed=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      DATASET.NUM_SHOTS ${SHOTS}
  done
done

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r; do
  LOADEP=6
  for SEED in 1 2 3; do
    MODEL_DIR=${FOLDER}/base2new/train_base/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    DIR=${FOLDER}/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    [ -d "$DIR" ] && echo "[Skip] ${DIR} exists" && continue
    echo "[Run] Evaluating ${DATASET}  seed=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir ${MODEL_DIR} \
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS}
  done
done