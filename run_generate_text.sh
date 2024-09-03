#!/usr/bin/bash
#SBATCH -o ./generated_samples/logs/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=32000                            # server memory requested (per node)
#SBATCH -t 960:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov                   # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1                           # Type/number of GPUs needed
#SBATCH --open-mode=append                     # Do not overwrite logs
#SBATCH --requeue                              # Requeue upon preemption
#SBATCH --mail-user=yzs2@cornell.edu           # Email
#SBATCH --mail-type=END                        # Request status by email

# for s in $(seq 1 20); do sbatch --job-name="ssdlm_generate_seed-${s}" run_generate_text.sh ${s}; done

source /share/kuleshov/yzs2/text-diffusion/setup_env.sh

SEED=${1}

python generate_text.py \
  --generated_samples_outfile="./generated_samples/generated_samples_seed-${SEED}.json" \
  --num_samples_to_generate=2 \
  --seed=${SEED} \
  --model_name_or_path="xhan77/ssdlm" \
  --use_slow_tokenizer=True \
  --max_seq_length=2050 \
  --one_hot_value=5 \
  --decoding_block_size=25 \
  --total_t=1000 \
  --projection_top_p=0.95
