#!/bin/bash
#SBATCH --job-name=bike_path_rl
#SBATCH --output=bike_path_rl_%A_%a.out
#SBATCH --error=bike_path_rl_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ungkuamer@gmail.com

# Configuration
CITY="Otley, UK"  # Replace with your city
BUDGET=300000
EVAL_EPISODES=10
DEVICE="cpu"

# Create array of different timestep values to test
TIMESTEPS_ARRAY=(10240 20480 51200 102400)
TIMESTEPS=${TIMESTEPS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Output directory - each run will have its own folder
RUN_DIR="bike_path_results_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $RUN_DIR
cd $RUN_DIR

# Print job information
echo "=========================================="
echo "Job Information:"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"
echo "=========================================="

# Print run configuration
echo "Run Configuration:"
echo "City: $CITY"
echo "Budget: $BUDGET"
echo "Timesteps: $TIMESTEPS"
echo "Evaluation Episodes: $EVAL_EPISODES"
echo "Device: $DEVICE"
echo "Output Directory: $RUN_DIR"
echo "=========================================="

# Load required modules (adjust based on your cluster's configuration)
module purge
module load python/3.9
module load gcc/11.2.0  # For compiling some dependencies

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy pandas matplotlib geopandas osmnx rustworkx gymnasium stable-baselines3 tqdm scipy shapely

# Copy the main script to the run directory
cp ../nx-rx-simple-ur.py .

# Create a configuration file for this run
cat > config.txt << EOF
City: $CITY
Budget: $BUDGET
Timesteps: $TIMESTEPS
Evaluation Episodes: $EVAL_EPISODES
Device: $DEVICE
Started: $(date)
EOF

# Run the script with the specified timesteps and number of CPUs
echo "Starting bike path optimization..."
python nx-rx-simple-ur.py --city "$CITY" --budget $BUDGET --timesteps $TIMESTEPS --eval_episodes $EVAL_EPISODES --device $DEVICE --n_envs $(($SLURM_CPUS_PER_TASK-1))

# Save completion time
echo "Completed: $(date)" >> config.txt

# Compress the results directory
cd ..
tar -czf ${RUN_DIR}.tar.gz $RUN_DIR

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results saved to: ${RUN_DIR}.tar.gz"
echo "=========================================="

# Deactivate virtual environment
deactivate