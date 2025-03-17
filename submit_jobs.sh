#!/bin/bash
# Script to submit and manage bike path optimization jobs

# Default values
DEFAULT_CITY="Otley, UK"
DEFAULT_BUDGET=100000
DEFAULT_TIMESTEPS_ARRAY=(10240 20480 51200 102400)
DEFAULT_EVAL_EPISODES=10
DEFAULT_CPUS=8
DEFAULT_MEMORY=16G
DEFAULT_TIME="2:00:00"
DEFAULT_EMAIL="ungkuamer@gmail.com"

# Parse command-line arguments
CITY=${1:-"$DEFAULT_CITY"}
BUDGET=${2:-$DEFAULT_BUDGET}

# Print banner
echo "======================================================"
echo "      Bike Path Optimization Job Submission"
echo "======================================================"
echo "This script will submit jobs to optimize bike paths"
echo "for the specified city with different timestep values."
echo

# Confirm settings
echo "Job Settings:"
echo "  City: $CITY"
echo "  Budget: $BUDGET"
echo "  Timesteps to test: ${DEFAULT_TIMESTEPS_ARRAY[*]}"
echo "  Evaluation episodes per run: $DEFAULT_EVAL_EPISODES"
echo "  CPUs per job: $DEFAULT_CPUS"
echo "  Memory per job: $DEFAULT_MEMORY"
echo "  Time limit: $DEFAULT_TIME"
echo "  Notification email: $DEFAULT_EMAIL"
echo
read -p "Continue with these settings? (y/n): " CONFIRM

if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# Create a temporary SLURM script with the provided parameters
TMP_SCRIPT=$(mktemp)

cat > $TMP_SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=bike_path_rl
#SBATCH --output=bike_path_rl_%A_%a.out
#SBATCH --error=bike_path_rl_%A_%a.err
#SBATCH --time=$DEFAULT_TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$DEFAULT_CPUS
#SBATCH --mem=$DEFAULT_MEMORY
#SBATCH --array=0-$((${#DEFAULT_TIMESTEPS_ARRAY[@]}-1))
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$DEFAULT_EMAIL

# Configuration
CITY="$CITY"
BUDGET=$BUDGET
EVAL_EPISODES=$DEFAULT_EVAL_EPISODES
DEVICE="cpu"

# Create array of different timestep values to test
TIMESTEPS_ARRAY=(${DEFAULT_TIMESTEPS_ARRAY[*]})
TIMESTEPS=\${TIMESTEPS_ARRAY[\$SLURM_ARRAY_TASK_ID]}

# Output directory - each run will have its own folder
RUN_DIR="bike_path_results_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p \$RUN_DIR
cd \$RUN_DIR

# Print job information
echo "=========================================="
echo "Job Information:"
echo "SLURM Job ID: \$SLURM_JOB_ID"
echo "SLURM Array Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Running on node: \$SLURM_NODELIST"
echo "Number of CPUs: \$SLURM_CPUS_PER_TASK"
echo "Started at: \$(date)"
echo "=========================================="

# Print run configuration
echo "Run Configuration:"
echo "City: \$CITY"
echo "Budget: \$BUDGET"
echo "Timesteps: \$TIMESTEPS"
echo "Evaluation Episodes: \$EVAL_EPISODES"
echo "Device: \$DEVICE"
echo "Output Directory: \$RUN_DIR"
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
cat > config.txt << EOL
City: \$CITY
Budget: \$BUDGET
Timesteps: \$TIMESTEPS
Evaluation Episodes: \$EVAL_EPISODES
Device: \$DEVICE
Started: \$(date)
EOL

# Run the script with the specified timesteps and number of CPUs
echo "Starting bike path optimization..."
python nx-rx-simple-ur.py --city "\$CITY" --budget \$BUDGET --timesteps \$TIMESTEPS --eval_episodes \$EVAL_EPISODES --device \$DEVICE --n_envs \$((\$SLURM_CPUS_PER_TASK-1))

# Save completion time
echo "Completed: \$(date)" >> config.txt

# Compress the results directory
cd ..
tar -czf \${RUN_DIR}.tar.gz \$RUN_DIR

echo "=========================================="
echo "Job completed at: \$(date)"
echo "Results saved to: \${RUN_DIR}.tar.gz"
echo "=========================================="

# Deactivate virtual environment
deactivate
EOF

# Submit the job
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch $TMP_SCRIPT | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "Success! Job array submitted with ID: $JOB_ID"
    echo "Individual jobs will run with the following timesteps:"
    
    for i in "${!DEFAULT_TIMESTEPS_ARRAY[@]}"; do
        echo "  Job $JOB_ID.$i: ${DEFAULT_TIMESTEPS_ARRAY[$i]} timesteps"
    done
    
    echo
    echo "Monitor your jobs with:"
    echo "  squeue -u $USER"
    echo "  sacct -j $JOB_ID"
    echo
else
    echo "Error: Job submission failed."
fi

# Clean up temporary file
rm $TMP_SCRIPT