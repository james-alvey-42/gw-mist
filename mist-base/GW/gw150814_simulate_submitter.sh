#!/bin/bash
#SBATCH --job-name=python_job          # Job name
#SBATCH --output=output.txt            # Standard output log
#SBATCH --error=error.txt              # Standard error log
#SBATCH --time=100:00:00                # Time limit hrs:min:sec (optional)

python gw150814_simulate.py