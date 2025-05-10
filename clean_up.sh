
#!/bin/bash

# Clean up baseline/ and hyperparameter_tuning/ directories

echo "Cleaning up baseline/..."
rm -v baseline/*.csv 2>/dev/null

echo "Cleaning up hyperparameter_tuning/..."
# rm -v hyperparameter_tuning/*.csv 2>/dev/null
rm -v hyperparameter_tuning/training_logs/*.csv 2>/dev/null
rm -v hyperparameter_tuning/q_tables/*.csv 2>/dev/null

# Remove all .csv files in the modified_environments/ directory
echo "Cleaning up modified_environments/..."
rm -v modified_environments/*.csv 2>/dev/null

echo "Cleanup complete."