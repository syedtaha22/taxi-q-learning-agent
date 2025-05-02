
#!/bin/bash

# Clean up baseline/ and tuning/ directories

echo "Cleaning up baseline/..."
rm -v baseline/*.csv 2>/dev/null

echo "Cleaning up hyperparameter_tuning/..."
# rm -v hyperparameter_tuning/*.csv 2>/dev/null
rm -v hyperparameter_tuning/training_logs/*.csv 2>/dev/null
rm -v hyperparameter_tuning/q_tables/*.csv 2>/dev/null

echo "Cleanup complete."