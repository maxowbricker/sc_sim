To run your first full training session on the RMIT RACE cluster using your current "Gold Standard" setup, here is the recap of the compute choices and the execution command.

### **1. Recommended Compute Configurations**
Based on your code's optimization for CPU-bound physics, we narrowed it down to these two primary options:

* **Option A: The Performance Winner (C08 Compute Optimised - 4xLarge)**
    * **Specs**: 8 CPUs (16 threads), 32 GiB Memory, 4th Gen Intel Xeon Sapphire Rapid (up to 3.8 GHz).
    * **Why**: This is the fastest option for your engine. The high clock speed will process the $O(1)$ metric calculations and float math much faster than the older T4 instances.
    * **Cost**: $0.93 USD/Hour.

* **Option B: The Budget Powerhouse (C16 Balanced (Flex) - 2xLarge)**
    * **Specs**: 4 CPUs (8 threads), 32 GiB Memory, 4th Gen Intel Xeon Sapphire Rapid.
    * **Why**: It uses the same modern, fast processors but with fewer cores. Use this if you want to save budget and are okay running with `--num-cpu 4` instead of 8.
    * **Cost**: $0.48 USD/Hour.

### **2. The Training Command**
Once you have `rsync`-ed your data and the updated code to the workspace, navigate to your project root and run:

```bash
python rl/train_sb3.py --timesteps 100000 --num-cpu 8 --hyperparams rl/best_hyperparameters.json
```

**What this command does:**
* **`--timesteps 100000`**: Runs a full 100k step training session (approx. 2,000 episodes).
* **`--num-cpu 8`**: Spawns 8 parallel environments to collect experience 8x faster.
* **`--hyperparams rl/best_hyperparameters.json`**: Tells the script to load the optimal "brain settings" found during your Optuna study (e.g., the `large` [256, 256] network architecture and `gamma: 0.95`).

### **3. Final Checklist Before Launch**
* **Data Sync**: Ensure your `data/` folder is fully synced to the RACE machine (approximately 90GB).
* **Keep Awake**: If you are running this from a terminal that might disconnect, remember to use `nohup` and `caffeinate` as we did locally:
    `nohup caffeinate -i python rl/train_sb3.py ... > train.log 2>&1 &`.
* **Branch Check**: Make sure you are running this on your `conference-ready` branch to ensure you are using the most stable, non-experimental code.