#!/usr/bin/env python3
"""
Script to add remaining cells to the Experiment 015 analysis notebook.
"""

import json

# Read existing notebook
with open('analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Define new cells to add
new_cells = [
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Categorize experiments\n",
            "def categorize_experiment(row):\n",
            "    \"\"\"Categorize experiment by type.\"\"\"\n",
            "    name = row['exp_name']\n",
            "    strategy = row['strategy']\n",
            "    \n",
            "    if strategy == 'greedy':\n",
            "        return 'Greedy'\n",
            "    elif strategy == 'laf':\n",
            "        return 'LAF'\n",
            "    elif strategy == 'random_assign':\n",
            "        return 'Random'\n",
            "    elif strategy == 'ewma_only':\n",
            "        return 'EWMA-Only'\n",
            "    elif strategy == 'composite':\n",
            "        if 'Pareto' in name:\n",
            "            return 'Composite'\n",
            "        elif 'Gamma' in name:\n",
            "            return 'Composite (Gamma)'\n",
            "    return 'Other'\n",
            "\n",
            "df['category'] = df.apply(categorize_experiment, axis=1)\n",
            "\n",
            "# Extract parameters for Pareto experiments\n",
            "def extract_lambda_params(name):\n",
            "    \"\"\"Extract λ₁ and λ₃ from experiment name.\"\"\"\n",
            "    if 'Pareto_L1' in name:\n",
            "        import re\n",
            "        match = re.search(r'L1_(\\d+\\.\\d+)_L3_(\\d+\\.\\d+)', name)\n",
            "        if match:\n",
            "            return float(match.group(1)), float(match.group(2))\n",
            "    return None, None\n",
            "\n",
            "df['lambda_1'], df['lambda_3'] = zip(*df['exp_name'].apply(extract_lambda_params))\n",
            "\n",
            "# Extract gamma for gamma sensitivity experiments\n",
            "def extract_gamma(name):\n",
            "    \"\"\"Extract gamma from experiment name.\"\"\"\n",
            "    if 'Gamma' in name and '_G_' in name:\n",
            "        import re\n",
            "        match = re.search(r'_G_(\\d+\\.\\d+)', name)\n",
            "        if match:\n",
            "            return float(match.group(1))\n",
            "    elif 'EWMA_Only' in name:\n",
            "        return 0.5  # Default gamma for EWMA-Only\n",
            "    return None\n",
            "\n",
            "df['gamma'] = df['exp_name'].apply(extract_gamma)\n",
            "\n",
            "print(\"\\n📊 Experiment Categories:\")\n",
            "print(df['category'].value_counts())\n",
            "print(f\"\\n✅ Data preprocessing complete\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n# Section 1: Executive Summary & Key Findings\n\nHigh-level overview comparing all strategy types."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Summary statistics by category\n",
            "summary_stats = df.groupby('category').agg({\n",
            "    'jains_fairness_index': ['mean', 'std', 'min', 'max'],\n",
            "    'mean_wait_time_minutes': ['mean', 'std', 'min', 'max'],\n",
            "    'completed_tasks': ['mean', 'std'],\n",
            "    'mean_worker_utilization': ['mean', 'std'],\n",
            "    'ewma_final_mean': ['mean', 'std']\n",
            "}).round(3)\n",
            "\n",
            "print(\"📊 Summary Statistics by Strategy Category\")\n",
            "print(\"=\" * 80)\n",
            "summary_stats"
        ]
    }
]

# Add new cells to notebook
nb['cells'].extend(new_cells)

# Save updated notebook
with open('analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"✅ Added {len(new_cells)} cells to notebook")
print(f"   Total cells: {len(nb['cells'])}")






