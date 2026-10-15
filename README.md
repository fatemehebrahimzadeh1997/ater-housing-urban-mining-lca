# Post-War Structural Housing Stock: Monte Carlo LCA & Sensitivity Pipeline

A publication-grade data engineering pipeline reconstructing a building-scale Life Cycle Assessment (LCA) for a representative post-war social housing complex (ATER complex in Rovigo, Italy) based on original architectural and structural drawing inventory datasets.

This model calculates production impacts (A1–A3), demolition burdens (C1), transport emissions (C2), and potential circular economy benefits beyond the system boundary (**EN 15978 Module D Avoided Burdens**).

## Key Framework Features
*   **Stochastic Modeling Engine:** Runs a 30,000-row total probabilistic simulation (10,000 iterations per scenario) utilizing vectorized uniform distributions to handle parameter variability in material recovery rates.
*   **EN 15978 Module D Separability:** Maintains strict systemic boundary isolation between upfront gross impacts (A–C) and avoided-burden circular credits (Module D), calculating credits via primary vs. secondary material offset differentials.
*   **Scenario-Specific Sensitivity (Pearson's r):** Evaluates parameter correlations within isolated scenario boundaries to track how material-specific recovery thresholds alter global building-scale carbon trajectories.
*   **Data Integrity & Reproducibility:** Anchored using a fixed random generation seed (`SEED = 42`) to guarantee identical, auditable results across execution environments.

## Core Generated Data Artifacts
*   `lca_pipeline.py`: Vectorized Python script managing numerical distributions and chart generations.
*   `outputs/summary_statistics.csv`: Final processed metrics tracking Mean, Median, SD, and 5th/95th percentiles.
*   `outputs/scenario_sensitivity.csv`: Pearson r correlation metrics demonstrating how steel loops heavily dominate structural variance.
*   `outputs/scenario_distributions.png`: Probability density distributions mapping overlapping scenario risk boundaries.
