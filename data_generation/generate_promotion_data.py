"""
Generates synthetic promotion-readiness data from scratch.
Target: promoted (1 = promoted within the review cycle, 0 = not promoted)

Independent schema from the attrition dataset on purpose - this model answers
a different business question ("is this employee ready to move up?"), so it
should not share features 1:1 with the attrition model.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_ROWS = 4000

DEPARTMENTS = ["Sales", "Engineering", "HR", "Finance", "Operations", "Support"]
EDUCATION_LEVELS = ["Bachelors", "Masters", "PhD", "Diploma"]


def generate_promotion_data(n_rows: int = N_ROWS) -> pd.DataFrame:
    age = RNG.integers(22, 55, n_rows)
    department = RNG.choice(DEPARTMENTS, n_rows)
    education_level = RNG.choice(EDUCATION_LEVELS, n_rows, p=[0.5, 0.3, 0.1, 0.1])
    years_in_current_role = RNG.integers(0, 12, n_rows)
    avg_performance_rating_3yr = RNG.uniform(1.0, 5.0, n_rows).round(2)
    kpi_met_percentage = RNG.integers(40, 100, n_rows)
    training_score = RNG.integers(0, 100, n_rows)
    num_projects_completed = RNG.integers(0, 20, n_rows)
    awards_won = RNG.integers(0, 4, n_rows)
    manager_rating = RNG.integers(1, 5, n_rows)             # 1 (low) - 4 (high)
    years_since_last_promotion = RNG.integers(0, 10, n_rows)
    peer_review_score = RNG.uniform(1.0, 5.0, n_rows).round(2)

    # --- logistic-style score so labels reflect real relationships ---
    edu_bonus = pd.Series(education_level).map(
        {"Bachelors": 0.0, "Masters": 0.3, "PhD": 0.5, "Diploma": -0.1}
    ).to_numpy()

    score = (
        0.55 * avg_performance_rating_3yr
        + 0.02 * kpi_met_percentage
        + 0.015 * training_score
        + 0.10 * num_projects_completed
        + 0.35 * awards_won
        + 0.45 * manager_rating
        + 0.30 * peer_review_score
        + edu_bonus
        - 0.20 * years_since_last_promotion
        - 0.05 * years_in_current_role
        - 7.6  # intercept tuned so ~20-25% get promoted (realistic base rate)
        + RNG.normal(0, 1.3, n_rows)
    )
    prob_promoted = 1 / (1 + np.exp(-score))
    promoted = (RNG.uniform(0, 1, n_rows) < prob_promoted).astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(n_rows)],
        "age": age,
        "department": department,
        "education_level": education_level,
        "years_in_current_role": years_in_current_role,
        "avg_performance_rating_3yr": avg_performance_rating_3yr,
        "kpi_met_percentage": kpi_met_percentage,
        "training_score": training_score,
        "num_projects_completed": num_projects_completed,
        "awards_won": awards_won,
        "manager_rating": manager_rating,
        "years_since_last_promotion": years_since_last_promotion,
        "peer_review_score": peer_review_score,
        "promoted": promoted,
    })
    return df


if __name__ == "__main__":
    df = generate_promotion_data()
    out_path = "/Volumes/ml_dev/attrition_promotion/uploads/promotion_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Promotion rate: {df['promoted'].mean():.3f}")
    print(df.head())