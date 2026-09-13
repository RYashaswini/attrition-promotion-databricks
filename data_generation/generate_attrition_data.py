"""
Generates synthetic employee attrition data from scratch.
Target: Attrition (1 = left the company, 0 = stayed)

Signal is built in deliberately (not pure noise) so the trained model has
something real to learn: low satisfaction + overtime + long commute + no
recent promotion + low pay hike all push attrition probability up.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 4000

DEPARTMENTS = ["Sales", "Engineering", "HR", "Finance", "Operations", "Support"]
JOB_LEVELS = [1, 2, 3, 4, 5]


def generate_attrition_data(n_rows: int = N_ROWS) -> pd.DataFrame:
    age = RNG.integers(21, 60, n_rows)
    department = RNG.choice(DEPARTMENTS, n_rows)
    job_level = RNG.choice(JOB_LEVELS, n_rows, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    monthly_income = (job_level * 15000 + RNG.normal(0, 4000, n_rows)).clip(15000, None)
    years_at_company = RNG.integers(0, 25, n_rows)
    distance_from_home_km = RNG.integers(1, 60, n_rows)
    job_satisfaction = RNG.integers(1, 5, n_rows)          # 1 (low) - 4 (high)
    work_life_balance = RNG.integers(1, 5, n_rows)         # 1 (bad) - 4 (great)
    environment_satisfaction = RNG.integers(1, 5, n_rows)
    overtime = RNG.choice(["Yes", "No"], n_rows, p=[0.3, 0.7])
    num_companies_worked = RNG.integers(0, 8, n_rows)
    training_times_last_year = RNG.integers(0, 6, n_rows)
    percent_salary_hike = RNG.integers(5, 25, n_rows)
    years_since_last_promotion = RNG.integers(0, 15, n_rows)
    years_with_curr_manager = RNG.integers(0, 17, n_rows)
    performance_rating = RNG.choice([1, 2, 3, 4], n_rows, p=[0.05, 0.15, 0.55, 0.25])

    # --- build a logistic-style score so labels have real signal, not noise ---
    score = (
        -0.35 * job_satisfaction
        - 0.30 * work_life_balance
        - 0.25 * environment_satisfaction
        + 0.05 * distance_from_home_km
        + 0.9 * (overtime == "Yes").astype(int)
        + 0.15 * years_since_last_promotion
        - 0.08 * percent_salary_hike
        + 0.12 * num_companies_worked
        - 0.04 * years_at_company
        - 0.00003 * monthly_income
        + RNG.normal(0, 1.1, n_rows)  # irreducible noise
    )
    prob_attrition = 1 / (1 + np.exp(-score))
    attrition = (RNG.uniform(0, 1, n_rows) < prob_attrition).astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(n_rows)],
        "age": age,
        "department": department,
        "job_level": job_level,
        "monthly_income": monthly_income.round(2),
        "years_at_company": years_at_company,
        "distance_from_home_km": distance_from_home_km,
        "job_satisfaction": job_satisfaction,
        "work_life_balance": work_life_balance,
        "environment_satisfaction": environment_satisfaction,
        "overtime": overtime,
        "num_companies_worked": num_companies_worked,
        "training_times_last_year": training_times_last_year,
        "percent_salary_hike": percent_salary_hike,
        "years_since_last_promotion": years_since_last_promotion,
        "years_with_curr_manager": years_with_curr_manager,
        "performance_rating": performance_rating,
        "attrition": attrition,
    })
    return df


if __name__ == "__main__":
    df = generate_attrition_data()
    out_path = "/Volumes/ml_dev/attrition_promotion/uploads/attrition_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Attrition rate: {df['attrition'].mean():.3f}")
    print(df.head())