from typing import Literal

from pydantic import BaseModel


class AttritionRequest(BaseModel):
    age: int
    department: str
    job_level: int
    monthly_income: float
    years_at_company: int
    distance_from_home_km: int
    job_satisfaction: int
    work_life_balance: int
    environment_satisfaction: int
    overtime: Literal["Yes", "No"]
    num_companies_worked: int
    training_times_last_year: int
    percent_salary_hike: int
    years_since_last_promotion: int
    years_with_curr_manager: int
    performance_rating: int


class PromotionRequest(BaseModel):
    age: int
    department: str
    education_level: Literal["Bachelors", "Masters", "PhD", "Diploma"]
    years_in_current_role: int
    avg_performance_rating_3yr: float
    kpi_met_percentage: int
    training_score: int
    num_projects_completed: int
    awards_won: int
    manager_rating: int
    years_since_last_promotion: int
    peer_review_score: float


class PredictionResponse(BaseModel):
    prediction: int
    model: str


class BatchTriggerResponse(BaseModel):
    run_id: int
    job: str
    status: str = "SUBMITTED"


class BatchStatusResponse(BaseModel):
    run_id: int
    life_cycle_state: str
    result_state: str | None = None
    output_path: str | None = None