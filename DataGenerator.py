"""
HR Dataset Generator (8,950 records)

Creates a realistic synthetic HR dataset with:
- Employee ID
- First Name / Last Name
- Gender (Female 46%, Male 54%)
- State / City (predefined mapping)
- Hire Date (custom probabilities by year 2015–2024)
- Department (custom probabilities)
- Job Title (conditional on department with probabilities)
- Education Level (mapped from job title)
- Performance Rating (custom probabilities)
- Overtime (Yes 30%, No 70%)
- Salary (range by department + job title)
- Birth Date (age-group distribution + job-title minimum age; consistent with hire date)
- Termination Date (11.2% terminated; year probabilities 2015–2024; >= 6 months after hire)
- Adjusted Salary (computed from gender, education, age)

Outputs: CSV file (default: hr_dataset.csv)

Notes:
- This is synthetic data for demos/testing only.
- You can tweak distributions and ranges in the CONFIG section.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional


# =========================
# CONFIG
# =========================

N_RECORDS = 8950
OUTPUT_CSV = "hr_dataset.csv"
SEED = 42  # set to None for non-deterministic runs

GENDER_CHOICES = ["Female", "Male"]
GENDER_WEIGHTS = [0.46, 0.54]

OVERTIME_CHOICES = ["Yes", "No"]
OVERTIME_WEIGHTS = [0.30, 0.70]

PERFORMANCE_CHOICES = ["Excellent", "Good", "Satisfactory", "Needs Improvement"]
PERFORMANCE_WEIGHTS = [0.18, 0.52, 0.23, 0.07]

# Hire year probabilities (must sum to 1.0)
HIRE_YEAR_WEIGHTS = {
    2015: 0.06,
    2016: 0.07,
    2017: 0.08,
    2018: 0.10,
    2019: 0.12,
    2020: 0.09,
    2021: 0.12,
    2022: 0.13,
    2023: 0.13,
    2024: 0.10,
}

# Department probabilities (must sum to 1.0)
DEPARTMENT_WEIGHTS = {
    "Engineering": 0.19,
    "Sales": 0.15,
    "Marketing": 0.08,
    "HR": 0.06,
    "Finance": 0.07,
    "Operations": 0.12,
    "Customer Support": 0.10,
    "IT": 0.08,
    "Legal": 0.03,
    "Product": 0.07,
    "Data": 0.05,
}

# Job titles conditional on department with weights (each dept sums to 1.0)
JOB_TITLES_BY_DEPT: Dict[str, Dict[str, float]] = {
    "Engineering": {
        "Software Engineer I": 0.22,
        "Software Engineer II": 0.25,
        "Senior Software Engineer": 0.18,
        "Engineering Manager": 0.08,
        "QA Engineer": 0.12,
        "DevOps Engineer": 0.10,
        "Architect": 0.05,
    },
    "Sales": {
        "Sales Representative": 0.45,
        "Account Executive": 0.25,
        "Sales Manager": 0.10,
        "Sales Operations Analyst": 0.08,
        "Account Manager": 0.12,
    },
    "Marketing": {
        "Marketing Specialist": 0.38,
        "Content Strategist": 0.18,
        "SEO Analyst": 0.14,
        "Marketing Manager": 0.12,
        "Brand Manager": 0.10,
        "Graphic Designer": 0.08,
    },
    "HR": {
        "HR Coordinator": 0.40,
        "HR Generalist": 0.30,
        "Talent Acquisition Specialist": 0.18,
        "HR Manager": 0.12,
    },
    "Finance": {
        "Financial Analyst": 0.45,
        "Accountant": 0.30,
        "Senior Financial Analyst": 0.12,
        "Finance Manager": 0.08,
        "Payroll Specialist": 0.05,
    },
    "Operations": {
        "Operations Associate": 0.42,
        "Operations Manager": 0.14,
        "Supply Chain Analyst": 0.14,
        "Logistics Coordinator": 0.18,
        "Procurement Specialist": 0.12,
    },
    "Customer Support": {
        "Customer Support Agent": 0.60,
        "Senior Support Agent": 0.18,
        "Support Team Lead": 0.10,
        "Customer Success Manager": 0.12,
    },
    "IT": {
        "IT Support Specialist": 0.42,
        "Systems Administrator": 0.22,
        "Network Engineer": 0.18,
        "IT Manager": 0.10,
        "Security Analyst": 0.08,
    },
    "Legal": {
        "Paralegal": 0.45,
        "Legal Counsel": 0.35,
        "Compliance Specialist": 0.20,
    },
    "Product": {
        "Product Analyst": 0.22,
        "Product Manager": 0.34,
        "Senior Product Manager": 0.18,
        "UX Designer": 0.14,
        "UX Researcher": 0.12,
    },
    "Data": {
        "Data Analyst": 0.45,
        "Senior Data Analyst": 0.18,
        "Data Engineer": 0.22,
        "Data Scientist": 0.15,
    },
}

# Education mapping by job title
EDUCATION_BY_JOB: Dict[str, List[Tuple[str, float]]] = {
    # Tech
    "Software Engineer I": [("Bachelor", 0.75), ("Master", 0.20), ("PhD", 0.05)],
    "Software Engineer II": [("Bachelor", 0.65), ("Master", 0.28), ("PhD", 0.07)],
    "Senior Software Engineer": [("Bachelor", 0.55), ("Master", 0.35), ("PhD", 0.10)],
    "Engineering Manager": [("Bachelor", 0.50), ("Master", 0.40), ("PhD", 0.10)],
    "QA Engineer": [("Associate", 0.10), ("Bachelor", 0.75), ("Master", 0.15)],
    "DevOps Engineer": [("Bachelor", 0.60), ("Master", 0.30), ("PhD", 0.10)],
    "Architect": [("Bachelor", 0.45), ("Master", 0.45), ("PhD", 0.10)],

    # Sales/Marketing
    "Sales Representative": [("High School", 0.25), ("Associate", 0.25), ("Bachelor", 0.45), ("Master", 0.05)],
    "Account Executive": [("Associate", 0.15), ("Bachelor", 0.70), ("Master", 0.15)],
    "Sales Manager": [("Bachelor", 0.70), ("Master", 0.30)],
    "Sales Operations Analyst": [("Bachelor", 0.80), ("Master", 0.20)],
    "Account Manager": [("Bachelor", 0.80), ("Master", 0.20)],

    "Marketing Specialist": [("Bachelor", 0.80), ("Master", 0.20)],
    "Content Strategist": [("Bachelor", 0.75), ("Master", 0.25)],
    "SEO Analyst": [("Bachelor", 0.80), ("Master", 0.20)],
    "Marketing Manager": [("Bachelor", 0.60), ("Master", 0.40)],
    "Brand Manager": [("Bachelor", 0.55), ("Master", 0.45)],
    "Graphic Designer": [("Associate", 0.35), ("Bachelor", 0.60), ("Master", 0.05)],

    # HR/Finance
    "HR Coordinator": [("Associate", 0.40), ("Bachelor", 0.55), ("Master", 0.05)],
    "HR Generalist": [("Bachelor", 0.80), ("Master", 0.20)],
    "Talent Acquisition Specialist": [("Bachelor", 0.75), ("Master", 0.25)],
    "HR Manager": [("Bachelor", 0.60), ("Master", 0.40)],

    "Financial Analyst": [("Bachelor", 0.80), ("Master", 0.20)],
    "Accountant": [("Bachelor", 0.85), ("Master", 0.15)],
    "Senior Financial Analyst": [("Bachelor", 0.60), ("Master", 0.35), ("PhD", 0.05)],
    "Finance Manager": [("Bachelor", 0.55), ("Master", 0.40), ("PhD", 0.05)],
    "Payroll Specialist": [("Associate", 0.35), ("Bachelor", 0.60), ("Master", 0.05)],

    # Ops/Support/IT
    "Operations Associate": [("High School", 0.20), ("Associate", 0.40), ("Bachelor", 0.38), ("Master", 0.02)],
    "Operations Manager": [("Bachelor", 0.70), ("Master", 0.30)],
    "Supply Chain Analyst": [("Bachelor", 0.80), ("Master", 0.20)],
    "Logistics Coordinator": [("Associate", 0.45), ("Bachelor", 0.50), ("Master", 0.05)],
    "Procurement Specialist": [("Bachelor", 0.80), ("Master", 0.20)],

    "Customer Support Agent": [("High School", 0.35), ("Associate", 0.35), ("Bachelor", 0.28), ("Master", 0.02)],
    "Senior Support Agent": [("Associate", 0.35), ("Bachelor", 0.60), ("Master", 0.05)],
    "Support Team Lead": [("Bachelor", 0.75), ("Master", 0.25)],
    "Customer Success Manager": [("Bachelor", 0.75), ("Master", 0.25)],

    "IT Support Specialist": [("Associate", 0.35), ("Bachelor", 0.60), ("Master", 0.05)],
    "Systems Administrator": [("Bachelor", 0.80), ("Master", 0.20)],
    "Network Engineer": [("Bachelor", 0.78), ("Master", 0.20), ("PhD", 0.02)],
    "IT Manager": [("Bachelor", 0.60), ("Master", 0.40)],
    "Security Analyst": [("Bachelor", 0.75), ("Master", 0.25)],

    # Legal/Product/Data
    "Paralegal": [("Associate", 0.55), ("Bachelor", 0.40), ("Master", 0.05)],
    "Legal Counsel": [("Master", 0.85), ("PhD", 0.15)],  # stands in for JD/LLM-like advanced education
    "Compliance Specialist": [("Bachelor", 0.75), ("Master", 0.25)],

    "Product Analyst": [("Bachelor", 0.80), ("Master", 0.20)],
    "Product Manager": [("Bachelor", 0.60), ("Master", 0.40)],
    "Senior Product Manager": [("Bachelor", 0.45), ("Master", 0.50), ("PhD", 0.05)],
    "UX Designer": [("Bachelor", 0.75), ("Master", 0.20), ("Associate", 0.05)],
    "UX Researcher": [("Bachelor", 0.60), ("Master", 0.35), ("PhD", 0.05)],

    "Data Analyst": [("Bachelor", 0.85), ("Master", 0.15)],
    "Senior Data Analyst": [("Bachelor", 0.60), ("Master", 0.35), ("PhD", 0.05)],
    "Data Engineer": [("Bachelor", 0.65), ("Master", 0.30), ("PhD", 0.05)],
    "Data Scientist": [("Bachelor", 0.45), ("Master", 0.40), ("PhD", 0.15)],
}

# Minimum age by job title (for realism)
MIN_AGE_BY_JOB: Dict[str, int] = {
    "Software Engineer I": 21,
    "Software Engineer II": 23,
    "Senior Software Engineer": 26,
    "Engineering Manager": 30,
    "Architect": 32,
    "DevOps Engineer": 24,
    "QA Engineer": 21,

    "Sales Representative": 18,
    "Account Executive": 22,
    "Sales Manager": 28,
    "Sales Operations Analyst": 21,
    "Account Manager": 24,

    "Marketing Specialist": 21,
    "Content Strategist": 22,
    "SEO Analyst": 21,
    "Marketing Manager": 27,
    "Brand Manager": 28,
    "Graphic Designer": 19,

    "HR Coordinator": 19,
    "HR Generalist": 22,
    "Talent Acquisition Specialist": 22,
    "HR Manager": 28,

    "Financial Analyst": 21,
    "Accountant": 22,
    "Senior Financial Analyst": 26,
    "Finance Manager": 30,
    "Payroll Specialist": 20,

    "Operations Associate": 18,
    "Operations Manager": 28,
    "Supply Chain Analyst": 21,
    "Logistics Coordinator": 19,
    "Procurement Specialist": 22,

    "Customer Support Agent": 18,
    "Senior Support Agent": 20,
    "Support Team Lead": 24,
    "Customer Success Manager": 24,

    "IT Support Specialist": 19,
    "Systems Administrator": 23,
    "Network Engineer": 23,
    "IT Manager": 30,
    "Security Analyst": 23,

    "Paralegal": 20,
    "Legal Counsel": 27,
    "Compliance Specialist": 22,

    "Product Analyst": 21,
    "Product Manager": 25,
    "Senior Product Manager": 30,
    "UX Designer": 21,
    "UX Researcher": 23,

    "Data Analyst": 21,
    "Senior Data Analyst": 25,
    "Data Engineer": 23,
    "Data Scientist": 24,
}

# Salary ranges by (department, job title): (min, max)
SALARY_RANGES: Dict[Tuple[str, str], Tuple[int, int]] = {}
def _add_salary(dept: str, title: str, lo: int, hi: int) -> None:
    SALARY_RANGES[(dept, title)] = (lo, hi)

# Engineering
_add_salary("Engineering", "Software Engineer I", 70000, 95000)
_add_salary("Engineering", "Software Engineer II", 90000, 125000)
_add_salary("Engineering", "Senior Software Engineer", 120000, 170000)
_add_salary("Engineering", "Engineering Manager", 145000, 205000)
_add_salary("Engineering", "QA Engineer", 65000, 98000)
_add_salary("Engineering", "DevOps Engineer", 105000, 155000)
_add_salary("Engineering", "Architect", 160000, 230000)

# Sales
_add_salary("Sales", "Sales Representative", 45000, 90000)  # base-ish (commission not modeled)
_add_salary("Sales", "Account Executive", 65000, 120000)
_add_salary("Sales", "Sales Manager", 90000, 160000)
_add_salary("Sales", "Sales Operations Analyst", 60000, 100000)
_add_salary("Sales", "Account Manager", 65000, 115000)

# Marketing
_add_salary("Marketing", "Marketing Specialist", 55000, 95000)
_add_salary("Marketing", "Content Strategist", 60000, 105000)
_add_salary("Marketing", "SEO Analyst", 55000, 95000)
_add_salary("Marketing", "Marketing Manager", 85000, 140000)
_add_salary("Marketing", "Brand Manager", 90000, 155000)
_add_salary("Marketing", "Graphic Designer", 48000, 90000)

# HR
_add_salary("HR", "HR Coordinator", 42000, 65000)
_add_salary("HR", "HR Generalist", 55000, 90000)
_add_salary("HR", "Talent Acquisition Specialist", 60000, 110000)
_add_salary("HR", "HR Manager", 85000, 140000)

# Finance
_add_salary("Finance", "Financial Analyst", 65000, 105000)
_add_salary("Finance", "Accountant", 60000, 100000)
_add_salary("Finance", "Senior Financial Analyst", 85000, 135000)
_add_salary("Finance", "Finance Manager", 105000, 175000)
_add_salary("Finance", "Payroll Specialist", 50000, 85000)

# Operations
_add_salary("Operations", "Operations Associate", 40000, 70000)
_add_salary("Operations", "Operations Manager", 85000, 145000)
_add_salary("Operations", "Supply Chain Analyst", 65000, 110000)
_add_salary("Operations", "Logistics Coordinator", 45000, 80000)
_add_salary("Operations", "Procurement Specialist", 60000, 105000)

# Customer Support
_add_salary("Customer Support", "Customer Support Agent", 38000, 60000)
_add_salary("Customer Support", "Senior Support Agent", 48000, 75000)
_add_salary("Customer Support", "Support Team Lead", 60000, 95000)
_add_salary("Customer Support", "Customer Success Manager", 70000, 125000)

# IT
_add_salary("IT", "IT Support Specialist", 45000, 78000)
_add_salary("IT", "Systems Administrator", 70000, 120000)
_add_salary("IT", "Network Engineer", 75000, 130000)
_add_salary("IT", "IT Manager", 95000, 165000)
_add_salary("IT", "Security Analyst", 80000, 140000)

# Legal
_add_salary("Legal", "Paralegal", 50000, 90000)
_add_salary("Legal", "Legal Counsel", 120000, 220000)
_add_salary("Legal", "Compliance Specialist", 65000, 115000)

# Product
_add_salary("Product", "Product Analyst", 70000, 115000)
_add_salary("Product", "Product Manager", 105000, 170000)
_add_salary("Product", "Senior Product Manager", 140000, 220000)
_add_salary("Product", "UX Designer", 80000, 140000)
_add_salary("Product", "UX Researcher", 85000, 150000)

# Data
_add_salary("Data", "Data Analyst", 65000, 110000)
_add_salary("Data", "Senior Data Analyst", 90000, 145000)
_add_salary("Data", "Data Engineer", 100000, 165000)
_add_salary("Data", "Data Scientist", 110000, 185000)

# Age group distribution for birth dates (must sum to 1.0)
AGE_GROUPS: List[Tuple[Tuple[int, int], float]] = [
    ((18, 24), 0.15),
    ((25, 34), 0.35),
    ((35, 44), 0.25),
    ((45, 54), 0.15),
    ((55, 64), 0.09),
    ((65, 70), 0.01),
]

# Termination: 11.2% of employees
TERMINATION_RATE = 0.112

# Termination year probabilities (must sum to 1.0)
TERM_YEAR_WEIGHTS = {
    2015: 0.05,
    2016: 0.06,
    2017: 0.07,
    2018: 0.09,
    2019: 0.11,
    2020: 0.14,
    2021: 0.12,
    2022: 0.13,
    2023: 0.13,
    2024: 0.10,
}

# Adjusted Salary rules
GENDER_MULTIPLIER = {"Female": 0.985, "Male": 1.000}  # example pay-gap-like adjustment (synthetic)
EDU_INCREMENT = {
    "High School": 0.00,
    "Associate": 0.03,
    "Bachelor": 0.07,
    "Master": 0.12,
    "PhD": 0.18,
}
# Age increment: +0.2% per year over 30, capped at +6%
AGE_BASE = 30
AGE_STEP = 0.002
AGE_CAP = 0.06


# State/City mapping (US example)
STATE_CITIES: Dict[str, List[str]] = {
    "California": ["Los Angeles", "San Diego", "San Jose", "San Francisco", "Sacramento"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
    "New York": ["New York City", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale"],
    "Illinois": ["Chicago", "Aurora", "Naperville", "Joliet", "Springfield"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Harrisburg"],
    "Ohio": ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron"],
    "Georgia": ["Atlanta", "Augusta", "Savannah", "Athens", "Macon"],
    "North Carolina": ["Charlotte", "Raleigh", "Greensboro", "Durham", "Winston-Salem"],
    "Washington": ["Seattle", "Spokane", "Tacoma", "Vancouver", "Bellevue"],
}

# Simple name pools (no external dependencies)
FIRST_NAMES = [
    "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Amelia", "Harper", "Evelyn", "Abigail",
    "Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Alexander",
    "Charlotte", "Avery", "Ella", "Scarlett", "Grace", "Chloe", "Lily", "Hannah", "Zoey", "Riley",
    "Ethan", "Mason", "Logan", "Jackson", "Levi", "Sebastian", "Mateo", "Jack", "Owen", "Daniel",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


# =========================
# UTILITIES
# =========================

def seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)

def weighted_choice(items: List[str], weights: List[float]) -> str:
    # random.choices returns a list
    return random.choices(items, weights=weights, k=1)[0]

def weighted_choice_from_dict(d: Dict[str, float]) -> str:
    items = list(d.keys())
    weights = list(d.values())
    return weighted_choice(items, weights)

def random_date_in_year(year: int) -> date:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def years_between(d1: date, d2: date) -> int:
    # full years between dates (d2 assumed >= d1)
    years = d2.year - d1.year
    if (d2.month, d2.day) < (d1.month, d1.day):
        years -= 1
    return years

def add_months(d: date, months: int) -> date:
    # safe-ish month add without external libs
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # choose min day in target month
    # days in month:
    if m in (1, 3, 5, 7, 8, 10, 12):
        max_day = 31
    elif m in (4, 6, 9, 11):
        max_day = 30
    else:
        # Feb
        leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        max_day = 29 if leap else 28
    day = min(d.day, max_day)
    return date(y, m, day)


# =========================
# GENERATION LOGIC
# =========================

def generate_employee_id(i: int) -> str:
    # Example format: EMP-000001
    return f"EMP-{i:06d}"

def generate_name() -> Tuple[str, str]:
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)

def generate_gender() -> str:
    return weighted_choice(GENDER_CHOICES, GENDER_WEIGHTS)

def generate_state_city() -> Tuple[str, str]:
    state = random.choice(list(STATE_CITIES.keys()))
    city = random.choice(STATE_CITIES[state])
    return state, city

def generate_hire_date() -> date:
    year = weighted_choice_from_dict(HIRE_YEAR_WEIGHTS)
    return random_date_in_year(year)

def generate_department() -> str:
    return weighted_choice_from_dict(DEPARTMENT_WEIGHTS)

def generate_job_title(department: str) -> str:
    return weighted_choice_from_dict(JOB_TITLES_BY_DEPT[department])

def generate_education_level(job_title: str) -> str:
    options = EDUCATION_BY_JOB.get(job_title)
    if not options:
        # fallback
        return "Bachelor"
    levels = [lvl for (lvl, w) in options]
    weights = [w for (lvl, w) in options]
    return weighted_choice(levels, weights)

def generate_performance_rating() -> str:
    return weighted_choice(PERFORMANCE_CHOICES, PERFORMANCE_WEIGHTS)

def generate_overtime() -> str:
    return weighted_choice(OVERTIME_CHOICES, OVERTIME_WEIGHTS)

def generate_salary(department: str, job_title: str) -> int:
    lo, hi = SALARY_RANGES[(department, job_title)]
    # Use a triangular distribution to concentrate toward mid-high for senior roles
    mid = (lo + hi) / 2
    val = random.triangular(lo, hi, mid + (hi - mid) * 0.15)
    # Round to nearest 100
    return int(round(val / 100.0) * 100)

def pick_age_for_job(job_title: str) -> int:
    min_age = MIN_AGE_BY_JOB.get(job_title, 18)

    # Pick an age group then sample uniformly within it, respecting min age
    for _ in range(50):
        (a_lo, a_hi), w = random.choices(AGE_GROUPS, weights=[x[1] for x in AGE_GROUPS], k=1)[0]
        a_lo2 = max(a_lo, min_age)
        if a_lo2 <= a_hi:
            return random.randint(a_lo2, a_hi)

    # fallback if something goes weird
    return max(min_age, 25)

def generate_birth_date(hire_date: date, job_title: str) -> date:
    """
    Ensures:
    - Age respects job-title minimums
    - Person is at least (min_age) at hire date
    """
    min_age = MIN_AGE_BY_JOB.get(job_title, 18)

    # Choose an age at hire, then back-calculate birth year
    age_at_hire = pick_age_for_job(job_title)
    age_at_hire = max(age_at_hire, min_age)

    # Pick a birthday within the year so that the computed age_at_hire is consistent
    # We'll pick a birthdate in a range that yields the desired age_at_hire.
    # Approx: birth_year = hire_year - age_at_hire (then adjust day/month logic)
    target_birth_year = hire_date.year - age_at_hire

    # Pick a birthdate candidate; validate actual age at hire is >= min_age and close to chosen
    for _ in range(200):
        bd = random_date_in_year(target_birth_year)
        actual_age = years_between(bd, hire_date)
        if actual_age >= min_age and (actual_age == age_at_hire or abs(actual_age - age_at_hire) <= 1):
            return bd

    # fallback: force a safe date
    bd = date(target_birth_year, 6, 15)
    # if somehow too young (edge cases), push back one year
    while years_between(bd, hire_date) < min_age:
        bd = date(bd.year - 1, bd.month, bd.day)
    return bd

def should_terminate() -> bool:
    return random.random() < TERMINATION_RATE

def generate_termination_date(hire_date: date) -> Optional[date]:
    """
    If terminated:
    - Pick a termination year with probabilities 2015–2024
    - Termination date >= hire_date + 6 months
    - Termination date <= 2024-12-31
    If cannot find a valid date after multiple tries, return None (kept employed).
    """
    min_term = add_months(hire_date, 6)
    max_term = date(2024, 12, 31)
    if min_term > max_term:
        return None

    for _ in range(200):
        y = weighted_choice_from_dict(TERM_YEAR_WEIGHTS)
        td = random_date_in_year(y)
        if min_term <= td <= max_term:
            return td

    # Fallback: pick any date in [min_term, max_term]
    span = (max_term - min_term).days
    return min_term + timedelta(days=random.randint(0, span))

def compute_adjusted_salary(
    base_salary: int,
    gender: str,
    education: str,
    birth_date: date,
    as_of: date,
) -> int:
    """
    Adjusted Salary = base_salary * gender_multiplier * (1 + education_increment) * (1 + age_increment)
    - age_increment: +0.2% per year over 30, capped at +6%
    """
    age = years_between(birth_date, as_of)

    gender_mult = GENDER_MULTIPLIER.get(gender, 1.0)
    edu_inc = EDU_INCREMENT.get(education, 0.07)  # default bachelor-like

    age_over = max(0, age - AGE_BASE)
    age_inc = clamp(age_over * AGE_STEP, 0.0, AGE_CAP)

    adjusted = base_salary * gender_mult * (1.0 + edu_inc) * (1.0 + age_inc)

    # Round to nearest 100
    return int(round(adjusted / 100.0) * 100)


# =========================
# MAIN GENERATOR
# =========================

def generate_record(emp_num: int) -> Dict[str, object]:
    employee_id = generate_employee_id(emp_num)
    first_name, last_name = generate_name()
    gender = generate_gender()
    state, city = generate_state_city()

    department = generate_department()
    job_title = generate_job_title(department)

    hire_date = generate_hire_date()
    education = generate_education_level(job_title)

    performance = generate_performance_rating()
    overtime = generate_overtime()

    salary = generate_salary(department, job_title)
    birth_date = generate_birth_date(hire_date, job_title)

    terminated = should_terminate()
    termination_date = generate_termination_date(hire_date) if terminated else None
    # If we failed to generate a valid termination date, treat as not terminated
    if terminated and termination_date is None:
        terminated = False

    as_of = termination_date if terminated else date(2024, 12, 31)
    adjusted_salary = compute_adjusted_salary(
        base_salary=salary,
        gender=gender,
        education=education,
        birth_date=birth_date,
        as_of=as_of,
    )

    return {
        "Employee ID": employee_id,
        "First Name": first_name,
        "Last Name": last_name,
        "Gender": gender,
        "State": state,
        "City": city,
        "Hire Date": hire_date.isoformat(),
        "Department": department,
        "Job Title": job_title,
        "Education Level": education,
        "Performance Rating": performance,
        "Overtime": overtime,
        "Salary": salary,
        "Birth Date": birth_date.isoformat(),
        "Termination Date": termination_date.isoformat() if terminated else "",
        "Adjusted Salary": adjusted_salary,
    }

def generate_dataset(n: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for i in range(1, n + 1):
        records.append(generate_record(i))
    return records

def write_csv(path: str, records: List[Dict[str, object]]) -> None:
    if not records:
        raise ValueError("No records to write.")
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

def sanity_report(records: List[Dict[str, object]]) -> None:
    # Simple checks (prints to console)
    total = len(records)
    female = sum(1 for r in records if r["Gender"] == "Female")
    male = sum(1 for r in records if r["Gender"] == "Male")
    terminated = sum(1 for r in records if r["Termination Date"])

    print("=== Sanity Report ===")
    print(f"Total records: {total}")
    print(f"Female: {female} ({female/total:.3%})")
    print(f"Male: {male} ({male/total:.3%})")
    print(f"Terminated: {terminated} ({terminated/total:.3%})")

    # Check termination >= hire + 6 months
    bad_terms = 0
    for r in records:
        if r["Termination Date"]:
            hd = date.fromisoformat(r["Hire Date"])
            td = date.fromisoformat(r["Termination Date"])
            if td < add_months(hd, 6):
                bad_terms += 1
    print(f"Termination-date violations (<6 months after hire): {bad_terms}")

def main() -> None:
    seed_everything(SEED)

    records = generate_dataset(N_RECORDS)
    write_csv(OUTPUT_CSV, records)
    sanity_report(records)
    print(f"\nWrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()