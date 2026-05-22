"""Canned example: runs the text-to-sql planner CLI on a simple university schema.

Usage:
    uv run python example.py
"""

import subprocess
import sys

SCHEMA = """\
CREATE TABLE Employees (
    emp_id          INT PRIMARY KEY,
    first_name      VARCHAR(50) NOT NULL,
    last_name       VARCHAR(50) NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL,
    phone           VARCHAR(20),
    hire_date       DATE NOT NULL,
    termination_date DATE,
    date_of_birth   DATE,
    gender          VARCHAR(10),
    address         VARCHAR(200),
    city            VARCHAR(50),
    state           VARCHAR(50),
    zip_code        VARCHAR(10),
    dept_id         INT REFERENCES Departments(dept_id),
    job_id          INT REFERENCES Jobs(job_id),
    manager_id      INT REFERENCES Employees(emp_id),
    location_id     INT REFERENCES Locations(location_id),
    status          VARCHAR(20) DEFAULT 'Active'  -- Active, On Leave, Terminated
);

CREATE TABLE Departments (
    dept_id         INT PRIMARY KEY,
    dept_name       VARCHAR(100) NOT NULL,
    dept_head_id    INT REFERENCES Employees(emp_id),
    parent_dept_id  INT REFERENCES Departments(dept_id),
    cost_center     VARCHAR(20)
);

CREATE TABLE Jobs (
    job_id          INT PRIMARY KEY,
    job_title       VARCHAR(100) NOT NULL,
    job_level       INT,
    min_salary      DECIMAL(10,2),
    max_salary      DECIMAL(10,2),
    job_family      VARCHAR(50)
);

CREATE TABLE Compensation (
    comp_id         INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    effective_date  DATE NOT NULL,
    base_salary     DECIMAL(10,2),
    currency        VARCHAR(3) DEFAULT 'USD',
    pay_frequency   VARCHAR(20),  -- Annual, Monthly, Biweekly
    bonus_target    DECIMAL(5,2), -- percentage
    equity_shares   INT
);

CREATE TABLE Benefits_Plans (
    plan_id         INT PRIMARY KEY,
    plan_name       VARCHAR(100),
    plan_type       VARCHAR(50),  -- Medical, Dental, Vision, 401k, Life
    provider        VARCHAR(100),
    annual_cost     DECIMAL(10,2)
);

CREATE TABLE Benefits_Enrollment (
    enrollment_id   INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    plan_id         INT REFERENCES Benefits_Plans(plan_id),
    start_date      DATE NOT NULL,
    end_date        DATE,
    coverage_level  VARCHAR(20)   -- Employee, Employee+Spouse, Family
);

CREATE TABLE Leave_Balances (
    balance_id      INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    leave_type      VARCHAR(30),  -- PTO, Sick, Parental, Bereavement
    year            INT,
    accrued_hours   DECIMAL(6,2),
    used_hours      DECIMAL(6,2),
    carried_over    DECIMAL(6,2)
);

CREATE TABLE Leave_Requests (
    request_id      INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    leave_type      VARCHAR(30),
    start_date      DATE,
    end_date        DATE,
    hours_requested DECIMAL(6,2),
    status          VARCHAR(20),  -- Pending, Approved, Denied, Cancelled
    approved_by     INT REFERENCES Employees(emp_id),
    request_date    DATE
);

CREATE TABLE Performance_Reviews (
    review_id       INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    reviewer_id     INT REFERENCES Employees(emp_id),
    review_period   VARCHAR(20),  -- 2025-H1, 2025-Annual
    review_date     DATE,
    overall_rating  INT CHECK (overall_rating BETWEEN 1 AND 5),
    comments        TEXT,
    goals_met       DECIMAL(5,2)  -- percentage
);

CREATE TABLE Training_Courses (
    course_id       INT PRIMARY KEY,
    course_name     VARCHAR(200),
    category        VARCHAR(50),  -- Compliance, Technical, Leadership
    duration_hours  DECIMAL(5,2),
    mandatory       BOOLEAN DEFAULT FALSE
);

CREATE TABLE Training_Enrollment (
    enrollment_id   INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    course_id       INT REFERENCES Training_Courses(course_id),
    enrollment_date DATE,
    completion_date DATE,
    score           DECIMAL(5,2),
    status          VARCHAR(20)   -- Enrolled, In Progress, Completed, Expired
);

CREATE TABLE Locations (
    location_id     INT PRIMARY KEY,
    location_name   VARCHAR(100),
    address         VARCHAR(200),
    city            VARCHAR(50),
    state           VARCHAR(50),
    country         VARCHAR(50),
    zip_code        VARCHAR(10),
    capacity        INT
);

CREATE TABLE Emergency_Contacts (
    contact_id      INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    contact_name    VARCHAR(100),
    relationship    VARCHAR(30),
    phone_primary   VARCHAR(20),
    phone_secondary VARCHAR(20)
);

CREATE TABLE Job_History (
    history_id      INT PRIMARY KEY,
    emp_id          INT REFERENCES Employees(emp_id),
    job_id          INT REFERENCES Jobs(job_id),
    dept_id         INT REFERENCES Departments(dept_id),
    start_date      DATE,
    end_date        DATE,
    change_reason   VARCHAR(50)   -- Promotion, Transfer, Reorganization
);
"""

QUESTION = "How many employees work at location 'San Francisco'?"


def main():
    print("# Text-to-SQL Planner\n")
    print(f"**Question:** {QUESTION}\n")
    print(f"### Schema\n")
    print(f"```sql\n{SCHEMA.strip()}\n```\n")
    print(f"---\n", flush=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "text_to_sql_planner",
            "-q", QUESTION,
            "-s", SCHEMA,
            "-v",
        ],
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
