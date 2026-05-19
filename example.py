"""Canned example: runs the text-to-sql planner CLI on a simple university schema.

Usage:
    uv run python example.py
"""

import subprocess
import sys

SCHEMA = """\
CREATE TABLE Students (
    s_id   INT PRIMARY KEY,
    name   VARCHAR(100)
);

CREATE TABLE Courses (
    c_id   INT PRIMARY KEY,
    c_name VARCHAR(100),
    c_type VARCHAR(50)
);

CREATE TABLE Enrolled (
    s_id   INT,
    c_id   INT,
    grade  INT,
    FOREIGN KEY (s_id) REFERENCES Students(s_id),
    FOREIGN KEY (c_id) REFERENCES Courses(c_id)
);
"""

QUESTION = "Which students have taken one or more Computer Science courses?"


def main():
    print("=" * 60)
    print("Text-to-SQL Planner — Canned Example")
    print("=" * 60)
    print()
    print(f"Question: {QUESTION}")
    print()
    print("Schema:")
    for line in SCHEMA.strip().splitlines():
        print(f"  {line}")
    print()
    print("-" * 60)
    print("Running CLI...")
    print(flush=True)

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
