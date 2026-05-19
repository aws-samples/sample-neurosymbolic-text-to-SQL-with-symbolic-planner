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
