CREATE DATABASE patient_data;

\connect patient_data;

CREATE TABLE stroke(
    id INT,
    gender TEXT, 
    age FLOAT,
    hypertension INT, 
    heart_disease INT,
    ever_married TEXT,
    work_type TEXT,
    Residence_type TEXT,
    avg_glucose_level FLOAT,
    bmi FLOAT,
    smoking_status TEXT,
    stroke INT
);

\copy stroke FROM 'data/raw/train_2v.csv' DELIMITER ',' CSV HEADER;