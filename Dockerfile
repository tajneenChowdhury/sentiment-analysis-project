# Stage 1: Use an official Python image as a base image
FROM python:3.11-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy the application code and data into the container
# COPY <source_on_local_machine> <destination_in_container>
COPY src/predict.py src/predict.py
COPY models/sentiment.joblib models/sentiment.joblib

# Stage 5: Specify the command to run on container start
CMD ["python", "src/predict.py"]

