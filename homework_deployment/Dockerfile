FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies via pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the script and model into the container
COPY scoring.py .

# Install necessary dependencies
RUN pip install pandas pyarrow

# Command to run the script
CMD ["pipenv", "run", "python", "scoring.py"]