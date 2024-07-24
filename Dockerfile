# Use the rootproject/root base image
FROM rootproject/root:6.26.10-conda

# Set the working directory
WORKDIR /usr/app

# Copy the current directory contents into the container
COPY . .

# Install dependencies and create a virtual environment
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the entry point
ENTRYPOINT ["python3", "analysis.py"]
