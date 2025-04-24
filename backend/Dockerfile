FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (using ARG to make it configurable)
ARG PORT=5000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Command to run the application
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} run:app"]