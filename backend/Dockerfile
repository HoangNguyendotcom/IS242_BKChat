FROM python:3.11-slim-bullseye AS builder

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Multi-stage build for a smaller final image
FROM python:3.11-slim-bullseye

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY . .

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Expose port (using ARG to make it configurable)
ARG PORT=5000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Run as non-root user for better security
RUN useradd -m appuser
USER appuser

# Command to run the application
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} run:app"]