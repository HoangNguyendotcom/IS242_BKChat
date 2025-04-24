# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy app source code
COPY . .

# Build the application (if this is a production build)
# RUN npm run build

# Production stage
FROM node:18-alpine

# Create and use non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app

# Copy from build stage
COPY --from=builder --chown=appuser:appgroup /app .

# Expose port
ARG PORT=3000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Set NODE_ENV
ENV NODE_ENV=development

# Switch to non-root user
USER appuser

# Start the application with npm run dev
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]