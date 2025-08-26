# FastAPI Stock Apple Model

This project provides a FastAPI application for serving a machine learning model that predicts Apple stock trends.

## Features

- REST API built with FastAPI
- Pre-trained model included (`models/apple.joblib`)
- Ready-to-use Docker image

## Getting Started

### Run with Docker

You can pull and run the image directly from Docker Hub:

```sh
docker pull dataopsfusion/stock-apple-v1:latest
docker run -p 8000:8000 dataopsfusion/stock-apple-v1
```

The API will be available at http://localhost:8000.

### Docker Hub
Find the image and more details here:
https://hub.docker.com/repository/docker/dataopsfusion/stock-apple-v1/general
You can further customize this as your project evolves.