# Images-generations Microservice

The Images-generations microservice generates an image based on a descriptive text prompt. This service utilizes a Stable Diffusion (SD) model to perform the image generation task. It takes a text as input and produces a image as output.

## Table of contents

1.  [Architecture](#architecture)
2.  [Deployment Options](#deployment-options)
3.  [Validated Configurations](#validated-configurations)

## Architecture

The Images-generations service is a single microservice that exposes an API endpoint. It receives a request containing a text prompt, processes it using the Stable Diffusion model, and returns the generated image.

- **Images-generations Server**: This microservice is the core engine for the image generation task. It can be deployed on both CPU and HPU.

## Deployment Options

For detailed, step-by-step instructions on how to deploy the Images-generations microservice using Docker Compose on different Intel platforms, please refer to the deployment guide. The guide contains all necessary steps, including building images, configuring the environment, and running the service.

| Platform          | Deployment Method | Link                                                       |
| ----------------- | ----------------- | ---------------------------------------------------------- |
| Intel Xeon/Gaudi2 | Docker Compose    | [Deployment Guide](../deployment/docker_compose/README.md) |

## Validated Configurations

The following configurations have been validated for the Images-generations microservice.

| **Deploy Method** | **Core Models**  | **Platform**      |
| ----------------- | ---------------- | ----------------- |
| Docker Compose    | Stable Diffusion | Intel Xeon/Gaudi2 |
