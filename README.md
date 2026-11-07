# AI shopping assistant

Production deployment of a Streamlit shopping assistant backed by FastAPI,
Qdrant, PostgreSQL, and LLM APIs.

## Live deployment

Frontend:
<http://shopping-assistant-fe-alb-1860229733.ap-southeast-1.elb.amazonaws.com/>

> The current load balancer endpoint uses HTTP. HTTPS is a recommended next
> step before treating this as a public production service.

## Production architecture

The application runs in AWS `ap-southeast-1`:

- An internet-facing Application Load Balancer sends traffic to the Streamlit
  frontend on port 8501.
- The frontend and backend run as separate ECS services on AWS Fargate.
- Both services use private subnets and do not receive public IP addresses.
- ECS Service Connect routes frontend requests from `api:8000` to the FastAPI
  backend.
- A NAT Gateway provides outbound access to OpenAI, Qdrant, Supabase, and other
  external services.
- AWS Secrets Manager injects backend credentials when ECS starts a task.
- CloudWatch Logs receives application and Service Connect logs.

For complete diagrams, networking details, IAM role explanations, and current
limitations, see the
[AWS deployment architecture](documentation/aws-architecture.md).

## CI/CD

Pushes to `main` run `.github/workflows/build.yaml`, which:

1. Authenticates to AWS through GitHub OIDC.
2. Builds the frontend and backend container images.
3. Pushes commit, release, and `latest` image tags to Amazon ECR.
4. Registers release-tagged ECS task-definition revisions.

Releases use `release-YYYYMMDD-<short-sha>`, for example
`release-20261107-a1b2c3d4`. To deploy one, run **Deploy ECS release** from the
GitHub Actions page, enter its release name, and choose the backend, frontend,
or both services. The workflow updates the selected ECS services and waits for
them to become stable.

## Deployed services

- ECS cluster: `shopping-assistant-cluster`
- Frontend service: `amazon-fe-service`
- Backend service: `amazon-be-service`
- Frontend task definition: `amazon-fe`
- Backend task definition: `amazon-be`
- Service discovery namespace: `amazon.local`

## Bootcamp and local development

The original bootcamp prerequisites, YouTube videos, local environment setup,
and Docker Compose instructions are preserved in
[Bootcamp prerequisites and local development](documentation/ai-engineering-bootcamp-prerequisites.md).