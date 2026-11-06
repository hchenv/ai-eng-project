import os

import boto3
from botocore.exceptions import ClientError


AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
SECRET_PREFIX = "shopping-assistant"


def get_secret_reference(
    secrets_client, environment_name: str, secret_name: str
) -> dict[str, str]:
    secret_id = f"{SECRET_PREFIX}/{secret_name}"
    secret = secrets_client.describe_secret(SecretId=secret_id)
    return {"name": environment_name, "valueFrom": secret["ARN"]}


def create_task_definition():
    commit_sha = os.environ.get("GITHUB_SHA", "unknown")[:8]
    aws_account_id = os.environ.get("AWS_ACCOUNT_ID")
    if not aws_account_id:
        raise RuntimeError("AWS_ACCOUNT_ID is required")

    secrets_client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret_names = {
        "OPENAI_API_KEY": "openai-api-key",
        "LANGSMITH_API_KEY": "langsmith-api-key",
        "GROQ_API_KEY": "groq-api-key",
        "QDRANT_URL": "qdrant-url",
        "QDRANT_API_KEY": "qdrant-api-key",
        "LANGGRAPH_DB_URL": "langgraph-db-url",
        "TOOLS_DB_URL": "tools-db-url",
        "OPENROUTER_API_KEY": "openrouter-api-key",
        "CO_API_KEY": "co-api-key",
    }
    secrets = [
        get_secret_reference(secrets_client, environment_name, secret_name)
        for environment_name, secret_name in secret_names.items()
    ]

    task_definition = {
        "family": "amazon-be",
        "containerDefinitions": [
            {
                "name": "api",
                "image": (
                    f"{aws_account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/"
                    f"amazon-be:{commit_sha}"
                ),
                "cpu": 0,
                "portMappings": [
                    {
                        "name": "api-8000-tcp",
                        "containerPort": 8000,
                        "hostPort": 8000,
                        "protocol": "tcp",
                        "appProtocol": "http",
                    }
                ],
                "essential": True,
                "environment": [
                    {"name": "LANGSMITH_TRACING", "value": "true"},
                    {
                        "name": "LANGSMITH_ENDPOINT",
                        "value": "https://api.smith.langchain.com",
                    },
                    {"name": "LANGSMITH_PROJECT", "value": "rag-tracing"},
                ],
                "mountPoints": [],
                "volumesFrom": [],
                "secrets": secrets,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/amazon-be",
                        "mode": "non-blocking",
                        "awslogs-create-group": "true",
                        "max-buffer-size": "25m",
                        "awslogs-region": AWS_REGION,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
                "systemControls": [],
            }
        ],
        "taskRoleArn": (
            f"arn:aws:iam::{aws_account_id}:role/amazon-be-task-role"
        ),
        "executionRoleArn": (
            f"arn:aws:iam::{aws_account_id}:role/ecsTaskExecutionRole"
        ),
        "networkMode": "awsvpc",
        "volumes": [],
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "2048",
        "runtimePlatform": {
            "cpuArchitecture": "X86_64",
            "operatingSystemFamily": "LINUX",
        },
        "enableFaultInjection": False,
        "tags": [
            {"key": "CommitSHA", "value": commit_sha},
            {"key": "ManagedBy", "value": "GitHub-Actions"},
        ],
    }

    ecs_client = boto3.client("ecs", region_name=AWS_REGION)
    response = ecs_client.register_task_definition(**task_definition)

    print("Task definition registered successfully")
    print(f"Task Definition ARN: {response['taskDefinition']['taskDefinitionArn']}")
    print(f"Revision: {response['taskDefinition']['revision']}")
    print(f"Commit SHA: {commit_sha}")
    return response


if __name__ == "__main__":
    try:
        print("Creating ECS backend task definition...")
        create_task_definition()
    except ClientError as error:
        raise SystemExit(f"AWS error registering task definition: {error}") from error
    except Exception as error:
        raise SystemExit(f"Error registering task definition: {error}") from error
