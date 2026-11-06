import os

import boto3
from botocore.exceptions import ClientError


AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")


def create_task_definition():
    commit_sha = os.environ.get("GITHUB_SHA", "unknown")[:8]
    aws_account_id = os.environ.get("AWS_ACCOUNT_ID")
    if not aws_account_id:
        raise RuntimeError("AWS_ACCOUNT_ID is required")

    api_url = os.environ.get("API_URL") or "http://api:8000"
    task_definition = {
        "family": "amazon-fe",
        "containerDefinitions": [
            {
                "name": "chatbot-ui",
                "image": (
                    f"{aws_account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/"
                    f"amazon-fe:{commit_sha}"
                ),
                "cpu": 0,
                "portMappings": [
                    {
                        "name": "chatbot-ui-8501-tcp",
                        "containerPort": 8501,
                        "hostPort": 8501,
                        "protocol": "tcp",
                        "appProtocol": "http",
                    }
                ],
                "essential": True,
                "environment": [{"name": "API_URL", "value": api_url}],
                "mountPoints": [],
                "volumesFrom": [],
                "secrets": [],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/amazon-fe",
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
            f"arn:aws:iam::{aws_account_id}:role/amazon-fe-task-role"
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
        print("Creating ECS frontend task definition...")
        create_task_definition()
    except ClientError as error:
        raise SystemExit(f"AWS error registering task definition: {error}") from error
    except Exception as error:
        raise SystemExit(f"Error registering task definition: {error}") from error
