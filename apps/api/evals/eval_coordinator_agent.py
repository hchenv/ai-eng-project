from api.agents.agents import coordinator_agent
from api.agents.graph import State, CoordinatorAgentProperties


from time import sleep
from langsmith import Client

ls_client = Client()

SLEEP_TIME = 5
ACC_THRESHOLD = 0.65


def evaluate_coordinator_delegation(run, example):

    final_answer_match = run.outputs["coordinator_agent"]["final_answer"] == example.outputs["coordinator_agent"]["final_answer"]
    next_agent_match = run.outputs["coordinator_agent"]["next_agent"] == example.outputs["coordinator_agent"]["next_agent"]

    return final_answer_match and next_agent_match


results = ls_client.evaluate(
    lambda x: coordinator_agent(
        State(
            messages=x["input"]["messages"],
            coordinator_agent=CoordinatorAgentProperties(
                iteration=0,
                final_answer=False,
                plan=[],
                next_agent=""
            ),
            answer=""
        )
    ),
    data="coordinator-first-delegation-evaluation",
    evaluators=[
        evaluate_coordinator_delegation
    ],
    experiment_prefix="coordinator-delegation",
    max_concurrency=4,
    num_repetitions=1
)


feedback_stats_exist = False
while not feedback_stats_exist:

    print("Waiting for feedback stats to exist...")
    sleep(SLEEP_TIME)

    results_resp = ls_client.read_project(
        project_name=results.experiment_name,
        include_stats=True
    )

    feedback_stats = results_resp.feedback_stats or {}
    feedback_stats_exist = feedback_stats.get("evaluate_coordinator_delegation") is not None


avg_metric = feedback_stats.get("evaluate_coordinator_delegation").get("avg")
errors = feedback_stats.get("evaluate_coordinator_delegation").get("errors")

if avg_metric >= ACC_THRESHOLD:
    output_message = f"✅ Success: {avg_metric}"
else:
    output_message = f"❌ Failure: {avg_metric}"


if errors > 0:
    raise AssertionError(f"Evaluation failed with {errors} errors")
elif avg_metric >= ACC_THRESHOLD:
    print(output_message, flush=True)
else:
    raise AssertionError(output_message)