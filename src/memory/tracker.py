import mlflow
import json
from config.settings import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

class ExperimentTracker:
    def __init__(self):
        self.run = None
        self.run_id = None

    def start_run(self, question: str, hypothesis: str):
        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id
        mlflow.log_param("question", question[:250])
        mlflow.log_param("initial_hypothesis", hypothesis[:250])
        print(f"MLflow run started: {self.run_id}")

    def log_iteration(self, iteration: int, experiment: dict, finding: str, confidence: float):
        mlflow.log_metric("confidence", confidence, step=iteration)
        mlflow.log_metric("iteration", iteration, step=iteration)
        mlflow.log_text(
            json.dumps(experiment, indent=2),
            f"iteration_{iteration}_experiment.json"
        )
        mlflow.log_text(finding, f"iteration_{iteration}_finding.txt")

    def log_final(self, report: str, total_iterations: int, final_confidence: float):
        mlflow.log_metric("final_confidence", final_confidence)
        mlflow.log_metric("total_iterations", total_iterations)
        mlflow.log_text(report, "final_report.txt")

    def end_run(self):
        if self.run:
            mlflow.end_run()
            print(f"MLflow run ended: {self.run_id}")