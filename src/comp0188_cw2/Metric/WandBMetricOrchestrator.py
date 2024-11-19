from typing import Any, Dict
import wandb


class WandBMetricOrchestrator:

    def __init__(self) -> None:
        """Class to handle pushing of metrics to currently initialised weights
        and biases run
        """
        pass
    
    def add_metric(*args, **kwargs):
        # For API compatability
        pass
    
    def update_metrics(self, metric_value_dict:Dict[str, Dict[str,Any]]):
        """Method for updating multiple metrics simulaneously

          Args:
              metric_value_dict (Dict[str, Dict[str, Any]]): A dictionary
              containing the relevant update values of the form:
              {*metric_name*:{"label": *value_label*, "value": *value_value*}}
          """
        
        wandb.log({
            metric:metric_value_dict[metric]["value"]
            for metric in metric_value_dict.keys()
            })