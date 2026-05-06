"""
to move all ner metrics logic here including seqeval and nervaluate f
"""

from loguru import logger
import wandb

import numpy as np
import pandas as pd

import evaluate
from seqeval.metrics import classification_report
import nervaluate
from nervaluate import Evaluator

from .trainer_config import NerPredictions
from ...shared.metrics_base import WandbMetricsLogger


def seqeval_metrics(label_list):
    metric = evaluate.load("seqeval")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
            ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        report = classification_report(true_labels, true_predictions)
        logger.info(f"Eval Classification Report:\n {report}")
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            }
    return compute_metrics


def decode_all_predictions(logits, label_ids, id2label):
    preds = np.argmax(logits, axis=2)

    true_labels, true_preds = [], []
    
    for pred, label in zip(preds, label_ids):

        labels_seq, preds_seq = [], []

        for p, l in zip(pred, label):
            if l != -100:
                labels_seq.append(id2label[l])
                preds_seq.append(id2label[p])

        true_labels.append(labels_seq)
        true_preds.append(preds_seq)

    return true_labels, true_preds


class NervaluateEvaluator:
    def __init__(self, ner_predictions: NerPredictions):
        self.true_labels = ner_predictions.true_labels
        self.pred_labels = ner_predictions.pred_labels
        self.label_names = ner_predictions.label_names
        self.evaluator = None

    def run_evaluation(self):
        self.evaluator = Evaluator(
                                self.true_labels,
                                self.pred_labels,
                                tags=self.label_names,
                                loader="list"
                            )

        results = self.evaluator.evaluate()
        results_per_entity = results["entities"]
        self._log_to_cli()

        return results_per_entity


    def _log_to_cli(self):
        if self.evaluator is None:
            raise ValueError("Call run_evaluation() before logging.")
        
        logger.info(
            "\nNervaluate Summary (entities):\n"
            f"{self.evaluator.summary_report(mode='entities')}"
        )

        logger.info(
            "\nNervaluate Entity-Level (strict):\n"
            f"{self.evaluator.summary_report(mode='entities', scenario='ent_type')}"
        )

        logger.info(
            "\nNervaluate Partial Match Report:\n"
            f"{self.evaluator.summary_report(mode='entities', scenario='partial')}"
        )
  


class SeqevalLogger(WandbMetricsLogger):
    def __init__(self, ner_predictions: NerPredictions, wandb_run):
        super().__init__(wandb_run)
        self.true_labels = ner_predictions.true_labels
        self.pred_labels = ner_predictions.pred_labels
        self.label_names = ner_predictions.label_names

        self.table_columns = ["Entity", "Precision", "Recall", "F1", "Support"]


    def _create_wandb_table(self):
        report = classification_report(self.true_labels, self.pred_labels, digits=3)
        table_data = []
        report_lines = report.splitlines()
        for line in report_lines[2:(len(self.label_names)+2)]:
            table_data.append(line.split())

        return wandb.Table(data = table_data,
                          columns = self.table_columns)


    def log(self):
        table = self._create_wandb_table()

        self.wandb_run.log({"Seqeval Classification Report":table})



class NervaluateLogger(WandbMetricsLogger):
    def __init__(self, results, wandb_run):
        super().__init__(wandb_run)
        self.results = results


    def log(self):
        for entity, evals in self.results.items():
            artifact_aliases = self._compute_artifact_aliases(entity)
            artifact_name = self._compute_artifact_name(entity)

            df = self._extract_results(evals)
            logger.info(f"\nEntity: {entity}\n{df.to_string(index=True)}")

            table = self._create_wandb_table(df)

            self.wandb_run.log({
                f"{entity}_nervaluate_metrics": table
            })
            #add to artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="inference",
                description=f"Nervaluate metrics for {entity}",
                )
            artifact.add(table, name="nervaluate_metrics_table")
            self.wandb_run.log_artifact(artifact, aliases=artifact_aliases)

    def _convert_to_percent(self, df):

      for col in df.columns:
          if col.startswith(("precision", "recall", "f1")):
              df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")

      return df

    def _extract_results(self, results):
        """
        Extracts results from a nervaluator object, 
        passes to a dataframe and returns 
        a transposed version for better visualisation
        """
        extracted = {k: v.__dict__ for k, v in results.items()}

        df = pd.DataFrame(extracted).T
        df = self._convert_to_percent(df)

        return df


    def _create_wandb_table(self, df):
        return wandb.Table(
                  columns=["Evaluation Type"] + list(df.columns),
                  data=[[idx] + row.tolist() for idx, row in df.iterrows()]
                  )

    def _compute_artifact_name(self, entity):
        return f"{entity}_{self.wandb_run.name}_nervalute_metrics" 

    
    def _compute_artifact_aliases(self, entity):
        return list(self.wandb_run.tags) + [entity] + ["nervaluate_metrics"]


