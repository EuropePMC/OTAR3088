import re
from ast import literal_eval

from dataclasses import dataclass
from typing import (
                    List, Dict, Tuple,
                    Union, Optional, Any, 
                    Callable, Literal
                    )



from collections import Counter
from tqdm import tqdm

import spacy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

from .entity_processor import sentencize_and_align_entity_spans



sns.set_style("darkgrid")
sns.set_palette("rocket")


EntityDict = List[Dict[str, Any]]


@dataclass
class NerDataFields:
    text_col: str
    label_col: str
    ent_label_key: str


def get_relevant_cols(df):
  try:
    columns = ["text", "label"]
    df = df[columns]
    return df
  except KeyError:
    columns = ["text", "ner"]
    df = df[columns]
    df.columns = ["text", "label"]
    return df



def detect_section_headers(text: str, entities: EntityDict) -> bool:
  """
  Detects if a given text segment is likely a section header based on its content
  or if it's a short text containing 'PMC' without any associated entities.

  Args:
      text (str): The text segment to check.
      entities (List[Dict[str, Any]]): A list of entities associated with the text segment.

  Returns:
      bool: True if the text segment is likely a section header, False otherwise.
  """
  text = text.strip()
  common_headers = {"abstract", "background", "results",
                    "discussion", "source paper", "declaration of interests",
                    "data collection and avaliability", "acknowledgement",
                    "REAGENT or RESOURCE SOURCE IDENTIFIER Antibodies", "data source"
                    }

  # Check for short PMC IDs, which are often file identifiers/headers
  if "PMC" in text and len(text.split()) < 10:
    return True

  # Check for common section header titles if there are no entities
  if any(common_header in text.lower() for common_header in common_headers) and not entities:
    return True

  return False



class ArticleNormaliser:
    """
    Sentencises an article data by:
    - filtering section headers
    - strategically sentencising rows by enforcing hard length constraints
    - aligning entity spans at sentence level
    """

    def __init__(
        self,
        params,
        section_header_func: Callable,
        *,
        min_len: int = 50,
        max_len: int = 500,
    ):
        self.params = params
        self.section_header_func = section_header_func
        self.min_len = min_len
        self.max_len = max_len

    def __repr__(self):
        return (
            f"ArticleNormaliser(min_len={self.min_len}, "
            f"max_len={self.max_len}, params={self.params})"
        )

    def _resolve_entity_keys(self, ent):
        start_key = ent["start"] if isinstance(ent.get("start"), int) else ent["startOffset"]
        end_key = ent["end"] if isinstance(ent.get("end"), int) else ent["endOffset"]
        if self.params.ent_label_key in ent:
            label = ent[self.params.ent_label_key]
        elif "label" in ent:
            label = ent.get("label")
        else:
            raise KeyError("Missing Entity_Label in dict. Skipping")


        return start_key, end_key, label

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        all_sentence_records = []

        for i, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Sentencising Article--->",
            colour="blue",
        ):
            text = row[self.params.text_col]
            raw_entities = row[self.params.label_col]

            if not isinstance(text, str) or not text.strip():
                continue

            if self.section_header_func(text, raw_entities):
                continue

            entities = self._normalise_entity_schema(raw_entities, row_id=i)
            if not entities:
                continue

            segments = self._segment_row(text, entities)

            for seg_text, seg_entities in segments:
                sent_recs = sentencize_and_align_entity_spans(
                    document=seg_text,
                    doc_annotations=seg_entities,
                    label_field="label",
                )

                for sent_rec in sent_recs:
                    sent_rec["entities"] = self._validate_entity_alignment(
                        sent_rec, row_id=i
                    )
                    all_sentence_records.append(sent_rec)

        return pd.DataFrame(all_sentence_records)

    def _segment_row(
        self,
        text: str,
        entities: EntityDict,
    ) -> List[Tuple[str, EntityDict]]:
        """
        Enforces length constraints *before* sentencisation.
        Returns list of (text_segment, entities_segment).
        """
        text_len = len(text)

        # Short segments. No need to split
        if text_len < self.min_len:
            return [(text, entities)]

        # Acceptable length boundary
        if text_len <= self.max_len:
            return [(text, entities)]

        # Long segments: split if no periods OR has semicolons
        if text_len > self.max_len and (";" in text or "." not in text):
            return self._split_long_text(text, entities)

        # Otherwise, return as is for sentence-level splitting
        return [(text, entities)]

    def _split_long_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[tuple[str, List[Dict[str, Any]]]]:
        """
        Entity-aware splitting for overlong rows.
        """
        segments = []
        cursor = 0
        text_len = len(text)


        entities = sorted(entities, key=lambda e: e["start"])
        ent_idx = 0

        while cursor < text_len:
            # Start with max_len window
            window_end = min(cursor + self.max_len, text_len)
            original_window_end = window_end

            # Extend window to avoid cutting entities
            while ent_idx < len(entities) and entities[ent_idx]["start"] < window_end:
                if entities[ent_idx]["end"] > window_end:
                    window_end = min(entities[ent_idx]["end"], text_len)
                ent_idx += 1

            if window_end <= cursor:
                window_end = cursor + 1

            if window_end == original_window_end:
                # Force progress by moving cursor past current position
                # but first check if we're at the end
                if cursor >= text_len - 1:
                    break
                window_end = min(cursor + self.max_len, text_len)

            segment_text = text[cursor:window_end]

            segment_entities = []
            for ent in entities:
                # Check if entity overlaps with this segment
                if ent["end"] > cursor and ent["start"] < window_end:
                    # Calculate relative positions
                    rel_start = max(ent["start"] - cursor, 0)
                    rel_end = min(ent["end"] - cursor, window_end - cursor)

                    # Skip invalid/zero-length entities
                    if rel_start >= rel_end:
                        continue

                    # Preserve entity text if available
                    entity_text = ent.get("text", "")
                    if not entity_text:
                        # Extract from segment
                        entity_text = segment_text[rel_start:rel_end]

                    segment_entities.append({
                        "start": rel_start,
                        "end": rel_end,
                        "label": ent["label"],
                        "text": entity_text,
                    })

            segments.append((segment_text, segment_entities))
            cursor = window_end

        return segments

    def _normalise_entity_schema(
        self,
        entities: List[Dict[str, Any]],
        row_id: int,
    ) -> List[Dict[str, Any]]:
        cleaned_entities = []

        for ent in entities:
            try:
                start, end, label = self._resolve_entity_keys(ent)

                text = ent.get("text")

                cleaned_entities.append(
                    {"start": start, "end": end, "label": label, "text": text}
                )
            except Exception as e:
                logger.warning(
                    f"Skipping malformed entity in row {row_id}: {e}"
                )
                logger.info(f"Entity Dict missing key: {ent}")
            continue

        return cleaned_entities

    def _validate_entity_alignment(
        self,
        sent_rec: Dict[str, Any],
        row_id: int,
    ) -> List[Dict[str, Any]]:
        sent_text = sent_rec["sentence"]
        validated = []

        for ent in sent_rec["entities"]:
            start, end = ent["start"], ent["end"]
            entity_text = ent.get("text", "")

            # Validate bounds
            if start < 0 or end > len(sent_text) or start >= end:
                logger.warning(
                    f"Invalid entity bounds in row {row_id}: [{start}:{end}] "
                    f"for sentence of length {len(sent_text)}"
                )
                continue

            # Extract actual text from sentence
            actual_text = sent_text[start:end]

            # If entity text is empty, fill it
            if not entity_text:
                ent["text"] = actual_text
                validated.append(ent)
            # If there's a mismatch, warn but keep with corrected text
            elif actual_text != entity_text:
                logger.warning(
                    f"Entity text mismatch in row {row_id}. "
                    f"Expected: '{entity_text}', Got: '{actual_text}'. "
                    f"Correcting..."
                )
                ent["text"] = actual_text
                validated.append(ent)
            else:
                validated.append(ent)

        return validated




class NERDatasetAnalyser:
    """
    Analyse and visualise statistics of a sentence-level NER dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sent_col: str = "sentence",
        ent_col: str = "entities",
    ):
        self.df = df.copy()
        self.sent_col = sent_col
        self.ent_col = ent_col

        self.df["sent_len"] = self.df[self.sent_col].apply(lambda x: len(x.split()))

        self.sentence_stats = self._compute_sentence_stats()


    def _compute_sentence_stats(self) -> Dict[str, float]:
        lengths = self.df["sent_len"]
        return {
            "mean": lengths.mean(),
            "min": lengths.min(),
            "max": lengths.max(),
        }

    def compute_entity_stats(self) -> Dict[str, Any]:
        label_counter = Counter()
        sentences_with_entities = 0
        total_entities = 0

        for entities in self.df[self.ent_col]:
            if entities:
                sentences_with_entities += 1
                for ent in entities:
                    label_counter.update([ent["label"]])
                    total_entities += 1

        label_percentages = (
            {k: (v / total_entities) * 100 for k, v in label_counter.items()}
            if total_entities > 0
            else {}
        )

        return {
            "sentences_with_entities": sentences_with_entities,
            "percent_sents_stats": sentences_with_entities / len(self.df) * 100,
            "total_entities": total_entities,
            "label_counts": dict(label_counter),
            "label_percentages": label_percentages,

        }

    def print_summary(self) -> None:
        ent_stats = self.compute_entity_stats()

        print(
            f"Sentences with entities: {ent_stats['sentences_with_entities']} / {len(self.df)}"
        )
        print(f"Total entities in article: {ent_stats['total_entities']}")
        print(
          f"Label Counts:\n {ent_stats['label_counts']}"
        )

        print(f"Percent sentence stats: {ent_stats['percent_sents_stats']:.2f}%")
        print("\nSentence length stats:")
        for k, v in self.sentence_stats.items():
            print(f"  {k}: {v:.2f}")

        print("\nEntity distribution (%):")
        for label, pct in ent_stats["label_percentages"].items():
            print(f"  {label}: {pct:.2f}%")


    def plot_data(self) -> None:
      ent_stats = self.compute_entity_stats()

      fig, axes = plt.subplots(1, 2, figsize=(18, 8))

      sns.violinplot(
            y=self.df["sent_len"],
            ax= axes[0],
            color="blue",
            inner=None,
        )

      for name, value in self.sentence_stats.items():
        #ax[0].axhline(y=value, linestyle='--', color='gray', alpha=0.7, linewidth=1)
        #
        axes[0].axhline(value, linestyle="--", linewidth=1)
        axes[0].text(0.5, value, f'{name}: {value:.2f}', va='center', ha='left',
                     backgroundcolor='white', color='red', fontsize=16)

      axes[0].set_title("Sentence Length Distribution", fontsize=16)
      axes[0].set_ylabel("Sentence Length", fontsize=14)
      axes[0].tick_params(labelsize=12)


      ent_dist_counts = ent_stats["label_counts"]
      ent_dist_percentages = ent_stats["label_percentages"]


      plot_df = pd.DataFrame({'Label': list(ent_dist_counts.keys()),
                            'Count': list(ent_dist_counts.values()),
                            'Percentage': [ent_dist_percentages[label] for label in ent_dist_counts.keys()]})

      barplot = sns.barplot(x="Label", y="Count", data=plot_df, ax=axes[1], palette="pastel", hue="Label")
      axes[1].set_title("Entity Label Distribution", fontsize=16)
      axes[1].set_ylabel("Count")
      axes[1].tick_params(axis="x", rotation=45, labelsize=16)
      axes[1].tick_params(axis="y", labelsize=14)


      for index, p in enumerate(barplot.patches):
          percentage = f'{plot_df["Percentage"].iloc[index]:.2f}%'
          x = p.get_x() + p.get_width() / 2
          y = p.get_height()
          axes[1].annotate(percentage, (x, y), ha='center', va='bottom', fontsize=16, color='red') # Increased fontsize, red color

      plt.tight_layout()
      plt.show()

