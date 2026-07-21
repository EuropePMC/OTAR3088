import argparse
import ast
import json
import os
import pandas as pd
from typing import List, Literal
from pydantic import BaseModel, ConfigDict, SecretStr

from karenina import Benchmark
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.primitives import SetContainment

# --- 1. Define Entity Pydantic Schema ---
class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    start: int
    end: int
    label: Literal["HGVSVar", "RefSNP", "StarAllele", "ISCNVar", "Refgenome", "Other"]

# --- 2. Custom Answer Template with F1 scoring ---
class GeneticVariantAnswer(BaseAnswer):
    entities: List[Entity] = VerifiedField(
        description="A JSON list of all extracted genetic variant entities containing start, end, label, and text.",
        ground_truth=[],  # Populated dynamically
        verify_with=SetContainment(mode="exact")
    )

    def verify(self) -> bool:
        """True if exact match of all entities."""
        pred = set(self.entities)
        gold = set(self.correct.get("entities", []))
        return pred == gold

    def verify_granular(self) -> float:
        """Compute and return the F1-Score of the extracted entities."""
        pred = set(self.entities)
        gold = set(self.correct.get("entities", []))
        
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
            
        intersection = pred & gold
        precision = len(intersection) / len(pred)
        recall = len(intersection) / len(gold)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

def load_skill_content(skill_path: str) -> str:
    """Read prompt guidelines from SKILL.md."""
    if os.path.exists(skill_path):
        with open(skill_path, "r") as f:
            return f.read()
    return ""

def main():
    parser = argparse.ArgumentParser(description="Run genetic variant annotation benchmarking using Karenina.")
    parser.add_argument("--limit", type=int, default=10, help="Number of sentences to evaluate.")
    parser.add_argument("--run-type", choices=["zero_shot", "skill_guided"], default="zero_shot", help="Benchmarking mode.")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Model name.")
    parser.add_argument("--provider", type=str, default="google_genai", help="Model provider (e.g., google_genai, openai, anthropic).")
    parser.add_argument("--interface", type=str, default="langchain", choices=["langchain", "openai_endpoint", "openrouter"], help="Model interface (langchain, openai_endpoint).")
    parser.add_argument("--endpoint-base-url", type=str, default=None, help="Custom OpenAI-compatible endpoint base URL.")
    parser.add_argument("--endpoint-api-key", type=str, default=None, help="API key for custom endpoint.")
    parser.add_argument("--delay", type=float, default=None, help="Rate limit delay in seconds between requests.")
    args = parser.parse_args()

    # Automatically set defaults for local Ollama if using openai_endpoint
    if args.interface == "openai_endpoint":
        if not args.endpoint_base_url:
            args.endpoint_base_url = "http://localhost:11434/v1"
        if not args.endpoint_api_key:
            args.endpoint_api_key = "ollama"

    delay = args.delay
    if delay is None:
        delay = 4.0 if args.provider == "google_genai" else 0.0

    if delay > 0:
        print(f"Applying rate limit delay of {delay} seconds between LLM calls.")
        if args.provider == "google_genai":
            try:
                import time
                import asyncio
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                original_invoke = ChatGoogleGenerativeAI.invoke
                def patched_invoke(self, *args, **kwargs):
                    time.sleep(delay)
                    return original_invoke(self, *args, **kwargs)
                ChatGoogleGenerativeAI.invoke = patched_invoke

                original_ainvoke = ChatGoogleGenerativeAI.ainvoke
                async def patched_ainvoke(self, *args, **kwargs):
                    await asyncio.sleep(delay)
                    return await original_ainvoke(self, *args, **kwargs)
                ChatGoogleGenerativeAI.ainvoke = patched_ainvoke
                print("Successfully monkeypatched ChatGoogleGenerativeAI for rate limiting.")
            except ImportError:
                print("Warning: langchain_google_genai not installed, could not apply rate limiting patch.")
        elif args.provider == "openai":
            try:
                import time
                import asyncio
                from langchain_openai import ChatOpenAI
                
                original_invoke = ChatOpenAI.invoke
                def patched_invoke(self, *args, **kwargs):
                    time.sleep(delay)
                    return original_invoke(self, *args, **kwargs)
                ChatOpenAI.invoke = patched_invoke

                original_ainvoke = ChatOpenAI.ainvoke
                async def patched_ainvoke(self, *args, **kwargs):
                    await asyncio.sleep(delay)
                    return await original_ainvoke(self, *args, **kwargs)
                ChatOpenAI.ainvoke = patched_ainvoke
                print("Successfully monkeypatched ChatOpenAI for rate limiting.")
            except ImportError:
                print("Warning: langchain_openai not installed, could not apply rate limiting patch.")

    # Load dataset
    csv_path = "variant_ner_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)

    # Filter for rows that actually contain entities for testing
    df_with_ents = df[df['entities'].notna() & (df['entities'] != "[]")].head(args.limit)
    if df_with_ents.empty:
        print("No sentences with entities found.")
        return

    print(f"Loaded {len(df_with_ents)} evaluation sentences.")

    # Create Benchmark
    run_name = f"Variant NER - {args.run_type.upper()} ({args.model})"
    benchmark = Benchmark.create(
        name=run_name,
        description=f"Evaluation of {args.model} under {args.run_type} conditions.",
        version="1.0.0",
        creator="Antigravity Benchmarker"
    )

    # Add Questions and Templates
    for idx, row in df_with_ents.iterrows():
        sentence = row['sentence']
        try:
            sme_ents = ast.literal_eval(row['entities'])
        except Exception as e:
            print(f"Row {idx} parsing error: {e}")
            continue

        # Format ground truth expected entities
        expected_entities = []
        for ent in sme_ents:
            expected_entities.append(Entity(
                text=ent['text'],
                start=ent['start'],
                end=ent['end'],
                label=ent.get('label', ent.get('labels', ['Other']))[0]
            ))

        # We construct the question prompt
        question_text = f"Annotate the genetic variants in the following sentence. Return the annotations as a JSON list matching the output schema.\n\nSentence: {sentence}"
        
        qid = benchmark.add_question(
            question=question_text,
            raw_answer=json.dumps([e.model_dump() for e in expected_entities]),
            author={"name": "SME", "email": "curator@example.com"}
        )

        # Construct expected entities instantiations as a string
        entity_instantiations = "[" + ", ".join([
            f"Entity(text={repr(e.text)}, start={e.start}, end={e.end}, label={repr(e.label)})"
            for e in expected_entities
        ]) + "]"

        # Construct a self-contained answer template code string for this question
        template_code = f"""import pydantic
import typing
globals().update(
    BaseModel=pydantic.BaseModel,
    ConfigDict=pydantic.ConfigDict,
    ClassVar=typing.ClassVar,
    List=typing.List,
    Literal=typing.Literal
)
from karenina.schemas.entities import BaseAnswer, VerifiedField
from karenina.schemas.primitives import SetContainment

class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    start: int
    end: int
    label: Literal["HGVSVar", "RefSNP", "StarAllele", "ISCNVar", "Refgenome", "Other"]

globals().update(Entity=Entity)

class Answer(BaseAnswer):
    sentence: ClassVar[str] = {repr(sentence)}
    entities: List[Entity] = VerifiedField(
        description="A JSON list of all extracted genetic variant entities containing start, end, label, and text.",
        ground_truth={entity_instantiations},
        verify_with=SetContainment(mode="exact")
    )

    def verify(self) -> bool:
        pred_aligned = self.align_entities(self.entities)
        pred = set(pred_aligned)
        gold = set()
        for e in self.correct.get("entities", []):
            if isinstance(e, dict):
                gold.add(Entity(**e))
            else:
                gold.add(e)
        return pred == gold

    def verify_granular(self) -> float:
        pred_aligned = self.align_entities(self.entities)
        pred = set(pred_aligned)
        gold = set()
        for e in self.correct.get("entities", []):
            if isinstance(e, dict):
                gold.add(Entity(**e))
            else:
                gold.add(e)
        
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
            
        intersection = pred & gold
        precision = len(intersection) / len(pred)
        recall = len(intersection) / len(gold)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def align_entities(self, entities: List[Entity]) -> List[Entity]:
        aligned = []
        sentence_str = self.__class__.sentence
        for ent in entities:
            text = ent.text
            # Find all occurrences of text in sentence
            occurrences = []
            start_idx = 0
            while True:
                idx = sentence_str.find(text, start_idx)
                if idx == -1:
                    break
                occurrences.append(idx)
                start_idx = idx + 1
            
            if occurrences:
                # Find occurrence closest to predicted start
                closest_idx = min(occurrences, key=lambda x: abs(x - ent.start))
                aligned.append(Entity(
                    text=text,
                    start=closest_idx,
                    end=closest_idx + len(text),
                    label=ent.label
                ))
            else:
                aligned.append(ent)
        return aligned
"""
        # Register the template class string in the benchmark
        benchmark.update_template(qid, template_code)

    # Define Prompt Configuration based on Run Type
    system_prompt = "You are a precise biomedical information extraction assistant. Output only raw JSON."
    if args.run_type == "skill_guided":
        skill_text = load_skill_content("SKILL.md")
        if skill_text:
            system_prompt += f"\n\nFollow these guidelines:\n{skill_text}"
            print("Successfully loaded guidelines from SKILL.md.")

    # Configure Karenina Verification Run
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="target-model",
                model_name=args.model,
                model_provider=args.provider,
                interface=args.interface,
                endpoint_base_url=args.endpoint_base_url,
                endpoint_api_key=SecretStr(args.endpoint_api_key) if args.endpoint_api_key else None,
                temperature=0.0,
                system_prompt=system_prompt
            )
        ],
        parsing_models=[
            ModelConfig(
                id="judge-model",
                model_name=args.model,
                model_provider=args.provider,
                interface=args.interface,
                endpoint_base_url=args.endpoint_base_url,
                endpoint_api_key=SecretStr(args.endpoint_api_key) if args.endpoint_api_key else None,
                temperature=0.0
            )
        ],
        evaluation_mode="template_only",
        rubric_enabled=False,
        async_enabled=False,
        async_max_workers=1
    )

    print(f"Starting verification run using Karenina...")
    try:
        from tqdm import tqdm
        
        # Initialize tqdm progress bar (0% to 100%)
        pbar = tqdm(total=100, desc="Benchmarking Progress")
        last_completed = [0.0]
        
        def progress_update(percentage: float, message: str):
            pct_diff = (percentage - last_completed[0]) * 100
            if pct_diff > 0:
                pbar.update(pct_diff)
                last_completed[0] = percentage
            pbar.set_postfix_str(message)

        results = benchmark.run_verification(config, progress_callback=progress_update)
        pbar.close()
        print("Verification run completed successfully.")
        
        # Inspect and print metrics
        template_results = results.get_template_results()
        df_results = template_results.to_dataframe()
        
        # Map verify_granular_result from results list to the DataFrame
        granular_map = {}
        for r in template_results.results:
            if r.metadata and r.template:
                qid = r.metadata.question_id
                val = r.template.verify_granular_result
                if val is not None:
                    granular_map[qid] = val
        
        df_results['granular_score'] = df_results['question_id'].map(granular_map)
        
        # Write detailed diagnostics to a log file instead of console
        diagnostics_file = f"output/{args.run_type}_{args.model.replace(':', '_')}_diagnostics.txt"
        os.makedirs("output", exist_ok=True)
        with open(diagnostics_file, "w", encoding="utf-8") as df_log:
            df_log.write("=" * 60 + "\n")
            df_log.write("DIAGNOSTIC DETAILED RESPONSES\n")
            df_log.write("=" * 60 + "\n")
            for r in template_results.results:
                if r.metadata and r.template:
                    df_log.write(f"\nQuestion ID: {r.metadata.question_id}\n")
                    df_log.write(f"Sentence: {r.metadata.question_text}\n")
                    df_log.write(f"Raw LLM Response: {repr(r.template.raw_llm_response)}\n")
                    df_log.write(f"Parsed LLM Response: {r.template.parsed_llm_response}\n")
                    df_log.write(f"Parsed GT Response: {r.template.parsed_gt_response}\n")
                    df_log.write(f"F1-Score: {r.template.verify_granular_result}\n")
                    df_log.write(f"Exact Pass: {r.template.verify_result}\n")
            df_log.write("=" * 60 + "\n")
        print(f"Saved detailed diagnostics to: {diagnostics_file}")

        
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        pass_rates = list(template_results.aggregate_pass_rate().values())
        overall_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        print(f"Overall Pass Rate: {overall_pass_rate:.2%}")
        
        # Compute exact average F1 (granular score)
        avg_f1 = df_results['granular_score'].mean()
        print(f"Average Entity F1-Score: {avg_f1:.2%}")
        print("=" * 60)
        
        # Save checkpoint
        checkpoint_file = f"output/{args.run_type}_{args.model}_checkpoint.jsonld"
        os.makedirs("output", exist_ok=True)
        benchmark.save(checkpoint_file)
        print(f"Saved benchmark results checkpoint to: {checkpoint_file}")

    except Exception as e:
        print(f"Error executing verification run: {e}")

if __name__ == "__main__":
    main()
