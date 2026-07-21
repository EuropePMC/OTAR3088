import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
from typing import List
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import AIMessage

# Import class definitions from the orchestrator
from run_karenina_benchmarks import Entity, GeneticVariantAnswer
from karenina import Benchmark
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.config.models import ModelConfig

# Import regex and mapping tools from var_utils
from var_utils import (
    map_to_ascii,
    find_star_alleles,
    HGVS,
    CYTOBAND,
    GENOME_RE,
    REFSNP_RE,
    STAR_ALLELE_RE
)

class TestGeneticVariantMetrics(unittest.TestCase):
    def test_f1_exact_match(self):
        """Test F1 score calculation when predictions exactly match gold."""
        gold = [
            Entity(text="g.140453136A>T", start=10, end=25, label="HGVSVar"),
            Entity(text="rs4845618", start=40, end=49, label="RefSNP")
        ]
        pred = [
            Entity(text="g.140453136A>T", start=10, end=25, label="HGVSVar"),
            Entity(text="rs4845618", start=40, end=49, label="RefSNP")
        ]
        
        answer = GeneticVariantAnswer(entities=pred)
        answer.correct = {"entities": gold}
        
        self.assertTrue(answer.verify())
        self.assertAlmostEqual(answer.verify_granular(), 1.0)

    def test_f1_partial_match(self):
        """Test F1 score calculation with partial precision and recall."""
        # 2 gold entities
        gold = [
            Entity(text="g.140453136A>T", start=10, end=25, label="HGVSVar"),
            Entity(text="rs4845618", start=40, end=49, label="RefSNP")
        ]
        # 2 predicted entities: 1 correct (the rsID), 1 incorrect/extra
        pred = [
            Entity(text="rs4845618", start=40, end=49, label="RefSNP"),
            Entity(text="extra_entity", start=100, end=112, label="Other")
        ]
        
        answer = GeneticVariantAnswer(entities=pred)
        answer.correct = {"entities": gold}
        
        # Exact verify should fail
        self.assertFalse(answer.verify())
        
        # Precision = 1/2, Recall = 1/2 => F1 = 0.5
        self.assertAlmostEqual(answer.verify_granular(), 0.5)

    def test_f1_empty_cases(self):
        """Test boundary cases with empty predictions or gold entities."""
        # Both empty
        answer_both_empty = GeneticVariantAnswer(entities=[])
        answer_both_empty.correct = {"entities": []}
        self.assertTrue(answer_both_empty.verify())
        self.assertAlmostEqual(answer_both_empty.verify_granular(), 1.0)
        
        # Pred empty, gold has items
        answer_pred_empty = GeneticVariantAnswer(entities=[])
        answer_pred_empty.correct = {"entities": [Entity(text="rs123", start=0, end=5, label="RefSNP")]}
        self.assertFalse(answer_pred_empty.verify())
        self.assertAlmostEqual(answer_pred_empty.verify_granular(), 0.0)

        # Gold empty, pred has items
        answer_gold_empty = GeneticVariantAnswer(entities=[Entity(text="rs123", start=0, end=5, label="RefSNP")])
        answer_gold_empty.correct = {"entities": []}
        self.assertFalse(answer_gold_empty.verify())
        self.assertAlmostEqual(answer_gold_empty.verify_granular(), 0.0)


class TestKareninaPipelineOffline(unittest.TestCase):
    @patch("karenina.adapters.langchain.initialization.init_chat_model")
    def test_end_to_end_mock_verification(self, mock_init_chat_model):
        """Verify the Karenina benchmark suite runs end-to-end using mocked LLM responses."""
        # Set up a mock chat model
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm
        
        # The first invoke is the answering model generating raw annotations.
        mock_answer_json = json.dumps({
            "entities": [{"text": "g.140453136A>T", "start": 0, "end": 15, "label": "HGVSVar"}]
        })
        
        async def mock_base_ainvoke(lc_messages, **kwargs):
            return AIMessage(content=mock_answer_json)
        mock_llm.ainvoke = mock_base_ainvoke
        
        # Ensure that calling .with_structured_output(...) returns a mock model
        # that returns the parsed BaseModel schema instance
        def mock_with_structured_output(schema):
            mock_struct_model = MagicMock()
            
            # Instantiate the schema dynamically with parsed entity data
            parsed_instance = schema(entities=[{"text": "g.140453136A>T", "start": 0, "end": 15, "label": "HGVSVar"}])
            
            async def mock_ainvoke_structured(*args, **kwargs):
                return parsed_instance
                
            mock_struct_model.ainvoke = mock_ainvoke_structured
            return mock_struct_model
            
        mock_llm.with_structured_output.side_effect = mock_with_structured_output
        
        # Define a single-sentence benchmark
        benchmark = Benchmark.create(
            name="Mock Offline Test",
            description="Test with mocked LLM provider",
            version="1.0.0",
            creator="Offline Tester"
        )
        
        expected_entities = [
            Entity(text="g.140453136A>T", start=0, end=15, label="HGVSVar")
        ]
        
        # Add question with template
        qid = benchmark.add_question(
            question="Annotate: g.140453136A>T variant detected.",
            raw_answer=json.dumps([e.model_dump() for e in expected_entities]),
            author={"name": "SME", "email": "curator@example.com"}
        )
        
        # Answer template matching run_karenina_benchmarks.py dynamic schema
        entity_instantiations = "[Entity(text='g.140453136A>T', start=0, end=15, label='HGVSVar')]"
        template_code = f"""import pydantic
globals().update(BaseModel=pydantic.BaseModel, ConfigDict=pydantic.ConfigDict)
from typing import List, Literal
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
    entities: List[Entity] = VerifiedField(
        description="A JSON list of all extracted genetic variant entities.",
        ground_truth={entity_instantiations},
        verify_with=SetContainment(mode="exact")
    )

    def verify(self) -> bool:
        pred = set(self.entities)
        gold = set()
        for e in self.correct.get("entities", []):
            if isinstance(e, dict):
                gold.add(Entity(**e))
            else:
                gold.add(e)
        return pred == gold

    def verify_granular(self) -> float:
        pred = set(self.entities)
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
            
        return 2 * (precision * recall) / (precision + recall)
"""
        benchmark.update_template(qid, template_code)
        
        # Run verification using the mock model configuration
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="mock-target",
                    model_name="mock-model",
                    model_provider="openai",  # Interface mock ignores provider requirements
                    interface="langchain",
                    temperature=0.0,
                    system_prompt="Answering prompt"
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="mock-judge",
                    model_name="mock-model",
                    model_provider="openai",
                    interface="langchain",
                    temperature=0.0
                )
            ],
            evaluation_mode="template_only",
            rubric_enabled=False,
            async_enabled=False,
            async_max_workers=1
        )
        
        results = benchmark.run_verification(config)
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
        
        # Assertions
        self.assertIsNotNone(results)
        self.assertIn('granular_score', df_results.columns)
        self.assertAlmostEqual(df_results['granular_score'].iloc[0], 1.0)
        
        # aggregate_pass_rate returns a dict: {question_id: pass_rate}
        pass_rates = template_results.aggregate_pass_rate()
        self.assertEqual(len(pass_rates), 1)
        self.assertAlmostEqual(list(pass_rates.values())[0], 1.0)
        
        print("DEBUG df_results columns:", list(df_results.columns))
        print("DEBUG df_results row:\n", df_results.iloc[0].to_dict() if len(df_results) > 0 else "Empty")
        
        print("Offline pipeline execution test passed successfully.")


class TestVariantUtilities(unittest.TestCase):
    def test_map_to_ascii(self):
        # Cyrillic 'с' (U+0441) should be mapped to ASCII 'c' (U+0063)
        cyrillic_c_text = "\u0441.181T>G"
        self.assertEqual(map_to_ascii(cyrillic_c_text), "c.181T>G")
        
        # Cyrillic 'р' (U+0440) should map to ASCII 'p'
        cyrillic_p_text = "\u0440.V600E"
        self.assertEqual(map_to_ascii(cyrillic_p_text), "p.V600E")

        # Cyrillic 'е' (U+0435) should map to ASCII 'e'
        # Cyrillic 'a' (U+0430) should map to ASCII 'a'
        cyrillic_ea = "\u0435\u0430"
        self.assertEqual(map_to_ascii(cyrillic_ea), "ea")

        # Hyphen lookalikes
        en_dash = "g.120_121\u2013del"
        self.assertEqual(map_to_ascii(en_dash), "g.120_121-del")
        minus_sign = "c.123\u22124G>A"
        self.assertEqual(map_to_ascii(minus_sign), "c.123-4G>A")

    def test_ref_snp_regex(self):
        text = "The variant rs4845618 is associated with disease."
        matches = REFSNP_RE.findall(text)
        self.assertEqual(matches, ["rs4845618"])

    def test_star_allele_regex(self):
        # Full pharmacogenetic patterns
        texts = [
            ("CYP2D6*68", ["CYP2D6*68"]),
            ("CYP2C19*1/*1", ["CYP2C19*1/*1"]),
            ("CYP2D6*68 + *4", ["CYP2D6*68 + *4"]),
        ]
        for text, expected in texts:
            matches = STAR_ALLELE_RE.findall(text)
            self.assertEqual(matches, expected)

    def test_cytoband_regex(self):
        text = "A chr9q34.3 deletion was detected. 1p36.3 duplication is common."
        matches = [m.group() for m in CYTOBAND.finditer(text)]
        self.assertEqual(matches, ["chr9q34.3 deletion", "1p36.3 duplication"])

    def test_genome_regex(self):
        text = "Mapped against GRCh38 and hg19."
        matches = [m.group() for m in GENOME_RE.finditer(text)]
        self.assertEqual(matches, ["GRCh38", "hg19"])

    def test_hgvs_regex(self):
        # Test various HGVS types
        cases = [
            ("g.140453136A>T", "g.140453136A>T"),
            ("c.1799T>A", "c.1799T>A"),
            ("p.Val600Glu", "p.Val600Glu"),
            ("p.V600E", "p.V600E"),
            ("NM_004333.6:c.1799T>A", "NM_004333.6:c.1799T>A"),
            ("r.76a>u", "r.76a>u"),
            ("n.45G>C", "n.45G>C"),
        ]
        for text, expected in cases:
            m = HGVS.search(text)
            self.assertIsNotNone(m)
            self.assertEqual(m.group(), expected)

    def test_find_star_alleles(self):
        text = "The CYP2D6 *4 allele was analyzed."
        gene_spans = [("CYP2D6", 4, 10, "GeneProtein")]
        results = find_star_alleles(text, gene_spans)
        self.assertEqual(results, [("CYP2D6 *4", 4, 13, "StarAllele")])

if __name__ == "__main__":
    unittest.main()
