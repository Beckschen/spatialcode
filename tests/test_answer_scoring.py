"""Scoring regressions without loading a model or contacting an API."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Train/RL"))

from scripts import eval_code_to_answer as evaluate
from scripts import run_code_to_answer as run
from spatialcode.utils.scoring import score_sample


def sample(solution="35.9", qa_type="room_size_estimation"):
    return {
        "question": "What is the size of this room?",
        "ground_truth": solution,
        "question_type": qa_type,
        "gt_3d_bboxes": [{"label": "chair", "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0]}],
    }


class AnswerScoringTests(unittest.TestCase):
    def test_all_numeric_tasks_use_mra(self):
        for qa_type in ["object_counting", "object_abs_distance",
                        "object_size_estimation", "room_size_estimation"]:
            with self.subTest(qa_type=qa_type):
                self.assertEqual(score_sample("36", "35.9", qa_type), (1.0, True))
                self.assertEqual(score_sample("124", "100", qa_type), (0.6, True))
                self.assertEqual(score_sample("149", "100", qa_type), (0.1, True))
                self.assertEqual(score_sample("150", "100", qa_type), (0.0, True))

    def test_zero_and_invalid_numeric_predictions(self):
        for answer, solution, expected in [
            ("0", "0", 1.0), ("1", "0", 0.0),
            ("", "35.9", 0.0), ("unknown", "35.9", 0.0),
        ]:
            with self.subTest(answer=answer, solution=solution):
                self.assertEqual(score_sample(answer, solution, "object_counting"), (expected, True))

    def test_choice_and_yes_no_normalization(self):
        for answer, solution, qa_type, expected in [
            (" a. ", "A", "object_rel_direction_easy", 1.0),
            ("B", "A", "route_planning", 0.0),
            (" YES ", "Yes", "pairwise_configuration", 1.0),
            ("No", "Yes", "pairwise_compatibility", 0.0),
            ("", "", "unknown", 0.0),
        ]:
            with self.subTest(answer=answer, qa_type=qa_type):
                self.assertEqual(score_sample(answer, solution, qa_type), (expected, False))

    def run_default(self, samples, answers):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.json"
            path.write_text(json.dumps(samples))
            output = io.StringIO()
            args = ["run_code_to_answer.py", "--qa_json", str(path), "--limit", "10"]
            with patch.object(sys, "argv", args), patch.object(run, "llm") as llm, redirect_stdout(output):
                llm.return_value.reason.side_effect = answers
                run.main()
                calls = llm.return_value.reason.call_args_list
            return output.getvalue(), calls

    def test_default_script_reports_full_mra_credit(self):
        for field in ["qa_type", "question_type"]:
            item = sample()
            item[field] = item.pop("question_type")
            with self.subTest(field=field):
                output, calls = self.run_default([item], ["36"])
                self.assertIn("Score:    1.00 (MRA)", output)
                self.assertIn("Accuracy: 1.0/1 = 100.00%", output)
                self.assertEqual(calls[0].kwargs["task_type"], "room_size_estimation")

    def test_default_script_keeps_fractional_credit_and_excludes_skips(self):
        samples = [sample("100"), sample("Yes", "pairwise_configuration"), {"question": "Skipped"}]
        output, calls = self.run_default(samples, ["124", "yes"])
        self.assertIn("Score:    0.60 (MRA)", output)
        self.assertIn("Score:    1.00 (exact match)", output)
        self.assertIn("Accuracy: 1.6/2 = 80.00%", output)
        self.assertEqual(len(calls), 2)

    def test_empty_and_all_skipped_inputs(self):
        for samples in [[], [{"question": "Skipped"}]]:
            with self.subTest(samples=samples):
                output, calls = self.run_default(samples, [])
                self.assertIn("Accuracy: N/A (no samples)", output)
                self.assertEqual(calls, [])

    def test_evaluation_uses_question_type_and_same_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "qa.json"
            target = Path(directory) / "results.json"
            source.write_text(json.dumps([sample(), sample("100")]))
            args = ["eval_code_to_answer.py", "--qa_json", str(source), "--output_json", str(target)]
            with patch.object(sys, "argv", args), patch.object(evaluate, "llm") as llm:
                llm.return_value.backend.batch_generate.return_value = ["Final Answer: 36", "Final Answer: 124"]
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    evaluate.main()
            results = json.loads(target.read_text())
            self.assertEqual([r["score"] for r in results], [1.0, 0.6])
            self.assertEqual([r["qa_type"] for r in results], ["room_size_estimation"] * 2)

    def test_recompute_uses_same_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.jsonl"
            target = Path(directory) / "scored.json"
            rows = [dict(sample(), pred="36"), dict(sample("100"), pred="124")]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            args = ["eval_code_to_answer.py", "--recompute_from", str(source), "--output_json", str(target)]
            with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                evaluate.main()
            self.assertEqual([r["score"] for r in json.loads(target.read_text())], [1.0, 0.6])


if __name__ == "__main__":
    unittest.main()
