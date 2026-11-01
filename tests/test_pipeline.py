import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instamart_copilot.part_c import mask_pii, estimate_tokens, deterministic_classify
from instamart_copilot.part_d import PolicyRetriever, refund_recommendation

class TestPrivacyAndBudget(unittest.TestCase):
    def test_email_masked(self):
        self.assertNotIn("person@example.com", mask_pii("Contact person@example.com"))

    def test_indian_mobile_masked(self):
        self.assertNotIn("9876543210", mask_pii("Call 9876543210"))

    def test_token_estimate_positive(self):
        self.assertGreaterEqual(estimate_tokens("hello"), 1)

    def test_payment_routes_urgent(self):
        result = deterministic_classify("I was charged twice")
        self.assertIn(result["category"], {"refund","payment"})

class TestRetrievalSafety(unittest.TestCase):
    def test_empty_corpus_abstains(self):
        retriever = PolicyRetriever([])
        self.assertEqual(retriever.retrieve("refund please"), [])
        decision = refund_recommendation("refund please", [])
        self.assertEqual(decision["decision"], "human_review")

if __name__ == "__main__":
    unittest.main()
