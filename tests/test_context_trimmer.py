import json
import unittest
from app.harness.context_trimmer import content_fingerprint, normalize_url, simhash64, trim_context
from app.harness.token_counter import counter
from tests.helpers import evidence


class ContextTrimmerTests(unittest.TestCase):
    def test_url_normalization_preserves_semantic_query(self):
        self.assertEqual(normalize_url("https://EXAMPLE.com:443/a?b=2&utm_source=x&a=1#part"), "https://example.com/a?a=1&b=2")
        self.assertNotEqual(normalize_url("https://example.com/a?version=1"), normalize_url("https://example.com/a?version=2"))

    def test_stable_fingerprints(self):
        self.assertEqual(content_fingerprint("Ａ Test\n text"), content_fingerprint("a test text"))
        self.assertEqual(simhash64("中文 关键 结论"), simhash64("中文 关键 结论"))
        self.assertLess(simhash64("hello world"), 1 << 64)

    def test_exact_duplicates_merge_provenance(self):
        items = [evidence(id="a"), evidence("ragflow", id="b")]
        result = trim_context(items, "Revenue")
        self.assertEqual(result.duplicates_removed, 1)
        self.assertEqual(len(result.evidence), 1)
        self.assertTrue(result.evidence[0].aliases)

    def test_same_url_different_paragraphs_survive(self):
        a, b = evidence(text="Product price is 20 dollars.", id="a"), evidence(text="Annual support uses a different contract.", id="b")
        b.locator = a.locator
        self.assertEqual(len(trim_context([a, b], "price support").evidence), 2)

    def test_numbers_and_negation_are_not_deduplicated(self):
        text = "The audited annual revenue of this product is {number} million and the data uses the same financial accounting definition."
        items = [evidence(text=text.format(number=20), id="a"), evidence(text=text.format(number=21), id="b"),
                 evidence(text="This product does not support private deployment.", id="c"), evidence(text="This product does support private deployment.", id="d")]
        self.assertEqual(len(trim_context(items, "revenue private").evidence), 4)

    def test_near_duplicate_detection(self):
        base = "The public report describes a hosted platform with usage based billing and a separate enterprise deployment option. " * 5
        result = trim_context([evidence(text=base, id="a"), evidence(text=base + "Overview.", id="b")], "billing")
        self.assertEqual(result.duplicates_removed, 1)

    def test_currency_difference_survives_near_dedup(self):
        text = "The audited annual subscription price is 20 {currency} per user and this statement uses the full financial reporting definition. " * 4
        items = [evidence(text=text.format(currency="USD"), id="a"), evidence(text=text.format(currency="EUR"), id="b")]
        self.assertEqual(trim_context(items, "price", 1500).duplicates_removed, 0)

    def test_qualification_remains_attached_to_claim(self):
        item = evidence(text="Background. " * 30 + "Revenue is 42 million. However this number is not audited. " + "Other context. " * 30)
        result = trim_context([item], "revenue", 400)
        self.assertIn("Revenue is 42 million. However this number is not audited.", result.content)

    def test_decimal_numbers_are_not_cut_at_the_period(self):
        item = evidence(text="Background. " * 100 + "Critical price is 20.5 USD and revenue is 42.75 million. " + "Other context. " * 100)
        result = trim_context([item], "critical price revenue", 500)
        self.assertIn("20.5 USD", result.content)
        self.assertIn("42.75 million", result.content)

    def test_full_serialized_envelope_obeys_budget(self):
        item = evidence(text="Background notes. " * 300 + "Critical revenue is 42 million. " + "Other discussion. " * 300)
        result = trim_context([item], "Critical revenue", 500, {"status": "ok"})
        self.assertLessEqual(counter.count(result.content), 500)
        self.assertEqual(counter.count(result.content), result.tokens_after)
        self.assertIn("Critical revenue is 42 million.", result.content)
        self.assertTrue(result.truncated)
        self.assertGreater(result.tokens_before, result.tokens_after)
        json.loads(result.content)

    def test_sql_rows_stay_atomic(self):
        row = evidence("mysql", text='{"amount": "' + "1" * 500 + '", "currency": "USD"}', kind="row")
        result = trim_context([row], "amount", 180)
        self.assertEqual(result.evidence, [])
        self.assertTrue(result.truncated)

    def test_too_small_metadata_budget_rejected(self):
        with self.assertRaises(ValueError):
            trim_context([], "", 32, {"note": "x" * 100})

    def test_empty_result_and_source_coverage(self):
        self.assertEqual(json.loads(trim_context([], "").content)["evidence"], [])
        items = [evidence(source=s, text=f"{s} supported finding.", id=s) for s in ("tavily", "mysql", "ragflow")]
        self.assertEqual({x.source for x in trim_context(items, "finding", 1500).evidence}, {"tavily", "mysql", "ragflow"})
