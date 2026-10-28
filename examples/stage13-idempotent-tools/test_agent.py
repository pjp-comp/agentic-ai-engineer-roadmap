import unittest
from unittest.mock import patch
from agent import ChargeLedger, CachedTool


class RetryTests(unittest.TestCase):
    def test_retry_and_distinct_purchase(self):
        ledger = ChargeLedger()
        first = ledger.charge_card(49.99, 'customer', 'purchase-1')
        self.assertEqual(ledger.charge_card(49.99, 'customer', 'purchase-1'), first)
        self.assertEqual(len(ledger.actual_charges), 1)
        ledger.charge_card(49.99, 'customer', 'purchase-2')
        self.assertEqual(len(ledger.actual_charges), 2)

    def test_changed_payload_and_empty_identity_rejected(self):
        ledger = ChargeLedger()
        ledger.charge_card(10, 'customer', 'purchase-1')
        with self.assertRaises(ValueError):
            ledger.charge_card(20, 'customer', 'purchase-1')
        with self.assertRaises(ValueError):
            ledger.charge_card(10, 'customer', '')
        self.assertEqual(len(ledger.actual_charges), 1)

    def test_cache_expires(self):
        calls = []
        def lookup(city):
            calls.append(city)
            return city
        cache = CachedTool(lookup, ttl_seconds=5)
        with patch('agent.time.monotonic', return_value=0):
            cache(city='Paris')
            cache(city='Paris')
        self.assertEqual(len(calls), 1)
        with patch('agent.time.monotonic', return_value=6):
            cache(city='Paris')
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
