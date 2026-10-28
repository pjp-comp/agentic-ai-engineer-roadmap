"""Deterministic controls only; model injection resistance needs separate evals."""
import unittest
from agent import evaluate_arithmetic, enforce_allowlist, ToolNotAllowedError, run_calculator


class GuardrailTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertAlmostEqual(evaluate_arithmetic('(23*47+100)/3'), 393.6666666666667)
        self.assertEqual(evaluate_arithmetic('-5 + 9 // 2'), -1)

    def test_rejects_python_and_unbounded_inputs(self):
        for expr in ["__import__('os').listdir('.')", '(1).__class__', 'True',
                     '2**1000000', '1e100', '9'*201, '1+'*40+'1', '[1,2]']:
            with self.subTest(expr=expr), self.assertRaises((ValueError, SyntaxError)):
                evaluate_arithmetic(expr)

    def test_error_is_returned(self):
        self.assertTrue(run_calculator('1/0').startswith('error:'))

    def test_permissions_default_deny(self):
        for role, tool in [('support', 'calculate'), ('unknown', 'lookup_order'), ('admin', 'delete')]:
            with self.subTest(role=role, tool=tool), self.assertRaises(ToolNotAllowedError):
                enforce_allowlist(role, tool)
        enforce_allowlist('admin', 'calculate')


if __name__ == '__main__':
    unittest.main()
