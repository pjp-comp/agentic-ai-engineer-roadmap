import unittest
from app import calc, call_tool, ToolError


class FanoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_arithmetic(self):
        self.assertEqual((await calc('2*(3+4)'))['result'], 14)

    async def test_untrusted_request_is_rejected(self):
        for expr in ["__import__('os').getcwd()", '(1).__class__', '2**1000000']:
            with self.subTest(expr=expr), self.assertRaises(ToolError):
                await calc(expr)

    async def test_partial_failure(self):
        import asyncio
        async def bad():
            raise ToolError('forced failure')
        # A fresh test tool name avoids circuit state from previous runs.
        results = dict(await asyncio.gather(call_tool('test_bad', bad()), call_tool('test_calc', calc('2+2'))))
        self.assertIn('error', results['test_bad'])
        self.assertEqual(results['test_calc']['result'], 4)


if __name__ == '__main__':
    unittest.main()
