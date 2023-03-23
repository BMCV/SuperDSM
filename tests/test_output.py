import unittest
import numpy as np
import superdsm.output

from . import testsuite


class output(unittest.TestCase):

    def test_get_output(self):
        out1 = superdsm.output.get_output(None)
        self.assertFalse(out1.muted)
        out2 = superdsm.output.get_output(out1)
        self.assertIs(out1, out2)
        out3 = superdsm.output.get_output('muted')
        self.assertTrue(out3.muted)


if __name__ == '__main__':
    unittest.main()
