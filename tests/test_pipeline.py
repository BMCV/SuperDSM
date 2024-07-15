import unittest
import ray
import superdsm.automation, superdsm.io, superdsm.render

from . import testsuite


class pipeline(unittest.TestCase):

    @testsuite.without_resource_warnings
    def setUp(self):
        ray.init(num_cpus=4, log_to_driver=False, logging_level=ray.logging.ERROR)
        self.pipeline = superdsm.pipeline.create_default_pipeline()

    @testsuite.without_resource_warnings
    def tearDown(self):
        ray.shutdown()
        del self.pipeline


if __name__ == '__main__':
    unittest.main()
