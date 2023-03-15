import unittest
import testsuite
import numpy as np
import ray
import superdsm.automation, superdsm.io, superdsm.render


class render(unittest.TestCase):

    def setUp(self):
        ray.init(num_cpus=4, log_to_driver=False, logging_level=ray.logging.ERROR)
        self.pipeline = superdsm.pipeline.create_default_pipeline()

    def tearDown(self):
        ray.shutdown()
        del self.pipeline

    def test_render_result_over_image(self):
        data_path = testsuite.require_data('bbbc033')
        img_3d = superdsm.io.imread(data_path)
        img = img_3d[28]
        data, _, _ = superdsm.automation.process_image(self.pipeline, superdsm.config.Config(), img)
        seg = superdsm.render.render_result_over_image(data, normalize_img=False)
        testsuite.validate_image(self, 'render.render_result_over_image/bbbc033-z28.png', seg)


if __name__ == '__main__':
    unittest.main()
