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

    def test_Pipeline_process_image(self):
        data_path = testsuite.require_data('bbbc033', 'C2.tif')
        img_3d = superdsm.io.imread(data_path)
        img = img_3d[28]
        cfg = superdsm.config.Config()
        cfg['dsm/mu'] = 1
        data, _, _ = superdsm.automation.process_image(self.pipeline, cfg, img)
        seg = superdsm.render.render_result_over_image(data, normalize_img=False)
        testsuite.validate_image(self, 'pipeline.Pipeline.process_image/bbbc033-z28.png', seg)


if __name__ == '__main__':
    unittest.main()
