import unittest
import ray
import superdsm.automation, superdsm.io, superdsm.render

from . import testsuite


class render(unittest.TestCase):

    @classmethod
    @testsuite.without_resource_warnings
    def setUpClass(self):
        ray.init(num_cpus=4, log_to_driver=False, logging_level=ray.logging.ERROR)
        img_3d = superdsm.io.imread(data_path)
        img = img_3d[28]
        pipeline = superdsm.pipeline.create_default_pipeline()
        data_path = testsuite.require_data('bbbc033', 'C2.tif')
        self.data, _, _ = superdsm.automation.process_image(self.pipeline, superdsm.config.Config(), img)

    @classmethod
    @testsuite.without_resource_warnings
    def tearDownClass(self):
        ray.shutdown()

    def test_render_result_over_image(self):
        seg = superdsm.render.render_result_over_image(self.data, normalize_img=False)
        testsuite.validate_image(self, 'render.render_result_over_image/bbbc033-z28.png', seg)

    def test_render_adjacencies(self):
        seg = superdsm.render.render_adjacencies(self.data, normalize_img=False)
        testsuite.validate_image(self, 'render.render_adjacencies/bbbc033-z28.png', seg)

    def test_render_ymap(self):
        seg = superdsm.render.render_ymap(self.data)
        testsuite.validate_image(self, 'render.render_ymap/bbbc033-z28.png', seg)

    def test_normalize_image(self):
        seg = superdsm.render.normalize_image(self.data)
        testsuite.validate_image(self, 'render.normalize_image/bbbc033-z28.png', seg)

    def test_render_atoms(self):
        seg = superdsm.render.render_atoms(self.data, normalize_img=False)
        testsuite.validate_image(self, 'render.render_atoms/bbbc033-z28.png', seg)

    def test_render_foreground_clusters(self):
        seg = superdsm.render.render_foreground_clusters(self.data, normalize_img=False)
        testsuite.validate_image(self, 'render.render_foreground_clusters/bbbc033-z28.png', seg)

    def test_rasterize_labels(self):
        seg = superdsm.render.rasterize_labels(self.data)
        testsuite.validate_image(self, 'render.rasterize_labels/bbbc033-z28.png', seg)


if __name__ == '__main__':
    unittest.main()
