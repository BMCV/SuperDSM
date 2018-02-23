class PCIntensityLabels:
    def __init__(self, g, f0, f1):
        self.g  = g
        self.f0 = f0
        self.f1 = f1
    
    def __call__(self, x):
        gx = self.g(x)
        return (self.f1 - gx) ** 2 - (self.f0 - gx) ** 2
    
    def get_map(self):
        return (self.f1 - self.g.model) ** 2 - (self.f0 - self.g.model) ** 2


class ThresholdedLabels:
    def __init__(self, g, threshold):
        self.g = g
        self.t = threshold
    
    def __call__(self, x):
        gx = self.g(x)
        return self.g(x) - self.t
    
    def get_map(self):
        return self.g.model - self.t


class CustomLabels:
    def __init__(self, g):
        self.g = g
    
    def __call__(self, x):
        return self.g(x)
    
    def get_map(self):
        return self.g.model

