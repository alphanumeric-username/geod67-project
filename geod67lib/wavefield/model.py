class VelocityModel:
    def __init__(self, data, hx, hz):
        self.data = data
        self.hx = hx
        self.hz = hz
    
    @property
    def width(self):
        return (self.data.shape[0] - 1) * self.hx
    
    @property
    def height(self):
        return (self.data.shape[1] - 1) * self.hz
    
    @property
    def shape(self):
        return self.data.shape