import chainer
from chainer import Chain,report
import chainer.functions as F
import chainer.links as L
class SRCNN(Chain):
    def __init__(self):
        super(SRCNN, self).__init__()
        f1,f2,f3 = 9,1,5
        n1,n2 = 64,32
        initializer = chainer.initializers.Normal(scale=0.001)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3, out_channels=n1, ksize=f1, stride=1, initialW=initializer)
            self.conv2 = L.Convolution2D(
                in_channels=n1, out_channels=n2, ksize=f2, stride=1, initialW=initializer)
            self.conv3 = L.Convolution2D(
                in_channels=n2, out_channels=3, ksize=f3, stride=1, initialW=initializer)

    def __call__(self, x, y):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        if chainer.config.train:
            h = F.mean_squared_error(h,y)
            report({'loss':h})
        return h