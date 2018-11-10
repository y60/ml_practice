from chainer import Chain
import chainer.functions as F
import chainer.links as L
class SRCNN(Chain):
    def __init__(self):
        super(SRCNN, self).__init__()
        f1,f2,f3 = 9,1,5
        n1,n2 = 64,32
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3, out_channels=n1, ksize=f1, stride=1,pad=f1//2)
            self.conv2 = L.Convolution2D(
                in_channels=n1, out_channels=n2, ksize=f2, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=n2, out_channels=3, ksize=f3, stride=1,pad=f3//2)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        return h