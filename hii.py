import tensorflow as tf
from tensorflow.keras import activations, initializers, backend
class EdgeNetwork(tf.keras.layers.Layer):
    """ Submodule for Message Passing """

    def __init__(self,
                 n_pair_features=8,
                 n_hidden=100,
                 init='glorot_uniform',
                 **kwargs):
        super(EdgeNetwork, self).__init__(**kwargs)
        self.n_pair_features = n_pair_features
        self.n_hidden = n_hidden
        self.init = init

    def get_config(self):
        config = super(EdgeNetwork, self).get_config()
        config['n_pair_features'] = self.n_pair_features
        config['n_hidden'] = self.n_hidden
        config['init'] = self.init
        return config

    def build(self, input_shape):

        def init(input_shape):
            return self.add_weight(name='kernel',
                                   shape=(input_shape[0], input_shape[1]),
                                   initializer=self.init,
                                   trainable=True)

        n_pair_features = self.n_pair_features
        n_hidden = self.n_hidden
        self.W = init([n_pair_features, n_hidden * n_hidden])
        print(self.W.shape)
        self.b = tf.zeros(shape=(n_hidden * n_hidden,), dtype=tf.dtypes.float32)
        print(self.b.shape)
        self.built = True

    def call(self, inputs):
        pair_features, atom_features, atom_to_pair = inputs
        print(pair_features.shape)
        print(self.W.shape)
        print(self.b.shape)
        p = tf.matmul(pair_features, self.W)
        print(p.shape)
        A = tf.nn.bias_add(p, self.b)
        print(A.shape)
        A = tf.reshape(A, (-1, self.n_hidden, self.n_hidden))
        print(A.shape)
        l = tf.gather(atom_features, atom_to_pair[:, 1])
        print(l.shape)
        out = tf.expand_dims(l, axis=2)
        print(out.shape.as_list())
        out = tf.squeeze(tf.matmul(A, out), axis=2)
        print(out.shape)
        print(atom_to_pair[:,0].shape)
        print("hello")
        t = tf.math.segment_sum(out, atom_to_pair[:, 0])
        print(t.shape)
        print(tf.math.segment_sum(out, atom_to_pair[:, 0]))
        return tf.math.segment_sum(out, atom_to_pair[:, 0])

pair_features = tf.random.uniform(shape=[868, 14])
atom_features = tf.random.uniform(shape=[2945, 75])
atom_to_pair = tf.sort(tf.random.uniform(shape=[868, 2], dtype=tf.dtypes.int32, maxval=100), axis=0)
inputs = [pair_features, atom_features, atom_to_pair]

n_pair_features = 14
n_hidden = 75
init = 'glorot_uniform'
layer = EdgeNetwork(n_pair_features, n_hidden, init)
t = layer(inputs)
print("riya")
print(t.shape)
print(t)
