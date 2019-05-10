import numpy as np
import theano, theano.tensor as T
import lasagne
from collections import OrderedDict
# import tensorflow as tf


def utils_floatX(arr):
    return np.asarray(arr, dtype=theano.config.floatX)


def adam_update(grads, params, learning_rate=0.001, beta1=0.9,
                beta2=0.999, epsilon=1e-8):

    t_prev = theano.shared(utils_floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param + step

    updates[t_prev] = t
    return updates


class PGLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = T.tensor4('states')
        actions = T.ivector('actions')
        values = T.vector('values')

        # print 'network_input_height=', pa.network_input_height
        # print 'network_input_width=', pa.network_input_width
        # print 'network_output_dim=', pa.network_output_dim

        # image representation
        self.l_out = \
            build_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)



        # compact representation
        # self.l_out = \
        #     build_compact_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        params = lasagne.layers.helper.get_all_params(self.l_out)

        #print ' params=', params, ' count=', lasagne.layers.count_params(self.l_out)

        self._get_param = theano.function([], params)

        # ===================================
        # training function part
        # ===================================

        prob_act = lasagne.layers.get_output(self.l_out, states) ## get the output of network

        self._get_act_prob = theano.function([states], prob_act, allow_input_downcast=True)

        # --------  Policy Gradient  --------

        N = states.shape[0]

        loss = T.log(prob_act[T.arange(N), actions]).dot(values) / N  # call it "loss"

        grads = T.grad(loss, params) # gradients of loss with respect to params

        # updates = rmsprop_updates(
        #     grads, params, self.lr_rate, self.rms_rho, self.rms_eps)

        updates = adam_update(
            grads, params, self.lr_rate)

        self._train_fn = theano.function([states, actions, values], loss,
                                         updates=updates, allow_input_downcast=True)

        self._get_loss = theano.function([states, actions, values], loss, allow_input_downcast=True)

        self._get_grad = theano.function([states, actions, values], grads, allow_input_downcast=True)

        # --------  Supervised Learning  --------

        su_target = T.ivector('su_target')

        # su_diff = su_target - prob_act
        # su_loss = 0.5 * su_diff ** 2

        su_loss = lasagne.objectives.categorical_crossentropy(prob_act, su_target)
        su_loss = su_loss.mean()

        l2_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l2)
        # l1_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l1)

        su_loss += 1e-3*l2_penalty
        print 'lr_rate=', self.lr_rate

        su_updates = lasagne.updates.rmsprop(su_loss, params,
                                             self.lr_rate, self.rms_rho, self.rms_eps)
        #su_updates = lasagne.updates.nesterov_momentum(su_loss, params, self.lr_rate)

        self._su_train_fn = theano.function([states, su_target], [su_loss, prob_act], updates=su_updates)

        self._su_loss = theano.function([states, su_target], [su_loss, prob_act])

        self._debug = theano.function([states], [states.flatten(2)])

    # get the action based on the estimated value
    def choose_action(self, state): ## this function is  only called by slow_down_cdf.py

        act_prob = self.get_one_act_prob(state)

        csprob_n = np.cumsum(act_prob)
        act = (csprob_n > np.random.rand()).argmax()

        #print("action probality", act_prob,"action", act)

        return act

    def train(self, states, actions, values):

        loss = self._train_fn(states, actions, values)
        return loss

    def get_params(self):

        return self._get_param()

    def get_grad(self, states, actions, values):

        return self._get_grad(states, actions, values)

    def get_one_act_prob(self, state):

        states = np.zeros((1, 1, self.input_height, self.input_width), dtype=theano.config.floatX)
        states[0, :, :] = state
        act_prob = self._get_act_prob(states)[0]

        return act_prob

    def get_act_probs(self, states):  # multiple states, assuming in floatX format
        act_probs = self._get_act_prob(states)
        return act_probs

    #  -------- Supervised Learning --------
    def su_train(self, states, target):
        loss, prob_act = self._su_train_fn(states, target)
        return np.sqrt(loss), prob_act

    def su_test(self, states, target):
        loss, prob_act = self._su_loss(states, target)
        return np.sqrt(loss), prob_act

    #  -------- Save/Load network parameters --------
    def return_net_params(self):
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_net_params(self, net_params):
        lasagne.layers.helper.set_all_param_values(self.l_out, net_params)


class CriticLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = T.tensor4('states')
        actions = T.ivector('actions')
        values = T.vector('values')

















# ===================================
# build actor neural network
# ===================================


def build_pg_network(input_height, input_width, output_length):


    l_in = lasagne.layers.InputLayer(
        shape=(None, 1, input_height, input_width),
    )

    l_hid = lasagne.layers.DenseLayer(
        l_in,
        num_units=200,
        # nonlinearity=lasagne.nonlinearities.tanh,
        nonlinearity=lasagne.nonlinearities.rectify,
        # W=lasagne.init.Normal(.0201),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hid,
        num_units=output_length,
        nonlinearity=lasagne.nonlinearities.softmax,
        # W=lasagne.init.Normal(.0001),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    return l_out

# ===================================
# build critic neural network
# ===================================
def build_critic_network(input_height, input_width):


    l_1 = lasagne.layers.InputLayer(
        shape=(None, 1, input_height, input_width),
    )

    l_2 = lasagne.layers.DenseLayer(
        l_1,
        num_units=500,
        # nonlinearity=lasagne.nonlinearities.tanh,
        nonlinearity=lasagne.nonlinearities.rectify,
        # W=lasagne.init.Normal(.0201),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    Vlu = lasagne.layers.DenseLayer(
        l_2,
        num_units=1,
        nonlinearity=None,
        # W=lasagne.init.Normal(.0001),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    return Vlu

