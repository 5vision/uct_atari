import numpy as np
import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict


def get_adam_steps_and_updates(all_grads, params, learning_rate=0.001, beta1=0.9,
                               beta2=0.999, epsilon=1e-8):
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    adam_steps = []
    for param, g_t in zip(params, all_grads):
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

        adam_steps.append(step)

    updates[t_prev] = t
    return adam_steps, updates


def _build_model(state_shape, num_act):
    # input layer
    l_input = lasagne.layers.InputLayer([None] + list(state_shape))

    l_conv = l_input
    '''
    for i in xrange(4):
        l_conv = lasagne.layers.Conv2DLayer(
            l_conv, 32, (3, 3), (2, 2),
            pad='same',
            nonlinearity=lasagne.nonlinearities.elu
        )
    '''
    l_conv = lasagne.layers.Conv2DLayer(
        l_conv, 32, (8, 8), (4, 4),
        W=lasagne.init.HeUniform(),  # Defaults to Glorot
        b=lasagne.init.Constant(.1),
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l_conv = lasagne.layers.Conv2DLayer(
        l_conv, 64, (4, 4), (2, 2),
        W=lasagne.init.HeUniform(),  # Defaults to Glorot
        b=lasagne.init.Constant(.1),
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l_conv = lasagne.layers.Conv2DLayer(
        l_conv, 64, (3, 3), (1, 1),
        W=lasagne.init.HeUniform(),  # Defaults to Glorot
        b=lasagne.init.Constant(.1),
        nonlinearity=lasagne.nonlinearities.rectify
    )

    l_fc = lasagne.layers.ReshapeLayer(l_conv, ([0], -1))
    l_fc = lasagne.layers.DenseLayer(
            l_fc,
            num_units=512,
            W=lasagne.init.HeUniform(),  # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            nonlinearity=lasagne.nonlinearities.rectify
    )
    l_fc = lasagne.layers.DenseLayer(
            l_fc,
            num_units=256,
            W=lasagne.init.HeUniform(),  # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            nonlinearity=lasagne.nonlinearities.rectify
    )

    # actor network
    l_actor = lasagne.layers.DenseLayer(
        l_fc,
        num_units=num_act,
        W=lasagne.init.Normal(),
        nonlinearity=lasagne.nonlinearities.softmax
    )

    # critic network
    l_critic = lasagne.layers.DenseLayer(
        l_fc,
        num_units=1,
        nonlinearity=None
    )

    return l_actor, l_critic


def build_model_checkpoints(state_shape, num_act,
                            critic_loss_coeff=0.5,
                            entropy_coeff=0.001,
                            learning_rate=0.0001):
    # input tensors
    states = T.tensor4('states')
    q_vals = T.matrix('q_vals')

    l_actor, l_critic = _build_model(state_shape, num_act)

    # calculate prediction
    a_probs = lasagne.layers.get_output(l_actor, states)
    v_vals = lasagne.layers.get_output(l_critic, states)

    # CRITIC
    v_target = T.sum(q_vals*a_probs, axis=1, keepdims=True)
    v_target = theano.gradient.disconnected_grad(v_target)
    td_error = v_target - v_vals
    critic_loss = 0.5 * (td_error ** 2)
    critic_loss = T.sum(critic_loss)

    # ACTOR
    adv = q_vals - v_target
    objective = a_probs*adv
    entropy = -1. * T.sum(T.log(a_probs + 1e-8) * a_probs, axis=1, keepdims=True)
    actor_loss = -1. * T.sum(objective + entropy_coeff*entropy)

    # total loss
    total_loss = actor_loss + critic_loss_coeff*critic_loss

    # combine params
    actor_params = lasagne.layers.get_all_params(l_actor)
    crit_params = lasagne.layers.get_all_params(l_critic)
    params = [p for p in crit_params if p not in actor_params] + actor_params

    # calculate grads and steps
    grads = T.grad(total_loss, params)
    grads = lasagne.updates.total_norm_constraint(grads, 10)
    steps, updates = get_adam_steps_and_updates(grads, params, learning_rate)
    steps_fn = theano.function([states, q_vals], steps, updates=updates)

    prob_fn = theano.function([states], a_probs)
    val_fn = theano.function([states], v_vals)

    return steps_fn, prob_fn, val_fn, params


def build_model(state_shape, num_act,
                critic_loss_coeff=0.5,
                entropy_coeff=0.001,
                learning_rate=0.0001):
    # input tensors
    states = T.tensor4('states')
    v_targets = T.vector('v_target')
    actions = T.ivector('actions')

    l_actor, l_critic = _build_model(state_shape, num_act)

    # calculate prediction
    a_probs = lasagne.layers.get_output(l_actor, states)
    v_vals = lasagne.layers.get_output(l_critic, states)
    v_vals = T.flatten(v_vals)

    # CRITIC
    td_error = v_targets - v_vals
    critic_loss = 0.5 * (td_error ** 2)
    critic_loss = T.sum(critic_loss)

    # ACTOR
    # entropy terms
    log_prob_all = T.log(a_probs + 1e-8)
    entropy = -1. * T.sum(log_prob_all * a_probs, axis=1)
    # objective part
    batch_size = states.shape[0]
    log_prob = log_prob_all[T.arange(batch_size), actions]
    adv = theano.gradient.disconnected_grad(td_error)
    actor_loss = -1. * (log_prob * adv + entropy_coeff * entropy)
    actor_loss = T.sum(actor_loss)

    # total loss
    total_loss = actor_loss + critic_loss_coeff*critic_loss

    # combine params
    actor_params = lasagne.layers.get_all_params(l_actor)
    crit_params = lasagne.layers.get_all_params(l_critic)
    params = [p for p in crit_params if p not in actor_params] + actor_params

    # calculate grads and steps
    grads = T.grad(total_loss, params)
    grads = lasagne.updates.total_norm_constraint(grads, 10)
    steps, updates = get_adam_steps_and_updates(grads, params, learning_rate)
    steps_fn = theano.function([states, v_targets, actions], steps, updates=updates)

    prob_fn = theano.function([states], a_probs)
    val_fn = theano.function([states], v_vals)

    return steps_fn, prob_fn, val_fn, params
