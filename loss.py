import theano
import theano.tensor as T


def policy_loss(values, a_probs, norm=True, entropy_coeff=.0):
    bias = T.sum(a_probs*values, axis=1, keepdims=True)
    adv = (values - bias)
    if norm:
        adv /= (T.abs_(bias) + 1e-8)
    adv = theano.gradient.disconnected_grad(adv)
    objective = a_probs * adv
    entropy = -1. * T.sum(T.log(a_probs + 1e-8) * a_probs, axis=1, keepdims=True)
    actor_loss = -1. * T.mean(objective + entropy_coeff*entropy, axis=-1)
    return actor_loss


def value_softmax(values, a_probs, norm=True, norm_coeff=10):
    val_max = T.max(values, axis=1, keepdims=True)
    if norm:
        val_min = T.min(values, axis=1, keepdims=True)
        values = 0.5 + (values - val_min) / 2. / (val_max - val_min + 1e-8)
    else:
        values = (values - val_max)
    values /= norm_coeff
    targets = T.nnet.softmax(values)
    return T.mean(T.nnet.categorical_crossentropy(a_probs, targets), axis=-1)
