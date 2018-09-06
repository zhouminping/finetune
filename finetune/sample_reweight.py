import tensorflow as tf


def learning_to_reweight(gold_data, gold_targets, data, targets, model, lr=1e-5):
    # lines 405 initial forward pass to compute the initial weighted loss
    def meta_net(data, targets):
        with tf.variable_scope("meta_net", reuse=tf.AUTO_REUSE):
            return model(data, targets)

    # Lines 4 - 5 initial forward pass to compute the initial weighted loss

    y_f_hat_meta, cost_meta = meta_net(data, targets)

    meta_net_vars = tf.global_variables(scope="meta_net")
    net_vars = [var for var in tf.global_variables() if "meta_net" not in var.name]

    re_init_vars = []
    for n_v, met_v in zip(net_vars, meta_net_vars):
        re_init_vars.append(met_v.assign(n_v))

    with tf.control_dependencies(re_init_vars):
        cost_meta = tf.identity(cost_meta)

    eps = tf.zeros_like(cost_meta)
    l_f_meta = tf.reduce_sum(cost_meta * eps)

    # Line 6 perform a parameter update
    grads = tf.gradients(l_f_meta, meta_net_vars)
    patch_dict = dict()
    for grad, var in zip(grads, meta_net_vars):
        if grad is None:
            print("None grad for variable {}".format(var.name))
        else:
            patch_dict[var.name] = -grad * lr

    # Monkey patch get_variable
    old_get_variable = tf.get_variable

    def _get_variable(*args, **kwargs):
        var = old_get_variable(*args, **kwargs)
        return var + patch_dict.get(var.name, 0.0)

    tf.get_variable = _get_variable

    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
    y_g_hat, l_g_meta = meta_net(gold_data, gold_targets)

    tf.get_variable = old_get_variable
    grad_eps_es = tf.gradients(l_g_meta, eps)[0]

    # Line 11 computing and normalizing the weights
    w_tilde = tf.maximum(-grad_eps_es, 0.)
    norm_c = tf.reduce_sum(w_tilde)

    w = w_tilde / (norm_c + tf.cast(tf.equal(norm_c, 0.0), dtype=tf.float32))

    return tf.stop_gradient(w)
