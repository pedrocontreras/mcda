def parameter_running(iter,iter_stochastic):
    iter_stochastic = iter_stochastic
    iter = km_iter(50)
    return iter,iter_stochastic

def parameter_outranking(lam,beta):
    lam = lambda_parameter(0.5)
    beta = beta_parameter(0.1)
    return lam,beta

def lambda_parameter(lam):
    return lam

def beta_parameter(beta):
    return beta

def stochastic_iter(iter_stochastic):
    return iter_stochastic

def km_iter(iter):
    return iter

def p_dir_param(p_dir):
    return p_dir

def q_dir_param(q_dir):
    return q_dir

def p_inv_param(p_inv):
    return p_inv

def q_inv_param(q_inv):
    return q_inv

def get_umbrales(pd,qd,pi,qi):
    """
    umbrales de preferencia directos e inversos de cada criterio
    :return: p_dir, q_dir, p_inv, q_inv
    """

    p_dir = p_dir_param(pd)
    q_dir = q_dir_param(qd)
    p_inv = p_inv_param(pi)
    q_inv = q_inv_param(qi)

    return p_dir, q_dir, p_inv, q_inv

def get_weights(w):
    """
    get criteria weights
    :return: w
    """
    return w
