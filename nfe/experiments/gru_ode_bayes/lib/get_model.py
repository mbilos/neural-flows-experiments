from nfe.experiments.gru_ode_bayes.lib.gru_ode_bayes import NNFOwithBayesianJumps


def get_gob_model(input_size, args, cov_size, gob_p_hidden, gob_prep_hidden, mixing, gob_cov_hidden):

    model = NNFOwithBayesianJumps(input_size=input_size, args=args,
                                  p_hidden=gob_p_hidden, prep_hidden=gob_prep_hidden,
                                  mixing=mixing, cov_size=cov_size, cov_hidden=gob_cov_hidden)

    return model
