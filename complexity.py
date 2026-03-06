def maat_score(H, B, S, V, R):

    epsilon = 1e-9

    return (H * B * S * V * (1/(1+R))) + epsilon