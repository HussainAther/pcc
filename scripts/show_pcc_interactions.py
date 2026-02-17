#!/usr/bin/env python3

from metrics.pcc_payoff import pcc_payoff, print_interactions, dominance_matrix


def main():

    A = pcc_payoff(mu=1.0, sigma0=1.2)

    print("\nPayoff matrix:\n")
    print(A)

    print()
    print_interactions(A)

    D = dominance_matrix(A)

    print("Dominance adjacency matrix:\n")
    print(D.astype(int))


if __name__ == "__main__":
    main()

