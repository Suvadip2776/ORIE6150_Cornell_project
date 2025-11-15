"""
Implementation of several voting theoretic axioms 

We implement the following axioms
* Anonymity
* Weak anonymity (a weakening of the usual anonymity axiom)
* Neutrality
* Weak neutrality (a weakening of the usual neutrality axiom)
* Unanimity
* Condorcet
* Pareto
* Independence

An axiom is a function mapping a rule and a profile (in the sense of 
the `pref_voting` package) to a value  in {-1,1,0}, where 0 means the 
axiom is not applicable, -1 means the axiom is violated (applicable 
and false) and 1 means the axiom is satisfied (applicable and true). 

We also add an optional `sample` argument. If it is given, not all 
permutations are checked but only `sample`-many are randomly checked.

(Each axiom also has a continuous version which is described in 
`axioms_continuous.py`. These output real numbers describing the 
degree to which the axiom is satisfied (for use in optimization).)
"""

from random import shuffle, randint, choice, sample
import itertools

import pref_voting
from pref_voting.profiles import Profile


def ax_anonymity(rule, profile, sample=None):
    """
    Implementation of the anonymity axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package), and optionally `sample` of type int. Recommended value for 
    `sample` is 50. Also see the dictionary `dict_axioms_sample` in `utils.py`.

    Output: 1 if for all (resp., `sample` many) permutations of voters, the
    output of the rule for the original profile is the same as for the permuted
    profile. -1 otherwise.
    """

    # Copy original profile
    original_profile = profile
    original_winners = set(rule(original_profile))
    if sample is None:
        # Build permuted profile
        profile_list = profile.rankings
        for permuted_profile_list in list(itertools.permutations(profile_list)):
            permuted_profile = Profile(permuted_profile_list)
            permuted_winners = set(rule(permuted_profile))
            # Check if rule on original and permuted profile agrees
            if original_winners == permuted_winners:
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction
    else:
        # Build permuted profile
        profile_list = profile.rankings
        for s in range(sample):
            shuffle(profile_list)
            permuted_profile = Profile(profile_list)
            permuted_winners = set(rule(permuted_profile))
            # Check if rule on original and permuted profile agrees
            if original_winners == permuted_winners:
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction


def ax_anonymity_weak(rule, profile, sample=None):
    """
    A weakened version of anonymity

    This is just like the anonymity axiom except that we only require that the
    permuted version of the original profile overlaps (as opposed to being
    identical) with the original profile, unless both of them are empty.
    """
    # Copy original profile
    original_profile = profile
    original_winners = set(rule(original_profile))
    if sample is None:
        # Build permuted profile
        profile_list = profile.rankings
        for permuted_profile_list in list(itertools.permutations(profile_list)):
            permuted_profile = Profile(permuted_profile_list)
            permuted_winners = set(rule(permuted_profile))
            # Check if (original and permuted winners OVERLAPS)
            # or (both original and permuted winners are both empty)
            if not original_winners.isdisjoint(permuted_winners) or (
                not original_winners and not permuted_winners
            ):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction
    else:
        # Build permuted profile
        profile_list = profile.rankings
        for s in range(sample):
            shuffle(profile_list)
            permuted_profile = Profile(profile_list)
            permuted_winners = set(rule(permuted_profile))
            # Check if (original and permuted winners OVERLAPS)
            # or (both original and permuted winners are both empty)
            if not original_winners.isdisjoint(permuted_winners) or (
                not original_winners and not permuted_winners
            ):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction


def ax_neutrality(rule, profile, sample=None):
    """
    Implementation of the neutrality axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package), and optionally `sample` of type int. Recommended value for 
    `sample` is 50. Also see the dictionary `dict_axioms_sample` in `utils.py`.

    Output: 1 if for all (resp., sample many) permutations of alternatives, the
    output of the rule for the permuted profile is the permuted output of the
    original profile. -1 otherwise.
    """

    # Copy profile as list of rankings
    profile_original = profile
    original_winners_list = rule(profile_original)
    profile_list = profile.rankings
    num_alternatives = profile.num_cands
    if sample is None:
        # consider all the permutations p of the alternatives
        for p in list(itertools.permutations(range(num_alternatives))):
            # build permuted list of rankings
            permuted_profile_list = []
            for ranking in profile_list:
                permuted_ranking = [p[alt] for alt in ranking]
                permuted_profile_list.append(permuted_ranking)
            profile_permuted = Profile(permuted_profile_list)
            winners_of_permuted_profile = rule(profile_permuted)
            # first permuting then voting should be the same as first voting
            # then permuting
            if set(winners_of_permuted_profile) == set(
                [p[alt] for alt in original_winners_list]
            ):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction
    else:
        for s in range(sample):
            # choose a permutation p of the alternatives
            p = list(range(num_alternatives))
            shuffle(p)
            # build permuted list of rankings
            permuted_profile_list = []
            for ranking in profile_list:
                permuted_ranking = [p[alt] for alt in ranking]
                permuted_profile_list.append(permuted_ranking)
            profile_permuted = Profile(permuted_profile_list)
            winners_of_permuted_profile = rule(profile_permuted)
            # first permuting then voting should be the same as first voting 
            # then permuting
            if set(winners_of_permuted_profile) == set(
                [p[alt] for alt in original_winners_list]
            ):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction


def ax_neutrality_weak(rule, profile, sample=None):
    """
    A weakened version of neutrality

    This is just like the neutrality axiom except that we only require that the
    permuted version overlaps (as opposed to being identical) with the original
    case, unless both of them are empty.
    """

    # Copy profile as list of rankings
    profile_original = profile
    original_winners_list = rule(profile_original)
    profile_list = profile.rankings
    num_alternatives = profile.num_cands
    if sample is None:
        # consider all the permutations p of the alternatives
        for p in list(itertools.permutations(range(num_alternatives))):
            # build permuted list of rankings
            permuted_profile_list = []
            for ranking in profile_list:
                permuted_ranking = [p[alt] for alt in ranking]
                permuted_profile_list.append(permuted_ranking)
            profile_permuted = Profile(permuted_profile_list)
            winners_of_permuted_profile = rule(profile_permuted)
            # first permuting then voting should be OVERLAPPING with first 
            # voting then permuting OR (both original and permuted winners are
            # both empty)
            if not set(winners_of_permuted_profile).isdisjoint(
                set([p[alt] for alt in original_winners_list])
            ) or (not original_winners_list and not winners_of_permuted_profile):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction
    else:
        for s in range(sample):
            # choose a permutations p of the alternatives
            p = list(range(num_alternatives))
            shuffle(p)
            # build permuted list of rankings
            permuted_profile_list = []
            for ranking in profile_list:
                permuted_ranking = [p[alt] for alt in ranking]
                permuted_profile_list.append(permuted_ranking)
            profile_permuted = Profile(permuted_profile_list)
            winners_of_permuted_profile = rule(profile_permuted)            
            # first permuting then voting should be OVERLAPPING with first 
            # voting then permuting OR (both original and permuted winners are
            # both empty)
            if not set(winners_of_permuted_profile).isdisjoint(
                set([p[alt] for alt in original_winners_list])
            ) or (not original_winners_list and not winners_of_permuted_profile):
                satisfaction = 1
                continue
            else:
                satisfaction = -1
                break
        return satisfaction


def ax_unanimity(rule, profile, sample=None):
    """
    Implementation of the unanimity axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package). The `sample` argument will not be used, it is only there so the 
    function is of the same type as the other axioms (a warning will be raised
    if sample is not None).

    Output: 1 if some alternative is first for all voters and this is the 
    single output/winner. -1 if some alternative is first for all voters and 
    this is not the single output/winner. 0 if there is no alternative that is
    first for every voter.
    """
    if sample is not None:
        print(
            "For the unanimity axiom it does not not make sense to choose \
    a (not None) sample, since no permutations are computed which could \
    be sampled. We continue and ignore it."
        )
    # Write input as a list
    profile_list = profile.rankings
    # collect the first alternatives
    firsts = [ranking[0] for ranking in profile_list]
    if len(set(firsts)) == 1:
        if set(rule(profile)) == set(firsts):
            satisfaction = 1
        else:
            satisfaction = -1
    else:
        satisfaction = 0
    return satisfaction


def ax_condorcet(rule, profile, sample=None):
    """
    Implementation of the condorcet axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package). The `sample` argument will not be used, it is only there so the
    function is of the same type as the other axioms (a warning will be raised
    if sample is not None).

    Output: 1 if some alternative beats all other alternatives in a pairwise
    majority contest and this is the single output/winner. -1 if some 
    alternative beats all other alternatives in a pairwise majority contest and
    this is not the single output/winner.  0 if there is no alternative that 
    beats all other alternatives in a pairwise majority contest.
    """
    if sample is not None:
        print(
            "For the condorcet axiom it does not not make sense to choose \
    a (not None) sample, since no permutations are computed which could \
    be sampled. We continue and ignore it."
        )
    # compute condorcet winner with *pref_voting*
    a = profile.condorcet_winner()
    if a is not None:
        if set(rule(profile)) == {a}:
            satisfaction = 1
        else:
            satisfaction = -1
    else:
        satisfaction = 0
    return satisfaction


def ax_pareto(rule, profile, sample=None):
    """
    Implementation of the pareto axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package). The `sample` argument will not be used, it is only there so the 
    function is of the same type as the other axioms (a warning will be raised
    if sample is not None).

    Output: 1 if for all alternatives x and y, if each voter prefers x over y,
    then y should not win. -1 if for some alternatives x and y, each voter
    prefers x over y and y wins. 0 otherwise.
    """
    if sample is not None:
        print(
            "For the condorcet axiom it does not not make sense to choose \
    a (not None) sample, since no permutations are computed which could \
    be sampled. We continue and ignore it."
        )
    # Name the relevant parts of the input
    num_alternatives = profile.num_cands
    profile_list = profile.rankings
    # Quantify over all possible alternatives a and b
    satisfaction = 0
    for a in range(num_alternatives):
        for b in range(num_alternatives):
            # Check if each voters ranks a over b, i.e., a has lower index in 
            # the ranking submitted by the voter than the index of b
            if all(ranking.index(a) < ranking.index(b) for ranking in profile_list):
                if b not in set(rule(profile)):
                    satisfaction = 1
                else:
                    satisfaction = -1
    return satisfaction


def ax_independence(rule, profile, sample=None):
    """
    Implementation of the independence axiom

    Input: `rule` and `profile` (both in the sense of the `pref_voting` 
    package), and optionally `sample` of type int. If sample is given, then 
    num_winners (num_cand - num_winners) sample**sample many permutations 
    are checked. Recommended value for `sample` is 4. Also see the dictionary
    `dict_axioms_sample` in `utils.py`.

    The profile is assumed to be such that num_rankings == num_voters, 
    otherwise apply 'recast_profile_wo_mult'.

    Output: 1 if for every alternative x that is a winner (i.e., is in the set 
    of winners selected by the rule) and y that is a loser (i.e., not a 
    winner), and for all permutations over the remaining alternatives for each
    voter, y is a loser in the permuted profile. -1 if for some alternatives x
    that is a winner and y that is a loser, and for some permutation over the
    remaining alternatives, y is a winner in the permuted profile. 0 otherwise
    (i.e., if all alternatives are winners in the original profile or none 
    are).
    """
    original_profile_list = profile.rankings
    # first compute the winners and losers
    winners = set(rule(profile))
    losers = set(profile.candidates) - winners
    # if all alternatives are winners, satisfaction is 0
    # (typically voting rules produce always at least one winner, but the rules
    # learned by neural networks may fail this, so check for that, too)
    if winners == set(profile.candidates) or winners == set([]):
        satisfaction = 0
        return satisfaction
    # so from now on, there are some losers
    else:
        if sample is None:
            # we now consider all ways of building new rankings where the set
            # of voters raking a above b is the same
            for a in winners:
                for b in losers:  # so a != b
                    # For each voter, build the list of rankings that respect
                    # the order of a and b
                    allowed_rankings = []
                    for ranking in original_profile_list:
                        possible_rankings = [
                            p
                            for p in list(
                                itertools.permutations(range(profile.num_cands))
                            )
                            if (p.index(a) > p.index(b))
                            == ((ranking.index(a) > ranking.index(b)))
                        ]
                        allowed_rankings.append(possible_rankings)
                    # For every way of assigning an allowed ranking to each
                    # voter...
                    for choice_of_rankings in list(
                        itertools.product(*allowed_rankings)
                    ):
                        new_profile = Profile(choice_of_rankings)
                        # ... check that b is a loser in permuted profile
                        if b in set(rule(new_profile)):
                            satisfaction = -1
                            return satisfaction
                        else:
                            satisfaction = 1
                            continue
            return satisfaction

        if sample is not None:
            # we now randomly sample *sample*-many new rankings where the set
            # of voters raking a above b is the same
            for a in winners:
                for b in losers:  # so a != b
                    # For each voter (or, rather, corresponding ranking), build
                    # the list of rankings that respect the order of a and b,
                    # then collect them as a list
                    allowed_rankings = []
                    for ranking in original_profile_list:
                        possible_rankings = []
                        while len(possible_rankings) < sample:
                            p = list(range(profile.num_cands))
                            shuffle(p)
                            if (p.index(a) > p.index(b)) == (
                                (ranking.index(a) > ranking.index(b))
                            ):
                                possible_rankings.append(p)
                        allowed_rankings.append(possible_rankings)
                    # sample^sample many times sample an allowed choice of
                    # ranking...
                    for i in range(int(pow(sample, sample))):
                        choice_of_rankings = []
                        for possible_rankings in allowed_rankings:
                            choice_of_rankings.append(choice(possible_rankings))
                        new_profile = Profile(choice_of_rankings)
                        # and check if b is a loser in permuted profile
                        if b in set(rule(new_profile)):
                            satisfaction = -1
                            return satisfaction
                        else:
                            satisfaction = 1
                            continue
            return satisfactionk