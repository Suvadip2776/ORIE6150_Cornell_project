"""
Continuous version of the voting-theoretic axioms in `axioms.py`

We implement the following continuous axioms:
* Continuous anonymity
* Continuous neutrality
* Continuous unanimity
* Continuous condorcet (two versions)
* Continuous pareto (two versions)
* Continuous independence
Plus axioms on admissibility ensuring that the outputted winners
are of the right form:
* discouraging choosing no winners at all
* discouraging calling all alternatives winners
* discouraging inadmissable winners (those the don't occur in the 
  inputted profile)
* encouraging resoluteness (i.e., picking only few winners)


In `axioms.py`, we implemented various voting-theoretic axioms. They take 
as input a rule and a profile (in the sense of the `pref_voting` package) 
and output either -1, 1, or 0, depending on whether for this rule and 
profile the axiom is violated, satisfied, or not applicable.

However, for use in optimization we need continuous/differentiable versions.
Specifically, we want to use the axioms as a (semantic) loss function, so 
neural networks can, during training, also optimize axiom satisfaction 
and not just the match with the data. But for this we need a continuous/
differentiable version of the axioms, so they can indeed serve as loss 
functions. Here we formulate the axioms as loss functions that pytorch 
can use.

The idea is that, given a neural network/model (realizing a voting rule) 
and an input profile X (or batch thereof), the continuous version of an 
axiom outputs a continuous number indicating how close the realized rule 
at the inputted profile is to satisfying the axiom. 

For example, for the axiom of anonymity: 
* we consider the input profile X, 
* compute the predicted winning set (as logits) y, 
* then we permute the voters in the profile X to obtain a permuted 
  profile X', 
* we again compute the predicted winning set (as logits) y', 
* and then we check the distance between y and y': the smaller this 
  distance, the more the axiom is satisfied.

Formally, we compute distances between predicted logits using the 
Kullback-Leibler divergence for logits (called KLD resp KLD0 below).
"""

from random import shuffle, sample
import torch
from torch import nn

import pref_voting
from pref_voting.profiles import Profile

# Some distance function that we use:
KLD = lambda x, y : nn.KLDivLoss(log_target=True, reduction='batchmean')(x.log_softmax(dim=1), y.log_softmax(dim=1))
L2 = lambda x, y: (1/len(x))*sum(nn.PairwiseDistance(p=2)(x,y))
cos_sim = lambda x, y: (1/len(x))*sum(nn.CosineSimilarity(dim=1, eps=1e-8)(x,y))



# ADMISSABILITY


def ax_no_winners_cont(model_on_profiles, X):
    # Compute prediction of model
    predictions = model_on_profiles(X)
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        prof = X[i]
        logits = predictions[i]
        ad_cands = tuple(prof.candidates) 
        ad_logits = logits[ad_cands,]
        # We want at least one of the numbers in 
        # torch.sigmoid(ad_logits)
        # to be above 0.5, i.e., the maximum norm should be above 0.5.
        # So the loss, i.e., how much this is not fulfilled, is 
        # 0.5 - maximum norm, and 0 if 0.5 < maximum norm   
        max_norm = torch.linalg.norm(torch.sigmoid(ad_logits), float('inf'))
        loss += torch.maximum(0.5 - max_norm , torch.zeros(1).squeeze())
    return (1/len(X))*loss


def ax_all_winners_cont(model_on_profiles, X, distance):
    """
    Outputs a loss that is high if all (admissible) alternatives are winners
    
    Idea: The model output should be far away from declaring all admissible 
    alternatives winners. So the loss is the inverse distance to the vector 
    where all alternatives are winners. 
    """
    # Compute prediction of model
    predictions = model_on_profiles(X)
    max_num_alternatives = len(predictions[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        # The vector of all admissible winners for the i-th profile:
        all_winners = torch.tensor(
            [1 if idx in X[i].candidates else 0 for idx in range(max_num_alternatives)],
            dtype=torch.float
        )
        # So we define the loss as the inverse distance to all_winners
        # (With [None, :] we add a singleton batch dimension
        # to fit the type of the distance function)
        loss += 1/distance(predictions[i][None, :], all_winners[None, :])
    return (1/len(X))*loss



def ax_inadmissibility_cont(model_on_profiles, X):
    # Compute prediction of model
    predictions = model_on_profiles(X)
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        prof = X[i]
        logits = predictions[i]
        inad_cands = tuple(
            [i for i in range(len(logits)) if i not in prof.candidates]
        ) 
        inad_logits = logits[inad_cands,]
        # want the following to be as close to zero as possible
        loss += torch.linalg.vector_norm(torch.sigmoid(inad_logits)) 
    return (1/len(X))*loss


def ax_resoluteness_cont(model_on_profiles, X): 
    """
    Outputs a loss describing how non-resolute the model is
    
    Idea: Looking at the logits outputted by the model for an inputted 
    profile, transform it with softmax into a probability distribution. 
    If that distribution has high (Shannon) entropy, it is very 
    'spread out', i.e., picks many winners. If it has low entropy, it 
    is concentrated on few winners, i.e., is resolute. So the output 
    of the loss function is the average entropy. 
    """
    # Compute prediction of model
    predictions = model_on_profiles(X)
    #Turn into probabilities
    probabilities = torch.nn.Softmax(dim=1)(predictions)
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(probabilities)):
        # Compute the (Shannon) entropy of the i-th probability
        loss += torch.sum(- probabilities[i] * torch.log(probabilities[i]))
    return (1/len(X))*loss






def ax_parity_cont(model_on_profiles, X):
    """
    The frequency (across a wide range of profiles) with which 
    a given alternative is called a winner should be the same for all 
    alternatives
    """
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    # For each alternative, compute its frequency
    frequencies = []
    for i in range(max_num_alternatives):
        # across all profiles (which is dim 0) look at the prediction
        # for the i-th alternative and take the average value of these
        average = torch.sum(original_prediction[:, i])/len(original_prediction)
        frequencies.append(average)
    # recast the list as a tensor
    frequencies = torch.stack(frequencies, dim=0)
    # We want the frequencies to be close together,
    # so they should have low variance
    loss = torch.var(frequencies)
    return loss



# AXIOMS


def ax_anonymity_cont(model_on_profiles, X, num_samples, distance):
    """
    Implementing the continuous version of the anonymity axiom.

    Input: 
    * A `model_on_profiles` which here is assumed to be a function 
      that takes a list of profiles as input and outputs a tensor 
      of the logits predicted for each profile
    * a list `X` of profiles (where each profile has at most as 
      many voters and alternatives as the model can handle)
    * a positive integer `num_samples` describing how many permutations 
      are sampled on which the axiom is checked. Recommended value 
      is 50. Also see the dictionary `dict_axioms_sample` 
      in `utils.py`.
    * a `distance` function which takes as input two tensors of 
      dimension [batches,predictions] and outputs the distance 
      between them.
      
    Output: Real number (as tensor) describing the average distance 
    between prediction on original profiles and permuted profiles.
    """
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for _ in range(num_samples):
        # Permute each profile in X
        X_permuted = [Profile(sample(profile.rankings, len(profile.rankings)))
                      for profile in X]
        # Compute prediction on permuted profiles
        permuted_prediction = model_on_profiles(X_permuted)
        # Compute distance between original and permuted prediction
        # and add it to the loss
        loss += distance(original_prediction, permuted_prediction)
    # return average loss
    return (1/num_samples)*loss



def ax_neutrality_cont(model_on_profiles, X, num_samples, distance):
    """
    Implementing the continuous version of the neutrality axiom.

    Input: 
    * A `model_on_profiles` which here is assumed to be a function 
      that takes a list of profiles as input and outputs a tensor 
      of the logits predicted for each profile
    * a nonempty list `X` of profiles (where each profile has at 
      most as many voters and alternatives as the model can handle)
    * a positive integer `num_samples` describing how many permutations 
      are sampled on which the axiom is checked. Recommended value 
      is 50. Also see the dictionary `dict_axioms_sample` 
      in `utils.py`.
    * a `distance` function which takes as input two tensors of 
      dimension [batches,predictions] and outputs the distance 
      between them.
      
    Output: Real number (as tensor) describing the average distance 
    between prediction on original profiles and permuted profiles.
    """
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for _ in range(num_samples):
        # (1) Compute the result of first_permuting_then_voting
        # Permute each profile in X
        list_of_permutations = []
        X_permuted = []
        for prof in X:
            # choose a permutation p of the alternatives
            p = sample(list(range(prof.num_cands)), prof.num_cands)
            # keep the permutation for later in the following format
            list_of_permutations.append(tuple(
                p + list(range(len(p),max_num_alternatives))
            ))
            permuted_prof = Profile(
                [[p[alt] for alt in ranking] for ranking in prof.rankings]
            )
            X_permuted.append(permuted_prof)
        first_permuting_then_voting = model_on_profiles(X_permuted)
        # (2) Now compute first_voting_then_permuting, initialized as
        first_voting_then_permuting = torch.zeros_like(original_prediction)
        for i in range(len(X)):
            #print(original_prediction[i])
            #print(list_of_permutations[i])
            first_voting_then_permuting[i] = original_prediction[i][list_of_permutations[i],]
            #print(first_voting_then_permuting[i])
        # (3) Compute distance between original and permuted version
        loss += distance(first_permuting_then_voting, first_voting_then_permuting)
        #print(loss)
    # return average loss
    return (1/num_samples)*loss







def ax_condorcet1_cont(model_on_profiles, X, distance):
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        # Check if the i-th profile has a condorcet winner 
        # (computed with `pref_voting`)
        w = X[i].condorcet_winner()
        if w is not None: # so w is a condorcet winner 
            # Then the ideal answer is
            correct_prediction = torch.tensor(
                [1 if idx==w else 0 for idx in range(max_num_alternatives)],
                dtype=torch.float
            )
            # Compute how closely the model predicted this
            # (With [None, :] we add a singleton batch dimension 
            # to fit the type of the distance function)
            loss += distance(
                original_prediction[i][None, :], 
                correct_prediction[None, :]
            )
    return (1/len(X)) * loss


def ax_condorcet2_cont(model_on_profiles, X, distance):
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        # Check if the i-th profile has a condorcet winner 
        # (computed with `pref_voting`)
        w = X[i].condorcet_winner()
        if w is not None: # so w is a condorcet winner 
            # Then the model should assign high probability to w being 
            # the winner, i.e., low probability for it not winning
            loss += 1 - torch.sigmoid(original_prediction[i][w])
    return (1/len(X)) * loss


def ax_pareto1_cont(model_on_profiles, X, distance):
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        # Check if the i-th profile has alternatives a and b 
        # such that each voter prefers a over b
        for a in X[i].candidates:
            for b in X[i].candidates:
                if all(ranking.index(a) < ranking.index(b) for ranking in X[i].rankings):
                    # If so, then the model's prediction should be 
                    # as far a way as possible from the 
                    anti_correct_prediction = torch.tensor(
                        [1 if idx==b else 0 for idx in range(max_num_alternatives)],
                        dtype=torch.float
                    )
                    # So we define the loss as the inverse distance
                    # (With [None, :] we add a singleton batch dimension 
                    # to fit the type of the distance function)
                    loss += 1/distance(
                        original_prediction[i][None, :], 
                        anti_correct_prediction[None, :]
                    )
    return (1/len(X)) * loss



def ax_pareto2_cont(model_on_profiles, X, distance):
    # Compute prediction of model
    original_prediction = model_on_profiles(X)
    max_num_alternatives = len(original_prediction[0])
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for i in range(len(X)):
        # Check if the i-th profile has alternatives a and b 
        # such that each voter prefers a over b
        for a in X[i].candidates:
            for b in X[i].candidates:
                if all(ranking.index(a) < ranking.index(b) for ranking in X[i].rankings):
                    # If so, the model's probability for taking b 
                    # to be a winner should be as low as possible
                    loss += torch.sigmoid(original_prediction[i][b])
    return (1/len(X)) * loss






def ax_independence_cont(model_on_profiles, X, num_samples, distance):
    # We only consider nontrivial profiles, i.e., with at least 2 alternatives
    X_nontrivial = [prof for prof in X if prof.num_cands > 1]
    # Compute prediction of model
    original_prediction = model_on_profiles(X_nontrivial)
    # initialize the loss
    loss = torch.zeros(1).squeeze()
    for _ in range(num_samples):
        # For each original nontrivial profile we generate a permuted version
        X_permuted = []
        # whose rankings agree with the corresponding original ones in the
        # order of a given pair (a,b) of alternatives
        pairs_of_alternatives = []
        for prof in X_nontrivial:
            # Choose two distinct alternatives a and b in prof
            pair_of_alternatives = sample(prof.candidates,2)
            a = pair_of_alternatives[0]
            b = pair_of_alternatives[1]
            pairs_of_alternatives.append((a,b))
            # now build permuted version of prof by randomly sampling rankings
            # that, however, must agree in the order of a and b with the
            # corresponding ranking in the original prof
            permuted_prof = []    
            for ranking in prof.rankings:
                while True: #we'll break this while loop eventually (see below)
                    # make a copy of the original ranking
                    permuted_ranking = list(ranking)
                    # shuffle the copy
                    shuffle(permuted_ranking)
                    # check if it still agrees in the order of a and b
                    # with original ranking
                    if (permuted_ranking.index(a) > permuted_ranking.index(b)) == ((ranking.index(a) > ranking.index(b))):
                        # if so, add it to permuted list of profiles and break
                        permuted_prof.append(permuted_ranking)
                        break
                    # we will break the while loop: there is a 50% chance that
                    # permuted_ranking agrees in the order of a and b with
                    # the original ranking 
            # Add the permuted profile to the list of permuted profiles
            X_permuted.append(Profile(permuted_prof))        
        # Next compute prediction of the model on the permuted profiles
        permuted_prediction = model_on_profiles(X_permuted)
        # Finally, for independence to be satisfied to a high degree, we want
        # a small distance between, on_the_one_hand the predictions for
        # alternatives a and b when inputting the original profile and
        # on_the_other_hand the predictions for a and b when inputting the
        # permuted profile
        on_the_one_hand = torch.stack(
            [original_prediction[i][pairs_of_alternatives[i],] for i in range(len(original_prediction))],
            dim=0
        )
        on_the_other_hand = torch.stack(
            [permuted_prediction[i][pairs_of_alternatives[i],] for i in range(len(permuted_prediction))],
            dim=0
        )
        loss += distance(on_the_one_hand, on_the_other_hand)
    # return average loss
    return (1/num_samples)*loss