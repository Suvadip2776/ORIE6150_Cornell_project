"""
Generation of datasets consisting of profiles with corresponding winning sets

A dataset is a list X of profiles and a list y of winning sets such that the 
winning set of the n-th profile in X is the n-th entry of y. 

The function `generate_data` samples such a dataset depending on various 
parameters which are passed as arguments to the function (number of voters, 
number of alternatives, probability model for sampling profiles, the rule to 
compute the corresponding winning set, etc.)
"""

import pref_voting
from pref_voting.generate_profiles import generate_profile
from pref_voting.profiles import Profile
from random import randint


import torch
from torch.utils.data import TensorDataset, DataLoader

import utils
from utils import recast_profile_wo_mult, padded_profile_list,flatten_list
from utils import profile_to_onehot, winner_to_vec, flatten_onehot_profile
import models

def generate_profile_data(
    max_num_voters_generation,
    max_num_alternatives_generation,
    dataset_size,
    election_sampling,
    rule_generation_list,
    merge,
    full_profile=False,
    axiom_generation_list=None,
    progress_report=None,
):
    """
    Samples a dataset consisting of profiles with corresponding winning sets

    Input:
    * `max_num_voters_generation` a positive integer describing the maximum
      number of voters occurring in the sampled profiles
    * `max_num_alternatives_generation` a positive integer describing the
      maximum number of alternatives occurring in the sampled profiles
    * `dataset_size` a positive integer describing the number of rows that the
      dataset should have (at least; depending on the choice of `merge` below 
      it might at times be slightly more)
    * `election_sampling` a dictionary describing the parameters for the 
      probability model with which profiles are generated. The most important 
      key is `probmodel`, e.g., `'IC'`, `'URN-R'`, `'MALLOWS-R'`. See 
      https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    * `rule_generation_list` is a list of voting rules (in the sense of the
      `pref_voting` package) with which winning sets will be computed. If the
      list is empty, only profiles are generated (which is useful when only
      profiles but no winning sets are needed)
    * `merge` describes how the winning sets computed by the different rules 
      are merged. The options are:
        * 'accumulative': for each rule in `rule_generation_list`, add the row
        consisting of the sampled profile and the winning set computed with 
        that rule.
        * 'selective': only add row if all the rules in `rule_generation_list`
        compute the very same winning set for the sampled profile.
        * 'empty': if `rule_generation_list` is empty, add the empty winning 
          set () as a dummy object.
    * `full_profile` is a Boolean variable, which is false by default. If true,
      only profiles of maximal numbers of voters and maximal numbers of
      alternatives are generated.
    * `axiom_generation_list` is None by default. If not None, it is a list of 
      the same length as `rule_generation_list`. The n-th element is a possibly
      empty list of tuples (a, d, s), with s optional, saying that we require, 
      for each sampled profile, that the n-th rule of `rule_generation_list` 
      satisfies axiom a to degree d, which is checked, if s is given, for each 
      profile using s-many samples. The axioms are implemented in `axioms.py`.
    * `progress_report` is None by default. If not None, it is a positive 
      integer n such that whenever n rows have been built a message is printed.

    Output:
    * A list `X` of profiles
    * A list `y` of the corresponding winning sets
    * A `sample_rate` which is the number of rows of the dataset divided by the
      total number of sampled profiles (i.e., both those that were added to the
      dataset and those that were not because they did not satisfy the 
      requirements)
    """

    # Initialize the list of profiles of the dataset
    X = []
    # Initialize the list of corresponding winning sets of the dataset
    y = []

    # Check inputs are of the right shape
    if axiom_generation_list is not None and \
       len(axiom_generation_list) != len(rule_generation_list):
        print(
            "The axiom list must have the same length as the rule list, even if all its entries are empty lists"
        )
        return
    else:
        # To count the percentage of success in the sampling process
        num_samples = 0
        # To count the number of rows the dataset has so far
        num_rows = 0
        while num_rows < dataset_size:
            if progress_report is not None:
                if num_rows % progress_report == 0:
                    print(
                        f"So far {num_rows} of the desired {dataset_size} rows have been generated"
                    )
            # Choose a number of alternatives and a number of voters
            if full_profile:
                num_alternatives = max_num_alternatives_generation
                num_voters = max_num_voters_generation
            else:
                num_alternatives = randint(1, max_num_alternatives_generation)
                # Note that randint chooses a random integer between and
                # including the boundaries of the interval
                num_voters = randint(1, max_num_voters_generation)

            # Generate a profile
            prof = generate_profile(num_alternatives, num_voters, election_sampling)
            prof = recast_profile_wo_mult(prof)

            # Compute the outputs of the rules (cast as tuples, so we can take
            # the set of the winning_list later)
            winning_list = []
            for rule in rule_generation_list:
                winning_list.append(tuple(rule(prof)))

            # If axiom_generation_list is provided, check that the axioms are
            # satisfied. We go through the winning_list and keep only those
            # entries for which the axioms are satisfied. We won't need anymore
            # the information from which rule a winning set came, so we don't
            # keep that information.
            if axiom_generation_list is not None:
                winning_list_checked = []
                for i in range(len(rule_generation_list)):
                    # If all provided axiom-lists are empty, there is nothing 
                    # to check, so we set the default to true:
                    checked = True
                    for ax_tuple in axiom_generation_list[i]:
                        axiom = ax_tuple[0]
                        degree = ax_tuple[1]
                        if len(ax_tuple) == 2:
                            if axiom(rule_generation_list[i], prof) in degree:
                                checked = True
                            else:
                                checked = False
                        if len(ax_tuple) == 3:
                            sample = ax_tuple[2]
                            if axiom(rule_generation_list[i], prof, sample) in degree:
                                checked = True
                            else:
                                checked = False
                    if checked == True:
                        winning_list_checked.append(winning_list[i])
                    # Now update the winning list with the checked one
                    winning_list = winning_list_checked

            # Merge results into dataset according to
            if merge == "accumulative":
                for win in winning_list:
                    X.append(prof)
                    y.append(win)
                num_samples += len(rule_generation_list)
                num_rows += len(winning_list)
            if merge == "selective":
                if len(set(winning_list)) == 1:
                    X.append(prof)
                    y.append(winning_list[0])
                    num_samples += 1
                    num_rows += 1
                else:  # do nothing and sample the next profile
                    num_samples += 1
                    num_rows += 0
            if merge == "empty":  # add empty winner tuple as dummy object
                X.append(prof)
                y.append(())
                num_samples += 1
                num_rows += 1
        if progress_report is not None:
            print(f'Done; {num_rows} rows have been generated')
        # Compute sample rate (ratio selected samples vs all considered ones)
        sample_rate = num_rows / num_samples
        return X, y, sample_rate
    



def pad_profile_data(X, y, max_num_voters, max_num_alternatives):
    """
    Recasts and pads a dataset of profiles and winning sets

    Input: a list `X` of profiles and list `y` of corresponding winning sets,
    and positive integers `max_num_voters` and `max_num_alternatives` to which
    lengths the profiles should be padded (see `padded_profile_list` in
    `utils.py`).

    Output: A pair `X_padded` and `y_padded`. Here `X_padded` is a list of
    profiles, where each profile is in the form of a list of lists padded by
    -1's to have rankings of length `max_number_alternatives` and
    `max_num_voters` many rankings. And `y_padded` is a list the characteristic
    functions (represented as binary lists) over `max_num_alternative` many
    profiles of the winning sets in y (see `winner_to_vec` in `utils.py`).
    """
    X_padded = [
        padded_profile_list(prof, max_num_voters, max_num_alternatives)
        for prof in X
    ]
    y_padded = [
        winner_to_vec(list(win), max_num_alternatives)
        for win in y
    ]
    return X_padded, y_padded



def onehot_profile_data(X, y, max_num_voters, max_num_alternatives):
    """
    One-hots a dataset of profiles and winning sets

    Input: a list `X` of profiles and list `y` of corresponding winning sets,
    and positive integers `max_num_voters` and `max_num_alternatives` to which
    lengths the profiles should be padded.

    Output: A pair `X_onehot` and `y_onehot`. Here `X_onehot` is the result of 
    applying `profile_to_onehot` (see the description at `utils.py`) to each 
    profile in X and turn the corresponding winning set into its characteristic 
    function using `winner_to_vec`. 
    """
    X_onehot = [
        profile_to_onehot(profile, max_num_voters, max_num_alternatives)
        for profile in X
    ]
    y_onehot = [
        winner_to_vec(list(win), max_num_alternatives)
        for win in y
    ]
    return X_onehot, y_onehot


def tensorize_profile_data_MLP(
        X,
        y,
        max_num_voters,
        max_num_alternatives,
        batch_size,
        shuffle=False
    ):
    """
    Turns voting data into tensor form to be used for MLP
    """
    # Onehot and pad training data
    X, y = onehot_profile_data(X, y, max_num_voters, max_num_alternatives)
    # Turn them into tensor of flat lists
    X_tensor = torch.tensor(
        [flatten_onehot_profile(prof_list) for prof_list in X], 
        dtype=torch.float
    )
    y_tensor = torch.tensor(y, dtype=torch.float)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader






def tensorize_profile_data_CNN(
        X,
        y,
        max_num_voters,
        max_num_alternatives,
        batch_size,
        shuffle=False
    ):
    """
    Turns voting data into tensor form to be used for CNN
    """
    # Onehot and pad training data
    X, y = onehot_profile_data(X, y, max_num_voters, max_num_alternatives)
    # Turn each profile into image and concatenate to a tensor
    X_tensor = torch.cat(
        [models.profile_list_to_image(prof_list) for prof_list in X], 
        dim=0
    )
    # Turn corresponding winning sets into tensor
    y_tensor = torch.tensor(y, dtype=torch.float)
    dataset = TensorDataset(X_tensor,y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def tensorize_profile_data_WEC(
        embedding_model,
        sentences,
        y,
        max_num_voters,
        max_num_alternatives,
        batch_size,
        num_of_unks=False,
        shuffle=False
    ):
    """
    Turns voting data into tensor form to be used for WEC

    `sentences` are the original list of profiles X turned into a corpus, i.e.,
    each profile in X is turned into a sentence by changing each of its 
    rankings into a word

    We assume that the `embedding_model` contains the words
    * 'UNK' used for unknown words
    * 'PAD' used for padding up sentences to full length.

    If `num_of_unks` is set to True, outputs how many 'UNKS's occurred in 
    `sentences`.
    """

    # To input a profile, i.e., list of words, into the neural net, 
    # we give it the tensorized lists of the indices of those words, 
    # padded to the maximal length.
    # So we stack all those tensors together as training data 
    unk_idx = embedding_model.wv.key_to_index['UNK']
    pad_idx = embedding_model.wv.key_to_index['PAD']
    pad_length = max_num_voters
    X_tensor = torch.stack(
        [models.sentence_to_idx(embedding_model, sentence, unk_idx, pad_idx, pad_length) 
         for sentence in sentences],
        dim=0
    )
    y_tensor = torch.tensor(
        [utils.winner_to_vec(list(win), max_num_alternatives) for win in y],
        dtype=torch.float
    )
    dataset = TensorDataset(X_tensor,y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if num_of_unks:
        # Compute number of UNKs in dev set
        num_unks = sum(
            [sentence_idx.count(unk_idx) for sentence_idx in X_tensor.tolist()]
        )
        all_words = len(X_tensor.tolist()[0]) * len(X_tensor.tolist())
        ratio = (100*num_unks)/all_words    
        summary_unks = {
            'num_unks':num_unks, 
            'all_words':all_words, 
            'ratio':ratio
        }
    else:
        summary_unks = None

    return dataloader, summary_unks