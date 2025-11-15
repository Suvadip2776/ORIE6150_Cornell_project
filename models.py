"""
The neural network architectures that we use
"""

import math
import torch
from torch import nn

from random import sample
import itertools

from utils import padded_profile_list, flatten_list, profile_to_onehot
from utils import flatten_onehot_profile
from utils import kendall_tau_order

import pref_voting
from pref_voting.profiles import Profile


class MLP(nn.Module):
    """A MultiLayer Perceptron architecture for onehot profiles"""
    def __init__(self, max_num_voters, max_num_alternatives):
        super(MLP, self).__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        self.linear_input = nn.Linear(
            max_num_voters * max_num_alternatives* max_num_alternatives,
            128
        ) 
        self.linear_hidden = nn.Linear(128, 128)
        self.linear_output = nn.Linear(128, max_num_alternatives)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.linear_input(x))
        y = self.relu(self.linear_hidden(y))
        y = self.relu(self.linear_hidden(y))
        y = self.relu(self.linear_hidden(y))
        y = self.linear_output(y)
        return y


class MLP_small(nn.Module):
    """A MultiLayer Perceptron architecture for onehot profiles"""
    def __init__(self, max_num_voters, max_num_alternatives):
        super(MLP_small, self).__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        self.linear_input = nn.Linear(
            max_num_voters * max_num_alternatives* max_num_alternatives,
            128
        ) 
        self.linear_hidden = nn.Linear(128, 64)
        self.linear_output = nn.Linear(64, max_num_alternatives)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.linear_input(x))
        y = self.relu(self.linear_hidden(y))
        y = self.linear_output(y)
        return y


class MLP_large(nn.Module):
    """A MultiLayer Perceptron architecture for onehot profiles"""
    def __init__(self, max_num_voters, max_num_alternatives):
        super(MLP_large, self).__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        self.linear_input = nn.Linear(
            max_num_voters * max_num_alternatives* max_num_alternatives,
            128
        ) 
        self.linear_hidden = nn.Linear(128, 128)
        self.linear_output = nn.Linear(128, max_num_alternatives)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        y = self.relu(self.norm(self.linear_input(x)))
        y = self.relu(self.norm(self.linear_hidden(y)))
        y = self.relu(self.norm(self.linear_hidden(y)))
        y = self.relu(self.norm(self.linear_hidden(y)))
        y = self.relu(self.norm(self.linear_hidden(y)))
        y = self.relu(self.norm(self.linear_hidden(y)))
        y = self.linear_output(y)
        return y        

def MLP2rule_prediction(model, profile, full=False):
    """
    Takes a MLP model and a profile and outputs the winners
    
    Due to the nature of the architecture of the model, it may considers 
    more alternatives than are actually present in the profile. The 
    outputted winners are those declared winners by the model *and* that 
    are actually in the profile. We ignore any potential alternatives that are 
    declared winners by the model but that are not in the profile.
    """
    model.eval()
    with torch.no_grad():
        # Recast profile so it can be inputted to model
        onehot_prof_list = profile_to_onehot(
            profile,
            model.max_num_voters,
            model.max_num_alternatives
        )
        x = torch.tensor(
            flatten_onehot_profile(onehot_prof_list),
            dtype=torch.float
        )
        # Compute logits prediction of model
        logits = model(x)
        # Turn logits into binary prediction
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i for i in range(len(binary)) 
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]



def MLP2rule_prediction_n(model, profile, neut_samples, full=False):
    """
    Takes MLP model and profile and outputs the neutrality-averaged winners
    
    We first generate `neut_samples`-many permutations of the alternatives (or
    all possible ones if `neut_samples == None) and compute the corresponding
    alternative-permuted versions of the inputted `profile`. Then we compute 
    the logits-prediction of the `model` on each of those permuted profiles 
    (in one batch). We de-permute the predictions again and average all of 
    them. We take this as the final logits-prediction.
    
    Then we take the winners to be those alternatives that are alternatives
    in the profile and received at least 50% probability (unless full=true, 
    then all alternatives with more than 50% are outputted).
    """
    model.eval()
    with torch.no_grad():
        # For later, record the number of alternatives
        num_alternatives = profile.num_cands

        # (1) Generate alternative-permuted profiles:
        # We produce a list of (`neut_samples`)-many alternative-permuted
        # versions of `profile`. We initialize this list as
        alt_perm_profiles = []
        # And we also produce a list of the permutations we used for
        # each permutation. We initialize this as
        alt_permutations = []
        if neut_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                # keep the permutation for later in the following format
                p_max = list(p)+list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)
        else:        
            for _ in range(neut_samples):
                # choose a permutation p of the alternatives
                p = sample(list(range(num_alternatives)), num_alternatives)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance
                if p in alt_permutations:
                    break
                # keep the permutation for later in the following format
                p_max = p + list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)

        # (2) Recast the profiles so they can be inputted to the model as batch
        # Initialize the batch
        batch = []
        # Now loop over the profiles and recast them
        for alt_permuted_profile in alt_perm_profiles:
            onehot_prof_list = profile_to_onehot(
                alt_permuted_profile, 
                model.max_num_voters, 
                model.max_num_alternatives
            )
            x = torch.tensor(flatten_onehot_profile(onehot_prof_list),
                             dtype=torch.float)
            batch.append(x.squeeze(dim=0))
        # Cast batch as a tensor
        batch = torch.stack(batch, dim=0)

        # (3) Now compute the logits-prediction of model
        logits = model(batch)

        # (4) Undo the permutations in the predictions, initialized as
        re_permuted_logits = torch.zeros_like(logits)
        for j in range(len(logits)):
            re_permuted_logits[j] = logits[j][alt_permutations[j],]
        # Now average these re-permuted predictions, which we take as
        # the model's final prediction
        prediction = re_permuted_logits.mean(dim=0)

        # (4) Finally, turn this into a binary prediction
        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i 
                    for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i 
                    for i in range(len(binary))
                    if int(binary[i]) == 1]


def MLP2rule_prediction_na(
        model,
        profile,
        neut_samples,
        anon_samples,
        full=False
    ):
    """
    Takes MLP model and profile and outputs the neut-and-anon-averaged winners
    
    We first generate `neut_samples`-many permutations of the alternatives (or
    all possible ones if `neut_samples == None) and compute the corresponding
    alternative-permuted versions of the inputted `profile`. For each such 
    alternative-permuted profile, we also generate `anon-samples` many versions 
    by permuting the voters/rankings.

    For each alternative-permuted profile, we gather all the voter-permuted 
    versions in a batch, compute the model's logits-predictions, and average 
    those (since they all should be the same according to anonymity). 
    
    Then we de-permute these averaged predictions again and, in turn, average 
    all of de-permuted ones (since they all should be the same according to 
    neutrality). We take this as the final logits-prediction.
    
    Then we take the winners to be those alternatives that are alternatives
    in the profile and received at least 50% probability (unless full=true, 
    then all alternatives with more than 50% are outputted).
    """    
    model.eval()
    with torch.no_grad():
        # For later, record the number of alternatives
        num_alternatives = profile.num_cands
        num_voters = profile.num_voters

        # (1) Generate alternative-permuted profiles:
        # We produce a list of (`neut_samples`)-many alternative-permuted
        # versions of `profile`. We initialize this list as
        alt_perm_profiles = []
        # And we also produce a list of the permutations we used for
        # each permutation. We initialize this as
        alt_permutations = []
        if neut_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                # keep the permutation for later in the following format
                p_max = list(p)+list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)
        else:        
            for _ in range(neut_samples):
                # choose a permutation p of the alternatives
                p = sample(list(range(num_alternatives)), num_alternatives)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance
                if p in alt_permutations:
                    break
                # keep the permutation for later in the following format
                p_max = p + list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)

        # (2) For each alternative-permuted profile,
        #       (a) generate `anon-sample` many voter-permuted versions,
        #       (b) compute their predictions,
        #       (c) average them,
        #       (d) re-permute them,
        #       (e) and store them in the following list:
        alt_permuted_predictions = []
        # Start looping through the alt-permuted profiles:
        for i in range(len(alt_perm_profiles)):
            alt_permuted_profile = alt_perm_profiles[i]
            # (a)
            # initialize a batch of voter-permuted profiles
            batch = []
            # initialize the list of voter permutations that we use
            vot_permutations = []
            for _ in range(anon_samples):
                # choose a permutation p of the voters
                p = sample(list(range(num_voters)), num_voters)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance 
                if p in vot_permutations:
                    break
                # keep the permutation to check for later duplicates
                vot_permutations.append(p)
                
                # Now voter-permute the alt_permuted_profile:
                prof = alt_permuted_profile.rankings
                vot_perm_prof = [prof[p[vot]] for vot in range(num_voters)]
                vot_permuted_profile = Profile(vot_perm_prof)

                # Recast vot-permuted profile to input it to the model
                onehot_prof_list = profile_to_onehot(
                    vot_permuted_profile,
                    model.max_num_voters,
                    model.max_num_alternatives
                )
                x = torch.tensor(flatten_onehot_profile(onehot_prof_list),
                             dtype=torch.float)
                # We ignore the first dimension (for batches), which is 1,
                # so it is removed by squeeze 
                batch.append(x.squeeze(dim=0))
            # Cast batch as a tensor
            batch = torch.stack(batch, dim=0)


            # (b) compute the logits-prediction of the model
            logits = model(batch)

            # (c) average them
            prediction = logits.mean(dim=0)

            # (d) re-permute them
            re_permuted_prediction = prediction[alt_permutations[i],]
            alt_permuted_predictions.append(re_permuted_prediction)
        
        # (3) Average the alt_permuted predictions
        alt_permuted_predictions = torch.stack(alt_permuted_predictions, dim=0)
        prediction = alt_permuted_predictions.mean(dim=0)

        # (4) Finally, turn this into a binary prediction
        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i 
                    for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i
                    for i in range(len(binary))
                    if int(binary[i]) == 1]




def MLP2rule(model, full=False):
    return lambda profile: MLP2rule_prediction(model, profile, full)

def MLP2rule_n(model, neut_samples, full=False):
    return lambda profile: MLP2rule_prediction_n(
                            model, 
                            profile, 
                            neut_samples, 
                            full
                            )


def MLP2rule_na(model, neut_samples, anon_samples, full=False):
    return lambda profile: MLP2rule_prediction_na(
                model,
                profile,
                neut_samples,
                anon_samples,
                full
            )


def MLP2logits(model, X):
    """
    Compute the logits prediction of the model on a list of profiles

    Input: The `model` and a list of profiles `X`.
    Output: A tensor of the logits prediction for each profile in X
    """
    # Don't set to `model.eval()` or `with torch.no_grad()` since we
    # later use this in training.
    # Turn list of profiles into tensors that can be inputted into the model
    tensorized_profiles = []
    for profile in X:
        # Recast profile so it can be inputted to model
        onehot_prof_list = profile_to_onehot(
            profile, model.max_num_voters,
            model.max_num_alternatives
        )
        x = torch.tensor(
            flatten_onehot_profile(onehot_prof_list),
            dtype=torch.float
        )
        tensorized_profiles.append(x)
    # Turn tensorized list of profiles into a batch
    batch = torch.stack(tensorized_profiles,dim=0)
    # Compute logits prediction of model
    tensor_of_logits = model(batch)
    return tensor_of_logits







class CNN(nn.Module):
    """A Convolutional Neural Network architecture for voting"""

    def __init__(self, max_num_voters, max_num_alternatives, kernel1, kernel2, channels):
        super(CNN, self).__init__()
        """
        The kernels are of the form (height, width)
        """
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives

        # The first convolutional layer:
        # The inputted profile is regarded as a pixel image with
        C1_in = max_num_alternatives
        # many color channels (since an alternative is represented as a one-hot 
        # vector) and of height
        H1_in = max_num_alternatives
        # and width
        W1_in = max_num_voters 
        # We use
        C1_out = channels
        # many output channels (which is the number of feature maps).
        # Use padding 0 (default), dilation 1 (default), and stride 1 (default)
        # We use a kernel map of (height, width)-dimension (K1_0 , K1_1) with
        K1_0 = kernel1[0]
        K1_1 = kernel1[1]
        # If B is batch size, the input to this convolutional layer is of shape
        # (B, C1_in, H1_in, W1_in).
        # The output shape is
        # (B, C1_out, H1_out, W1_out)
        # with (since padding is 0, dilation is 1, and stride is 1)
        H1_out = math.floor(H1_in - (K1_0 - 1))
        W1_out = math.floor(W1_in - (K1_1 - 1))
        assert H1_out > 0, f'The first component of kernel1 was chosen too big'
        assert W1_out > 0, f'The second component of kernel1 was chosen too big'

        self.conv1 = nn.Conv2d(
            in_channels=C1_in, out_channels=C1_out, kernel_size=(K1_0, K1_1)
        )
        self.act1 = nn.ReLU()

        # The second convolutional layer:
        # The input has size
        H2_in = H1_out
        W2_in = W1_out
        C2_in = C1_out
        # We use again
        C2_out = channels
        # Use padding 0 (default), dilation 1 (default), and stride 1 (default)
        # We use the same kernel map (height, width)-dimension
        K2_0 = kernel2[0]
        K2_1 = kernel2[1]
        # If B is batch size, the input to this convolutional layer is of shape
        # (B, C2_in, H2_in, W2_in).
        # The output shape is
        # (B, C2_out, H2_out, W2_out)
        # with (since padding is 0, dilation is 1, and stride is 1)
        H2_out = math.floor(H2_in - (K2_0 - 1))
        W2_out = math.floor(W2_in - (K2_1 - 1))
        assert H2_out > 0, f'The first component of kernel2 was chosen too big'
        assert W2_out > 0, f'The second component of kernel2 was chosen too big'

        self.conv2 = nn.Conv2d(
            in_channels=C2_in, out_channels=C2_out, kernel_size=(K2_0, K2_1)
        )
        self.act2 = nn.ReLU()

        # The flatten layer maps the output of the previous layer to a flat
        # dimension of size C2_out * H2_out * W2_out
        self.flat = nn.Flatten()

        # Finally, some linear layers until the output layer
        self.fc3 = nn.Linear(C2_out * H2_out * W2_out, 128)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, max_num_alternatives)

    def forward(self, x):
        # The input tensor x is of shape (B, C1_in, H1_in, W1_in)
        y = self.act1(self.conv1(x))
        y = self.act2(self.conv2(y))
        y = self.flat(y)
        y = self.act3(self.fc3(y))
        y = self.act3(self.fc4(y))
        y = self.act3(self.fc4(y))
        y = self.fc5(y)
        return y


def profile_list_to_image(prof_list):
    """
    Turns a profile into a tensor-image

    The profile `prof_list` is assumed to already come in one-hot 
    and padded form. For example, if the profile was
    [[0, 1, 2], [2, 0, 1]]
    then its one-hot version is
    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]]
    The function now renders this as an image that can be given to CNNs, i.e., 
    with dimensions [batch, channel, height, width] where for us
    * the width is the number of voters 
    * the height is the number of alternatives
    * each pixel has `num_alternatives`-many 'color' channels. 
    * since we have just a single image, we add a dummy-dimension for the batch 
      dimension
    (The choice for the height and width is because the pref-voting package 
    also displays profiles with rankings in the height-dimension and voters in 
    the width-dimension.)  
    So in the example, the output is a tensor of shape [1, 3, 3, 2], i.e., 
    tensor([[[[1., 0.],
              [0., 1.],
              [0., 0.]],
    
             [[0., 0.],
              [1., 0.],
              [0., 1.]],
    
             [[0., 1.],
              [0., 0.],
              [1., 0.]]]])
    """
    # We first tensorize the list
    x = torch.tensor(prof_list, dtype=torch.float)
    # So far, its dimensions are [voters/width , alternatives/height, channel],
    # but the expected input for CNNs is [batch, channel, height, width],
    # so we swap the first and last dimension and then add a batch dimension
    x = torch.transpose(x,0,2)
    x = x[None, :, :, :]
    return x


def CNN2rule_prediction(model, profile, full=False):
    """
    Takes a CNN model and a profile and outputs the winners

    Due to the nature of the architecture of the model, it may considers 
    more alternatives than are actually present in the profile. The 
    outputted winners are those declared winners by the model *and* that 
    are actually in the profile. We ignore any potential alternatives that are 
    declared winners by the model but that are not in the profile.

    Unless `full` is set to true, then all winners are outputted, regardless 
    whether they are in the profile or not.
    """
    model.eval()
    with torch.no_grad():
        # Recast profile so it can be inputted to model
        onehot_prof_list = profile_to_onehot(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        x = profile_list_to_image(onehot_prof_list)
        # Compute logits prediction of model
        logits = model(x)
        # Turn logits into binary prediction
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i for i in range(len(binary)) 
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]



def CNN2rule_prediction_kendall(model, profile, version, full=False):
    """
    Takes CNN model and profile, kendall tau orders profile, outputs winners

    Due to the nature of the architecture of the model, it may considers 
    more alternatives than are actually present in the profile. The 
    outputted winners are those declared winners by the model *and* that 
    are actually in the profile. We ignore any potential alternatives that are 
    declared winners by the model but that are not in the profile.

    Unless `full` is set to true, then all winners are outputted, regardless 
    whether they are in the profile or not.
    """
    model.eval()
    with torch.no_grad():
        # First Kendall Tau order the profile
        profile = kendall_tau_order(profile, version)
        # Recast profile so it can be inputted to model
        onehot_prof_list = profile_to_onehot(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        x = profile_list_to_image(onehot_prof_list)
        # Compute logits prediction of model
        logits = model(x)
        # Turn logits into binary prediction
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]



def CNN2rule_prediction_n(model, profile, neut_samples, full=False):
    """
    Takes CNN model and profile and outputs the neutrality-averaged winners
    
    We first generate `neut_samples`-many permutations of the alternatives (or
    all possible ones if `neut_samples == None) and compute the corresponding
    alternative-permuted versions of the inputted `profile`. Then we compute 
    the logits-prediction of the `model` on each of those permuted profiles 
    (in one batch). We de-permute the predictions again and average all of 
    them. We take this as the final logits-prediction.
    
    Then we take the winners to be those alternatives that are alternatives
    in the profile and received at least 50% probability (unless full=true, 
    then all alternatives with more than 50% are outputted).
    """
    model.eval()
    with torch.no_grad():
        # For later, record the number of alternatives
        num_alternatives = profile.num_cands

        # (1) Generate alternative-permuted profiles:
        # We produce a list of (`neut_samples`)-many alternative-permuted
        # versions of `profile`. We initialize this list as
        alt_perm_profiles = []
        # And we also produce a list of the permutations we used for
        # each permutation. We initialize this as
        alt_permutations = []
        if neut_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                # keep the permutation for later in the following format
                p_max = list(p)+list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)
        else:
            for _ in range(neut_samples):
                # choose a permutation p of the alternatives
                p = sample(list(range(num_alternatives)), num_alternatives)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance
                if p in alt_permutations:
                    break
                # keep the permutation for later in the following format
                p_max = p + list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)

        # (2) Recast the profiles so they can be inputted to the model as batch
        # Initialize the batch
        batch = []
        # Now loop over the profiles and recast them
        for alt_permuted_profile in alt_perm_profiles:
            onehot_prof_list = profile_to_onehot(
                alt_permuted_profile,
                model.max_num_voters,
                model.max_num_alternatives
            )
            x = profile_list_to_image(onehot_prof_list)
            # We ignore the first dimension (for batches), which is 1,
            # so it is removed by squeeze
            batch.append(x.squeeze(dim=0))
        # Cast batch as a tensor
        batch = torch.stack(batch, dim=0)

        # (3) Now compute the logits-prediction of model
        logits = model(batch)

        # (4) Undo the permutations in the predictions, initialized as
        re_permuted_logits = torch.zeros_like(logits)
        for j in range(len(logits)):
            re_permuted_logits[j] = logits[j][alt_permutations[j],]
        # Now average these re-permuted predictions, which we take as
        # the model's final prediction
        prediction = re_permuted_logits.mean(dim=0)

        # (4) Finally, turn this into a binary prediction
        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i 
                    for i in range(len(binary)) 
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i 
                    for i in range(len(binary))
                    if int(binary[i]) == 1]



def CNN2rule_prediction_na(
        model,
        profile,
        neut_samples,
        anon_samples,
        full=False
    ):
    """
    Takes CNN model and profile and outputs the neut-and-anon-averaged winners
    
    We first generate `neut_samples`-many permutations of the alternatives (or
    all possible ones if `neut_samples == None) and compute the corresponding
    alternative-permuted versions of the inputted `profile`. For each such 
    alternative-permuted profile, we also generate `anon-samples` many versions 
    by permuting the voters/rankings.

    For each alternative-permuted profile, we gather all the voter-permuted 
    versions in a batch, compute the model's logits-predictions, and average 
    those (since they all should be the same according to anonymity). 
    
    Then we de-permute these averaged predictions again and, in turn, average 
    all of de-permuted ones (since they all should be the same according to 
    neutrality). We take this as the final logits-prediction.
    
    Then we take the winners to be those alternatives that are alternatives
    in the profile and received at least 50% probability (unless full=true, 
    then all alternatives with more than 50% are outputted).
    """
    model.eval()
    with torch.no_grad():
        # For later, record the number of alternatives
        num_alternatives = profile.num_cands
        num_voters = profile.num_voters

        # (1) Generate alternative-permuted profiles:
        # We produce a list of (`neut_samples`)-many alternative-permuted
        # versions of `profile`. We initialize this list as
        alt_perm_profiles = []
        # And we also produce a list of the permutations we used for
        # each permutation. We initialize this as
        alt_permutations = []
        if neut_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                # keep the permutation for later in the following format
                p_max = list(p)+list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)
        else:        
            for _ in range(neut_samples):
                # choose a permutation p of the alternatives
                p = sample(list(range(num_alternatives)), num_alternatives)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance
                if p in alt_permutations:
                    break
                # keep the permutation for later in the following format
                p_max = p + list(range(len(p),model.max_num_alternatives))
                alt_permutations.append(tuple(p_max))
                # Initialize the p-permuted version of `profile`
                alt_permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    alt_permuted_profile.append(permuted_ranking)
                alt_permuted_profile = Profile(alt_permuted_profile)
                # Add permuted profile to list
                alt_perm_profiles.append(alt_permuted_profile)

        # (2) For each alternative-permuted profile,
        #       (a) generate `anon-sample` many voter-permuted versions,
        #       (b) compute their predictions,
        #       (c) average them,
        #       (d) re-permute them,
        #       (e) and store them in the following list:
        alt_permuted_predictions = []
        # Start looping through the alt-permuted profiles:
        for i in range(len(alt_perm_profiles)):
            alt_permuted_profile = alt_perm_profiles[i]
            # (a)
            # initialize a batch of voter-permuted profiles
            batch = []
            # initialize the list of voter permutations that we use
            vot_permutations = []
            for _ in range(anon_samples):
                # choose a permutation p of the voters
                p = sample(list(range(num_voters)), num_voters)
                # avoid duplicates of permutations, as they will skew/weigh 
                # the average and lead to worse performance 
                if p in vot_permutations:
                    break
                # keep the permutation to check for later duplicates
                vot_permutations.append(p)
                
                # Now voter-permute the alt_permuted_profile:
                prof = alt_permuted_profile.rankings
                vot_perm_prof = [prof[p[vot]] for vot in range(num_voters)]
                vot_permuted_profile = Profile(vot_perm_prof)

                # Recast vot-permuted profile to input it to the model
                onehot_prof_list = profile_to_onehot(
                    vot_permuted_profile,
                    model.max_num_voters,
                    model.max_num_alternatives
                )
                x = profile_list_to_image(onehot_prof_list)
                # We ignore the first dimension (for batches), which is 1,
                # so it is removed by squeeze 
                batch.append(x.squeeze(dim=0))
            # Cast batch as a tensor
            batch = torch.stack(batch, dim=0)


            # (b) compute the logits-prediction of the model
            logits = model(batch)

            # (c) average them
            prediction = logits.mean(dim=0)

            # (d) re-permute them
            re_permuted_prediction = prediction[alt_permutations[i],]
            alt_permuted_predictions.append(re_permuted_prediction)
        
        # (3) Average the alt_permuted predictions
        alt_permuted_predictions = torch.stack(alt_permuted_predictions, dim=0)
        prediction = alt_permuted_predictions.mean(dim=0)

        # (4) Finally, turn this into a binary prediction
        binary = torch.round(torch.sigmoid(prediction)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i
                    for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i
                    for i in range(len(binary))
                    if int(binary[i]) == 1]



def CNN2rule(model, full=False):
    return lambda profile: CNN2rule_prediction(model, profile, full)


def CNN2rule_kendall(model, version, full=False):
    return lambda profile: CNN2rule_prediction_kendall(model, profile, version, full)


def CNN2rule_n(model, neut_samples, full=False):
    return lambda profile: CNN2rule_prediction_n(
                model,
                profile,
                neut_samples,
                full
            )

def CNN2rule_na(model, neut_samples, anon_samples, full=False):
    return lambda profile: CNN2rule_prediction_na(
                model,
                profile,
                neut_samples,
                anon_samples,
                full
            )
    


def CNN2logits(model, X):
    """
    Compute the logits prediction of the model on a list of profiles

    Input: The `model` and a list of profiles `X`.
    Output: A tensor of the logits prediction for each profile in X
    """
    # Don't set to `model.eval()` or `with torch.no_grad()` since we
    # later use this in training.
    # Turn list of profiles into tensors that can be inputted into the model
    tensorized_profiles = []
    for profile in X:
        # Recast profile so it can be inputted to model
        onehot_prof_list = profile_to_onehot(
            profile, model.max_num_voters, model.max_num_alternatives
        )
        x = profile_list_to_image(onehot_prof_list)
        tensorized_profiles.append(x)
    # Turn tensorized list of profiles into a batch
    batch = torch.cat(tensorized_profiles,dim=0)
    # Compute logits prediction of model
    tensor_of_logits = model(batch)
    return tensor_of_logits











class WEC(nn.Module):
    """A Word Embedding Classifier architecture for voting"""

    def __init__(self, word_embeddings, max_num_voters, max_num_alternatives):
        super(WEC, self).__init__()
        self.max_num_voters = max_num_voters
        self.max_num_alternatives = max_num_alternatives
        self.word_embeddings = word_embeddings
        self.embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embeddings.wv.vectors), freeze=False
        )
        self.linear_hidden1 = nn.Linear(word_embeddings.vector_size, 128)
        self.linear_hidden2 = nn.Linear(128, 128)
        self.linear_output = nn.Linear(128, max_num_alternatives)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x is a tensor of shape [batch_size, sentence_length]
        # (where sentence_length is max_num_voters)
        assert (
            len(x.shape) == 2
        ), f"The tensor inputted to the model needs to have two dimensions (but has dimensions {x.shape}): the first dimension is for the batches (which may be a singleton) and the second dimension is for the words of the inputted sentence."
        # Now add to every word in the sentence its embedding vector,
        # resulting in a tensor of shape
        # [batch_size, sentence_length, word_embedding_size]
        x = self.embeddings(x)
        # Next average the embeddings of the words/rankings in the
        # sentence/profile, i.e., average along dimension 1
        x = x.mean(dim=1)  
        # Now x has shape [batch_size, word_embedding_size] and hence can
        # be fed into linear layers

        x = self.relu(self.linear_hidden1(x))
        x = self.relu(self.linear_hidden2(x))
        x = self.relu(self.linear_hidden2(x))
        x = self.linear_output(x)
        # returning the logits
        return x
    




def ranking_to_string(ranking):
    """Takes list of integers and outputs them as string separated by spaces"""
    return " ".join([str(a) for a in ranking])

def sentence_to_idx(word_embedding, sentence, unk_idx, pad_idx=None, pad_length=None):
    """Computes the list of indices of the words in the sentence
    
    Input: A `word_embedding` model, a `sentence` (i.e., list of 
    words/strings), and two optional parameters `pad_idx` and `pad_length` 
    which, if given, are integers. 
    
    We assume the word embedding model has an 'UNK' token in its vocabulary.

    Output: A list of integers, where the n-th integer is the index according
    to the embedding model of the n-th word in the sentence. If `pad_idx` and 
    `pad_length` are given, this list is padded with the integer `pad_idx` up
    to length `pad_length`.
    """

    indices = [
        word_embedding.wv.key_to_index.get(word, unk_idx) 
        for word in sentence
    ]
    if pad_idx is not None and pad_length is not None:
        indices += [pad_idx for _ in range(len(indices),pad_length)]
    return torch.tensor(indices, dtype=torch.long)



def WEC2rule_prediction(model, profile, full=False):
    """
    Takes a WEC model and a profile and outputs the winners
    
    Due to the nature of the architecture of the model, it may considers 
    more alternatives than are actually present in the profile. The 
    outputted winners are those declared winners by the model *and* that 
    are actually in the profile. We ignore any potential alternatives that are 
    declared winners by the model but that are not in the profile.

    Unless `full` is set to true, then all winners are outputted, regardless 
    whether they are in the profile or not.
    """    
    model.eval()
    with torch.no_grad():
        # Recast profile so it can be inputted to model
        sentence = [ranking_to_string(ranking) for ranking in profile.rankings]
        unk_idx = model.word_embeddings.wv.key_to_index['UNK']
        pad_idx = model.word_embeddings.wv.key_to_index['PAD']
        pad_length = model.max_num_voters
        x = sentence_to_idx(
            model.word_embeddings, 
            sentence, 
            unk_idx, 
            pad_idx, 
            pad_length
        )
        # (NB: Conceptually, we could also omit the padding, but since the
        # training and accuracy-testing of the model is done with the padding,
        # we include it here, too.)
        # Compute logits prediction of model
        logits = model(x[None, :])
        # Turn logits into binary prediction
        binary = torch.round(torch.sigmoid(logits)).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            return [i for i in range(len(binary))
                    if int(binary[i]) == 1 and i in profile.candidates]
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]







def WEC2rule_prediction_n(
        model,
        profile,
        num_samples,
        full=False,
        print_sigmoids=False
    ):
    """
    Takes a WEC model and a profile and outputs the neutrality-averaged winners
    
    We first generate `num_samples`-many permutations of the alternatives (or
    all possible ones if `num_samples == None) and compute the corresponding 
    versions of the inputted `profile`. Then we compute the logits-prediction 
    of the `model` on each of those permuted profiles (in one batch). 
    We de-permute the predictions again and average all of them. We take this 
    as the final logits-prediction.
    Then we take the winners to be those alternatives that are alternatives
    in the profile and received at least 50% probability (unless full=true, 
    then all alternatives with more than 50% are outputted).

    If `print_sigmoids` is True, then not only the winning set is outputted, 
    but also the sigmoids, i.e., the probability for each alternative in the 
    profile (i.e., `full` is False in this case).
    """
    model.eval()
    with torch.no_grad():
        # For later, record the number of alternatives
        num_alternatives = profile.num_cands
        # We will produce a list of (`num_samples`)-many permuted
        # versions of `profile`. We initialize this list as
        profiles = []
        # And we also produce a list of the permutations we used for
        # each permutation. We initialize this as
        permutations = []
        if num_samples is None:
            for p in list(itertools.permutations(range(num_alternatives))):
                # keep the permutation for later in the following format
                permutations.append(tuple(
                    list(p) + list(range(len(p),model.max_num_alternatives))
                ))
                # Initialize the p-permuted version of `profile`
                permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    permuted_profile.append(permuted_ranking)
                profile_permuted = Profile(permuted_profile)
                # Add permuted profile to list
                profiles.append(profile_permuted)
        else:        
            for _ in range(num_samples):
                # choose a permutation p of the alternatives
                p = sample(list(range(num_alternatives)), num_alternatives)
                # avoid duplicates of permutations, as they will skew/weigh
                # the average and lead to worse performance
                if p in permutations:
                    break
                # keep the permutation for later in the following format
                permutations.append(tuple(
                    p + list(range(len(p),model.max_num_alternatives))
                ))
                # Initialize the p-permuted version of `profile`
                permuted_profile = []
                for ranking in profile.rankings:
                    permuted_ranking = [p[alt] for alt in ranking]
                    permuted_profile.append(permuted_ranking)
                profile_permuted = Profile(permuted_profile)    
                # Add permuted profile to list
                profiles.append(profile_permuted)   

        # Recast the profiles so they can be inputted into the model as a batch
        # First initialize the batch
        batch = []
        # Collect the following parameters
        unk_idx = model.word_embeddings.wv.key_to_index['UNK']
        pad_idx = model.word_embeddings.wv.key_to_index['PAD']
        pad_length = model.max_num_voters
        # Now loop over the profiles and recast them
        for perm_profile in profiles:
            perm_sentence =  [ranking_to_string(ranking)
                              for ranking in perm_profile.rankings]
            x = sentence_to_idx(
                model.word_embeddings,
                perm_sentence,
                unk_idx,
                pad_idx,
                pad_length
            )
            batch.append(x)
        # Cast batch as a tensor
        batch = torch.stack(batch, dim=0)

        # Now compute the logits-prediction of model
        logits = model(batch)

        # Undo the permutations in the predictions, initialized as
        re_permuted_logits = torch.zeros_like(logits)
        for j in range(len(logits)):
            re_permuted_logits[j] = logits[j][permutations[j],]
        # Now average these re-permuted predictions, which we take as
        # the model's final prediction
        prediction = re_permuted_logits.mean(dim=0)

        # Turn this into binary prediction
        the_sigmoids = torch.sigmoid(prediction)
        binary = torch.round(the_sigmoids).squeeze()
        # Return the list of those alternatives that (1) the model declares
        # to be winners and that (2) are indeed candidates of the profile
        if full == False:
            if print_sigmoids == False:
                return [i for i in range(len(binary))
                        if int(binary[i]) == 1 and i in profile.candidates]
            else: 
                return {
                    'winning_set' : [i for i in range(len(binary)) 
                        if int(binary[i]) == 1 and i in profile.candidates],
                    'sigmoids' : the_sigmoids.squeeze().tolist(),
                }    
        else:
            return [i for i in range(len(binary)) if int(binary[i]) == 1]






def WEC2rule(model, full=False):
    return lambda profile: WEC2rule_prediction(model, profile, full)


def WEC2rule_n(model, sample, full=False, print_sigmoids=False):
    return lambda profile: WEC2rule_prediction_n(
        model, 
        profile, 
        sample, 
        full,
        print_sigmoids)


        

def WEC2logits(model, X):
    """
    Compute the logits prediction of the model on a list of profiles

    Input: The `model` and a list of profiles `X`.
    Output: A tensor of the logits prediction for each profile in X
    """
    # Don't set to `model.eval()` or `with torch.no_grad()` since we
    # later use this in training.
    # Gather basic data from model
    unk_idx = model.word_embeddings.wv.key_to_index['UNK']
    pad_idx = model.word_embeddings.wv.key_to_index['PAD']
    pad_length = model.max_num_voters
    # Turn list of profiles into tensors that can be inputted into the model
    tensorized_profiles = []
    for profile in X:
        # Recast profile so it can be inputted to model
        sentence = [ranking_to_string(ranking) for ranking in profile.rankings]
        x = sentence_to_idx(model.word_embeddings, sentence, unk_idx, pad_idx, pad_length)
        tensorized_profiles.append(x)
    # Turn tensorized list of profiles into a batch
    batch = torch.stack(tensorized_profiles,dim=0)
    # Compute logits prediction of model
    tensor_of_logits = model(batch)
    return tensor_of_logits