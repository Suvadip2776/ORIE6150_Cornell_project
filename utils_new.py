import votekit
from votekit.elections import Plurality

def get_candidates(profile):
    # should return list of candidates, in some unknown type of ordering 
    candidate_list = []
    for voter in profile:
        for candidate in voter: 
            if not (candidate in candidate_list):
                candidate_list.append(candidate)

    return candidate_list

def count_pairwise_wins(cand_a, cand_b, profile):
    num_wins = 0
    for ranking in profile:
        if ranking.index(cand_a) < ranking.index(cand_b):
            num_wins += 1 

    return num_wins

def ranking_to_pairwise(profile):
    # point is that given a profile, we output the matrix (n by n, where n is the number of candidates)
    # profile is given as a list of lists
    # so... given a profile, how do we get 
    # well when profile.indexOf(A) > profile.indexOf(B), i+=1
    candidate_list = get_candidates(profile)

    # so to create the first row, we take the 0th element of candidate_list and compare the number of times 
    # it wins against the jth element of candidate_list and add this to entry j of the first row 

    result_matrix = []

    # i,j = 0

    for i in range(len(candidate_list)):
        temp_vector = []
        for j in range(len(candidate_list)):
            if i == j:
                temp_vector.append(-1)
            else:
                wins_i_j = count_pairwise_wins(candidate_list[i], candidate_list[j], profile)
                temp_vector.append(wins_i_j)
        
        result_matrix.append(temp_vector)
                
    return result_matrix

def ranking_to_scores(profile):
    # given a profile, we return a matrix that
    return None