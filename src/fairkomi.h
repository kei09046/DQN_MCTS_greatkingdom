// Fair komi of each neutral setup : the komi at which a model's value of the empty board is even. Self-play and model
// comparison games draw komi around it (rnd::sample_fair_komi) so that games stay close for every neutral setup.
#ifndef FAIRKOMI_H
#define FAIRKOMI_H

#include <optional>
#include <ostream>
#include <vector>
#include "neuralNet.h"

using FairKomiTable = std::vector<float>; // indexed by rnd::neutral_setup_index

// net's fair komi of every neutral setup, from its value of the empty board (averaged over the setup's orientations)
// on a komi grid of 0.5 .. 8.5 : a step fit of where the value turns from black's favour to white's, robust to a bumpy
// curve. Fair komi beyond the grid comes out near its ends. nullopt for a setup whose value barely changes over the
// grid (e.g. an untrained model). net must not be evaluating anything else meanwhile.
std::vector<std::optional<float>> estimate_fair_komi(PolicyValueNet& net);

// one table from several models' estimates : per setup, the mean of the models that have an estimate, else fallback.
FairKomiTable combine_fair_komi(const std::vector<std::vector<std::optional<float>>>& estimates, float fallback);

// "(0,0)=4.2 (0,1)=2.8 ... none=2.6" : each setup by its first point (none : no neutral stone).
void print_fair_komi(std::ostream& out, const FairKomiTable& table);

#endif
