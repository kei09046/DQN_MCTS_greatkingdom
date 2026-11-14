#include "fairkomi.h"
#include "random.h"
#include <cmath>

std::vector<std::optional<float>> estimate_fair_komi(PolicyValueNet& net){
	// komi grid the value is read at : the range self-play actually plays (fair komi +- noise, never below 0.5).
	// Outside it the value is untrained and can be anything, which would mislead the fit below.
	constexpr float komiLo = 0.5f, komiHi = 8.5f;
	const int nKomi = static_cast<int>(komiHi - komiLo) + 1;
	const auto& classes = rnd::neutral_point_classes();
	const int nSetups = rnd::neutral_setup_count();

	// every orientation of every setup (each point of a class; the empty setup once) at every komi on the grid.
	std::vector<std::vector<Move>> setups;
	std::vector<int> setupIdx;
	for(int k=0; k<nSetups; ++k){
		if(k == static_cast<int>(classes.size())){
			setups.push_back({});
			setupIdx.push_back(k);
		}
		else
			for(const Move& m : classes[k]){
				setups.push_back({m});
				setupIdx.push_back(k);
			}
	}
	std::vector<int> count(nSetups, 0);
	for(int k : setupIdx)
		count[k]++;

	// value[k][j] : mean value of the empty board of setup k at komi komiLo + j.
	std::vector<std::vector<double>> value(nSetups, std::vector<double>(nKomi, 0.0));
	std::vector<std::pair<int, int>> pending; // (setup, komi index) of each position in input
	NNInput input;
	const auto flush = [&]{
		if(pending.empty())
			return;
		const auto out = net.batch_evaluate(input, static_cast<int>(pending.size()));
		for(size_t b=0; b<pending.size(); ++b)
			value[pending[b].first][pending[b].second] += out[b].winP / count[pending[b].first];
		pending.clear();
		input.clear();
	};
	for(size_t s=0; s<setups.size(); ++s)
		for(int j=0; j<nKomi; ++j){
			Game g(komiLo + j, setups[s]);
			g.set_policy_mask(); // fills the available move list (input channel 22), as search does before evaluating
			const NNInput in = PolicyValueNet::get_data(g);
			input.insert(input.end(), in.begin(), in.end());
			pending.emplace_back(setupIdx[s], j);
			if(static_cast<int>(pending.size()) == globalConfig.batchSize)
				flush();
		}
	flush();

	// The value should be monotone in komi, but a model that has learned komi only partly gives a bumpy curve with
	// spurious sign changes. So fit a step instead of taking the first sign change : orient the curve by its overall
	// trend (u = value signed so that it rises with komi), then pick the gap between grid points that minimizes how
	// strongly points disagree with "u < 0 below, u > 0 above", and interpolate inside that gap.
	constexpr double minTrend = 0.2; // least rise of the value across the grid for an estimate (else : no komi sense)
	std::vector<std::optional<float>> fair(nSetups);
	for(int k=0; k<nSetups; ++k){
		const auto& v = value[k];
		const double meanKomi = (nKomi - 1) / 2.0;
		double slope = 0.0, norm = 0.0;
		for(int j=0; j<nKomi; ++j){
			slope += (j - meanKomi) * v[j];
			norm += (j - meanKomi) * (j - meanKomi);
		}
		slope /= norm;
		if(std::abs(slope) * (nKomi - 1) < minTrend)
			continue;
		std::vector<double> u(v);
		if(slope < 0.0)
			for(double& x : u)
				x = -x;

		// cost of putting the threshold between grid points g and g + 1.
		int best = -1;
		double bestCost = 0.0;
		for(int g=0; g+1<nKomi; ++g){
			double cost = 0.0;
			for(int j=0; j<nKomi; ++j)
				cost += (j <= g) ? std::max(0.0, u[j]) : std::max(0.0, -u[j]);
			if(best < 0 || cost < bestCost){
				best = g;
				bestCost = cost;
			}
		}
		const double a = u[best], b = u[best + 1];
		const double t = (a <= 0.0 && b > 0.0) ? a / (a - b) : 0.5; // no clean crossing in the gap : its middle
		fair[k] = komiLo + best + static_cast<float>(t);
	}
	return fair;
}

FairKomiTable combine_fair_komi(const std::vector<std::vector<std::optional<float>>>& estimates, float fallback){
	FairKomiTable table(rnd::neutral_setup_count(), fallback);
	for(size_t k=0; k<table.size(); ++k){
		float sum = 0.0f;
		int n = 0;
		for(const auto& e : estimates)
			if(e.at(k)){
				sum += *e[k];
				n++;
			}
		if(n > 0)
			table[k] = sum / n;
	}
	return table;
}

void print_fair_komi(std::ostream& out, const FairKomiTable& table){
	const auto& classes = rnd::neutral_point_classes();
	for(size_t k=0; k<table.size(); ++k){
		if(k)
			out << " ";
		if(k < classes.size())
			out << "(" << static_cast<int>(classes[k][0].first) << "," << static_cast<int>(classes[k][0].second) << ")";
		else
			out << "none";
		out << "=" << std::round(table[k] * 10.0f) / 10.0f;
	}
}
