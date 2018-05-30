#include "cpu/decoder/guided_score.h"

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"

namespace amunmt {
namespace CPU {


////////////////////////////////////////////////

GuidedScorer::GuidedScorer(
	const God &god,
    const std::string& name,
    const YAML::Node& config,
    unsigned tab)
  : Scorer(god, name, config, tab)
{}

State* GuidedScorer::NewState() const {
  return new GSState();
}


}
}

