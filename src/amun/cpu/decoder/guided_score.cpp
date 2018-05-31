#include "cpu/decoder/guided_score.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#include "common/scorer.h"
#include "common/god.h"
#include "cpu/decoder/best_hyps.h"

using namespace std;

namespace amunmt {
namespace CPU {

using GSState = GuidedScorerState;

GuidedScorerState::GuidedScorerState()
{
}

std::string GuidedScorerState::Debug(unsigned verbosity) const
{
	return CPU::mblas::Debug(states_);
}

CPU::mblas::Tensor& GuidedScorerState::GetStates() {
  return states_;
}

CPU::mblas::Tensor& GuidedScorerState::GetEmbeddings() {
  return embeddings_;
}

const CPU::mblas::Tensor& GuidedScorerState::GetStates() const {
  return states_;
}

const CPU::mblas::Tensor& GuidedScorerState::GetEmbeddings() const {
  return embeddings_;
}

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

unsigned GuidedScorer::GetVocabSize() const {
}

BaseTensor& GuidedScorer::GetProbs() {
  return Probs_;
}

/////////////////////////////////////////////

GuidedScorerLoader::GuidedScorerLoader(
  const std::string name,
  const YAML::Node& config)
  : Loader(name, config)
{}

void GuidedScorerLoader::Load(const God&) {
  std::string type = Get<std::string>("type");
  LOG(info)->info("Model type: {}", type);
}

ScorerPtr GuidedScorerLoader::NewScorer(const God &god, const DeviceInfo&) const {
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  std::string type = Get<std::string>("type");
  return ScorerPtr(new GuidedScorer(god, name_, config_, tab));
}

BaseBestHypsPtr GuidedScorerLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  return BaseBestHypsPtr(new CPU::BestHyps(god));
}

}
}
