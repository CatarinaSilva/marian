#include "cpu/decoder/guided_score.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#include "common/scorer.h"
#include "common/god.h"
#include "cpu/decoder/best_hyps.h"
#include "common/vocab.h"
#include <algorithm>
#include "cpu/mblas/tensor.h"

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
    unsigned tab,
    std::vector<float> tpmap)
  : Scorer(god, name, config, tab), tpMap_(tpmap)
{}

State* GuidedScorer::NewState() const {
  return new GSState();
}

unsigned GuidedScorer::GetVocabSize() const {
  return tpMap_.size();
}

BaseTensor& GuidedScorer::GetProbs() {
  return Probs_;
}


void GuidedScorer::Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes) {
  size_t cols = Probs_.size();
  for(size_t i = 0; i < Probs_.dim(0); ++i) {
    std::copy(tpMap_.begin(), tpMap_.end(), Probs_.begin() + i * cols);
  }
}

//void GuidedScorer::AssembleBeamState(const State& in,
//                                     const Beam& beam,
//                                     State& out) {
//  std::vector<unsigned> beamWords;
//  std::vector<unsigned> beamStateIds;
//  for(auto h : beam) {
//      beamWords.push_back(h->GetWord());
//      beamStateIds.push_back(h->GetPrevStateIndex());
//  }
//
//  const EDState& edIn = in.get<EDState>();
//  EDState& edOut = out.get<EDState>();
//
//  edOut.GetStates() = mblas::Assemble<mblas::byRow, mblas::Tensor>(edIn.GetStates(), beamStateIds);
//  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
//}

//void GuidedScorer::LoadTranslationPieces(const Sentences& translation_pieces) {
//  translationPieces_ = translation_pieces;
//}


/////////////////////////////////////////////

GuidedScorerLoader::GuidedScorerLoader(
  const std::string name,
  const YAML::Node& config)
  : Loader(name, config)
{}

void GuidedScorerLoader::Load(const God& god) {
  string type = Get<string>("type");
  string path = Get<string>("path");
  LOG(info)->info("Model type: {}", type);

  const Vocab& tvcb = god.GetTargetVocab();
  tpMap_.resize(tvcb.size(), 0.0);

  if(Has("path")) {
    LOG(info)->info("Loading translation pieces (glossaries) from {}",Get<std::string>("path"));
    YAML::Node pieces = YAML::Load(InputFileStream(Get<std::string>("path")));
    for(auto&& word : pieces) {
      string entry = word.first.as<string>();
      tpMap_[tvcb[entry]] = 1.0;
    }
  }

}

ScorerPtr GuidedScorerLoader::NewScorer(const God &god, const DeviceInfo&) const{
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  return ScorerPtr(new GuidedScorer(god, name_, config_, tab, tpMap_));
}

BaseBestHypsPtr GuidedScorerLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  return BaseBestHypsPtr(new CPU::BestHyps(god));
}

}
}
