#include "cpu/decoder/guided_score.h"
#include "common/file_stream.h"
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

const CPU::mblas::Tensor& GuidedScorerState::GetStates() const {
  return states_;
}


////////////////////////////////////////////////

GuidedScorer::GuidedScorer(
	const God &god,
    const std::string& name,
    const YAML::Node& config,
    unsigned tab,
    std::vector<float> tpmap,
    const Vocab& tvcb)
  : Scorer(god, name, config, tab), tpMap_(tpmap), tvcb_(tvcb)
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

void GuidedScorer::AddTranslationPieces(State& state, unsigned batchSize, const TranslationPieces& translation_pieces) {
    const GSState& gsIn = state.get<GSState>();

    TranslationPiecePtr tp = translation_pieces.at(0);
    Words Lu = tp->GetUnigrams();
    for(size_t i = 0; i < Lu.size(); ++i){
        tpMap_[Lu[i]] = 1.0;
    }
}

void GuidedScorer::Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes) {
  size_t cols = tpMap_.size();
  Probs_.Resize(beamSizes[0], cols, 1, 1);
  for(size_t i = 0; i < Probs_.dim(0); ++i) {
    std::copy(tpMap_.begin(), tpMap_.end(), Probs_.begin() + i * cols);
  }
  std::copy(tpMap_.begin(), tpMap_.end(), Probs_.begin());
}

void GuidedScorer::AssembleBeamState(const State& in,
                                     const Beam& beam,
                                     State& out) {
  std::vector<unsigned> beamWords;
  std::vector<unsigned> beamStateIds;
  for(auto h : beam) {
      beamWords.push_back(h->GetWord());
      beamStateIds.push_back(h->GetPrevStateIndex());
  }
  string beamWordsLog(beamWords.begin(), beamWords.end());
  string beamStateIdsLog(beamStateIds.begin(), beamStateIds.end());

//  const EDState& edIn = in.get<EDState>();
//  EDState& edOut = out.get<EDState>();
//
//  edOut.GetStates() = mblas::Assemble<mblas::byRow, mblas::Tensor>(edIn.GetStates(), beamStateIds);
//  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
}



/////////////////////////////////////////////

GuidedScorerLoader::GuidedScorerLoader(
  const std::string name,
  const YAML::Node& config)
  : Loader(name, config)
{}

void GuidedScorerLoader::Load(const God& god) {
  string type = Get<string>("type");
  LOG(info)->info("Model type: {}", type);
  tpMap_.resize(74000, 0.0);
}

ScorerPtr GuidedScorerLoader::NewScorer(const God &god, const DeviceInfo&) const{
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  const Vocab& tvcb = god.GetTargetVocab();
  return ScorerPtr(new GuidedScorer(god, name_, config_, tab, tpMap_, tvcb));
}

BaseBestHypsPtr GuidedScorerLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  return BaseBestHypsPtr(new CPU::BestHyps(god));
}

}
}


