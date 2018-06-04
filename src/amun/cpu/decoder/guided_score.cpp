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
    //translation_pieces_ = tp;
    Words Lu = tp->GetUnigrams();
    for(size_t i = 0; i < Lu.size(); ++i){
        Word unigram = Lu[i];
        vector<Word> unigrams;
        unigrams.push_back(unigram);
        tpMap_[unigram] = tp->GetScore(unigrams);
        LOG(info)->info(tpMap_[unigram]);
    }
}

void GuidedScorer::Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes) {
  size_t cols = tpMap_.size();
  //Words Lu = translation_pieces_->GetUnigrams();
  Probs_.Resize(beamSizes[0], cols, 1, 1);
  for(size_t i = 0; i < Probs_.dim(0); ++i) {
    std::copy(tpMap_.begin(), tpMap_.end(), Probs_.begin() + i * cols);
    //std::vector<float> tp_map_temp(tpMap_);
    //for(size_t j = 0; j < Lu.size(); ++j){
      //Word unigram = Lu[j];
      //LOG(info)->info("Word {}", std::to_string(unigram));
      //vector<Word> unigrams_b;
      //unigrams_b.push_back(unigram);
      //tp_map_temp[unigram] = translation_pieces_->GetScore(unigrams_b);
      //LOG(info)->info("Added {}", std::to_string(translation_pieces_->GetScore(unigrams_b)));
      //if(i<last_ngrams_.size())
      //{
      //  Words b = last_ngrams_[i];
      //  for(size_t k=0; k<b.size(); ++k){
      //    unigrams_b.insert(unigrams_b.begin(), b[k]);
      //    tp_map_temp[unigram] += translation_pieces_->GetScore(unigrams_b);
      //  }
      //}
    //}
    //std::copy(tp_map_temp.begin(), tp_map_temp.end(), Probs_.begin() + i * cols);
  }
}

void GuidedScorer::AssembleBeamState(const State& in,
                                     const Beam& beam,
                                     State& out) {
  last_ngrams_.clear();
  for(auto h : beam) {
      int prevWordCount = 0;
      Words local_ngrams;
      HypothesisPtr hyp = h;
      while(hyp->GetPrevStateIndex() > 0 && prevWordCount < 2 ){
          local_ngrams.push_back(hyp->GetWord());
          hyp = hyp->GetPrevHyp();
          prevWordCount++;
      }
      local_ngrams.push_back(hyp->GetWord());
      last_ngrams_.push_back(local_ngrams);
  }
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


