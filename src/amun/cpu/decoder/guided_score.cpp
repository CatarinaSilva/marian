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
    const Vocab& tvcb,
    float sim_tresh,
    unsigned max_n_grams)
  : Scorer(god, name, config, tab), tpMap_(tpmap), tvcb_(tvcb), simThresh_(sim_tresh), maxNgrams_(max_n_grams)
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
    translation_pieces_ = translation_pieces.at(0);
}

void GuidedScorer::Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes) {
  size_t cols = tpMap_.size();
  Words Lu = translation_pieces_->GetUnigrams();
  Probs_.Resize(beamSizes[0], cols, 1, 1);
  for(size_t i = 0; i < Probs_.dim(0); ++i) {
    std::vector<float> tp_map_temp(tpMap_);
    for(size_t j = 0; j < Lu.size(); ++j){
      Word unigram = Lu[j];
      vector<Word> unigrams_b;
      unigrams_b.push_back(unigram);
      float unigram_score = translation_pieces_->GetScore(unigrams_b);
      if(unigram_score >= simThresh_){
        tp_map_temp[unigram] = translation_pieces_->GetScore(unigrams_b);
      }
      if(i<last_ngrams_.size())
      {
        Words b = last_ngrams_[i];
        for(size_t k=0; k<b.size(); ++k){
          unigrams_b.insert(unigrams_b.begin(), b[k]);
          float ngram_score = translation_pieces_->GetScore(unigrams_b);
          if(ngram_score >= simThresh_){
            tp_map_temp[unigram] += translation_pieces_->GetScore(unigrams_b);
          }
        }
      }
    }
    std::copy(tp_map_temp.begin(), tp_map_temp.end(), Probs_.begin() + i * cols);
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
      while(hyp->GetPrevStateIndex() > 0 && prevWordCount < (maxNgrams_ - 1) ){
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

  vocab_size_ = Has("out_emb_size") ? Get<int>("out_emb_size") : 0;
  similarityThreshold_ = Has("similarity_threshold") ? Get<float>("similarity_threshold") : 0.0;
  maxNgrams_ = Has("max_n_grams") ? Get<int>("max_n_grams") : 4;

  if(vocab_size_ > 0){
    tpMap_.resize(vocab_size_, 0.0);
  } else {
    tpMap_.resize(god.GetTargetVocab().size(), 0.0);
  }
}

ScorerPtr GuidedScorerLoader::NewScorer(const God &god, const DeviceInfo&) const{
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  const Vocab& tvcb = god.GetTargetVocab();
  return ScorerPtr(new GuidedScorer(god, name_, config_, tab, tpMap_, tvcb, similarityThreshold_, maxNgrams_));
}

BaseBestHypsPtr GuidedScorerLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  return BaseBestHypsPtr(new CPU::BestHyps(god));
}

}
}


