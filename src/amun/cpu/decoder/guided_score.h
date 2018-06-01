#pragma once

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include "common/scorer.h"
#include "common/loader.h"
#include "common/sentences.h"
#include "common/logging.h"
#include "common/base_best_hyps.h"
#include "cpu/mblas/tensor.h"

namespace amunmt {
namespace CPU {

class GuidedScorerState : public State {
  public:
    GuidedScorerState();
    GuidedScorerState(const GuidedScorerState&) = delete;

    virtual std::string Debug(unsigned verbosity = 1) const;

    CPU::mblas::Tensor& GetStates();
    const CPU::mblas::Tensor& GetStates() const;

  	CPU::mblas::Tensor& GetEmbeddings();
    const CPU::mblas::Tensor& GetEmbeddings() const;

  private:
    CPU::mblas::Tensor states_;
    CPU::mblas::Tensor embeddings_;
};

class GuidedScorer : public Scorer {
  private:
    using GSState = GuidedScorerState;

  public:
    GuidedScorer(
    	const God &god,
        const std::string& name,
        const YAML::Node& config,
        unsigned tab,
        const std::vector<float> tpmap);

    virtual State* NewState() const;

    virtual void *GetNBest()
    {
      assert(false);
      return nullptr;
    }

    virtual const BaseTensor *GetBias() const
    {
      assert(false);
      return nullptr;
    }

    virtual void Decode(
        const State& in,
        State& out,
        const std::vector<unsigned>& beamSizes);

    virtual void BeginSentenceState(State& state, unsigned batchSize){}

    virtual void Encode(const Sentences& sources){}

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out){}

    void GetAttention(mblas::Tensor& Attention){}
    mblas::Tensor& GetAttention();

    unsigned GetVocabSize() const;

    void SetSource(const Sentence& source);

    BaseTensor& GetProbs();

    void Filter(const std::vector<unsigned>& filterIds){}

    //void LoadTranslationPieces(const Sentences& translation_pieces);

  protected:
    std::vector<float> tpMap_;
    mblas::ArrayMatrix Probs_;
    std::vector<float> costs_;

};

class GuidedScorerLoader : public Loader {
  public:
    GuidedScorerLoader(const std::string name,
                       const YAML::Node& config);

    virtual void Load(const God& god);

    virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
    BaseBestHypsPtr GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const;

  protected:
    std::vector<float> tpMap_;
};


}
}
