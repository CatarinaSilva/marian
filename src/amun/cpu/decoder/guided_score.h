#pragma once

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"
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
        unsigned tab);

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

  protected:
    mblas::Tensor SourceContext_;
};

class GuidedScorerLoader : public Loader {
  public:
    GuidedScorerLoader(const std::string name,
                       const YAML::Node& config);

    virtual void Load(const God& god);

    virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
    BaseBestHypsPtr GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const;
};


}
}
