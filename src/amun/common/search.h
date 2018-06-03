#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"
#include "common/translation_pieces.h"

namespace amunmt {

class Histories;
class Filter;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    std::shared_ptr<Histories> Translate(const Sentences& sentences);
    std::shared_ptr<Histories> Translate(const Sentences& sentences, const TranslationPieces& translation_pieces);

  protected:
    States NewStates() const;
    void FilterTargetVocab(const Sentences& sentences);
    States Encode(const Sentences& sentences);
    States Encode(const Sentences& sentences, const TranslationPieces& translation_pieces);
    void CleanAfterTranslation();

    bool CalcBeam(
    		std::shared_ptr<Histories>& histories,
    		std::vector<unsigned>& beamSizes,
        Beam& prevHyps,
    		States& states,
    		States& nextStates);

    Search(const Search&) = delete;

  protected:
    DeviceInfo deviceInfo_;
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const Filter> filter_;
    const unsigned maxBeamSize_;
    const float maxLengthMult_;
    bool normalizeScore_;
    Words filterIndices_;
    BaseBestHypsPtr bestHyps_;

    //std::vector<unsigned> activeCount_;
    //void BatchStats();
};

}

