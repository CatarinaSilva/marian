#pragma once
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "types.h"

namespace amunmt {

class God;

class TranslationPiece {
  public:

    TranslationPiece(const God &god, unsigned vLineNum, const std::string& line);

    unsigned GetLineNum() const;
    Words GetUnigrams();
    float GetScore(Words ngrams);

  private:
    unsigned lineNum_;
    std::map<std::string,float> Du_;
    Words Lu_;

    std::string WordsToKey(Words ngrams) const;

    TranslationPiece(const TranslationPiece &) = delete;
};

using TranslationPiecePtr = std::shared_ptr<TranslationPiece>;



}