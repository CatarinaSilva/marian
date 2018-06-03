#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"

namespace amunmt {

class God;

class TranslationPiece {
  public:

    TranslationPiece(const God &god, unsigned vLineNum, const std::string& line);

    const Words& GetWords(unsigned index = 0) const;

    unsigned GetLineNum() const;

    std::string Debug(unsigned verbosity = 1) const;

  private:
    std::vector<Words> words_;
    unsigned lineNum_;

    TranslationPiece(const TranslationPiece &) = delete;
};

using TranslationPiecePtr = std::shared_ptr<TranslationPiece>;



}
