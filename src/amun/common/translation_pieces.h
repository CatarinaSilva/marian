#pragma once

#include "translation_piece.h"

namespace amunmt {

class TranslationPieces;
using TranslationPiecesPtr = std::shared_ptr<TranslationPieces>;

class TranslationPieces {
  public:
    TranslationPieces();
    ~TranslationPieces();

    void push_back(TranslationPiecePtr translation_piece);

    TranslationPiecePtr at(unsigned id) const;
    const TranslationPiece &Get(unsigned id) const;

    unsigned size() const;

    //void SortByBatchOrder(Sentences);

    TranslationPiecesPtr NextMiniBatch(unsigned batchsize);

  protected:
    std::vector<TranslationPiecePtr> coll_;

    TranslationPieces(const TranslationPieces &) = delete;
};

}

