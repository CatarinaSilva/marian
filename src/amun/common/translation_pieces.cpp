#include <algorithm>
#include <sstream>
#include "translation_pieces.h"

using namespace std;

namespace amunmt {

TranslationPieces::TranslationPieces()
{}

TranslationPieces::~TranslationPieces()
{}

TranslationPiecePtr TranslationPieces::at(unsigned id) const
{
  return coll_.at(id);
}

const TranslationPiece &TranslationPieces::Get(unsigned id) const
{
  return *coll_.at(id);
}

unsigned TranslationPieces::size() const {
  return coll_.size();
}

void TranslationPieces::push_back(TranslationPiecePtr translation_piece) {
  coll_.push_back(translation_piece);
}

TranslationPiecesPtr TranslationPieces::NextMiniBatch(unsigned batchsize)
{
  TranslationPiecesPtr translation_pieces(new TranslationPieces());

  unsigned startInd = (batchsize > size()) ? 0 : size() - batchsize;
  for (unsigned i = startInd; i < size(); ++i) {
    TranslationPiecePtr translation_piece = coll_[i];
    translation_pieces->push_back(translation_piece);
  }
  coll_.resize(startInd);

  return translation_pieces;
}

}

