#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "translation_piece.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

using namespace std;
using boost::property_tree::ptree;
using boost::property_tree::read_json;

namespace amunmt {

// TranslationPieces are a list of maps with following info:
// type: str (1-gram, 2-gram, 3-gram or 4-gram)
// words: words composing translation piece
// score: similarity prior score
TranslationPiece::TranslationPiece(const God &god, unsigned vLineNum, const std::string& line)
  : lineNum_(vLineNum)
{

  std::stringstream ss;
  ss << line;
  ptree translation_pieces;

  read_json(ss, translation_pieces);

  for (ptree::value_type &tpiece : translation_pieces.get_child("translationPieces"))
  {
    // Get the label of the node
    std::string tp = tpiece.first;
    // Get the content of the node
    std::string tpdata = tpiece.second.data();
    LOG(info)->info("{}: {}", tp, tpdata);
  }

}

unsigned TranslationPiece::GetLineNum() const {
  return lineNum_;
}

const Words& TranslationPiece::GetWords(unsigned index) const {
  return words_[index];
}

//const Words& TranslationPiece::GetScore(unsigned index) const {
//  return score_[index];
//}


}

