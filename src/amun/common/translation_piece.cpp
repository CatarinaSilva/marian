#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "translation_piece.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"
#include "common/utils.h"

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

  std::stringstream ssLine;
  ssLine << line;
  ptree root;

  read_json(ssLine, root);
  ptree &pieces = (*root.find("translationPieces")).second;
  BOOST_FOREACH(ptree::value_type &pmap, pieces)
  {
    float score = pmap.second.get<float>("score", 0.0);
    std::string ngrams = pmap.second.get<string>("n_grams", "");
    if(ngrams != "")
    {
      std::vector<std::string> tokens;
      Split(ngrams, tokens, " ");
      Words ngram = god.GetTargetVocab()(tokens, 0);
      Du_.insert(std::pair<std::string,float>(WordsToKey(ngram), score));
      if(tokens.size() == 1)
      {
        Lu_.push_back(god.GetTargetVocab()[ngrams]);
      }
    }
  }
}

unsigned TranslationPiece::GetLineNum() const {
  return lineNum_;
}

std::string TranslationPiece::WordsToKey(Words ngrams) const {
  std::stringstream keystring;
  std::copy(ngrams.begin(), ngrams.end(), std::ostream_iterator<unsigned>(keystring, " "));
  return keystring.str();
}

float TranslationPiece::GetScore(Words ngrams){
  std::string ngrams_str = WordsToKey(ngrams);
  if(Du_.count(ngrams_str) == 1){
    return Du_[ngrams_str];
  } else {
    return 0.0;
  }
}

Words TranslationPiece::GetUnigrams() {
  return Lu_;
}


}
