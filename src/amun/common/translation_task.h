#pragma once

#include <memory>

namespace amunmt {

class God;
class Histories;
class Sentences;
class TranslationPieces;

//void TranslationTaskAndOutput(const God &god, std::shared_ptr<Sentences> sentences, std::shared_ptr<Sentences> translation_pieces);

void TranslationTaskAndOutput(const God &god, std::shared_ptr<Sentences> sentences);
void TranslationTaskAndOutput(const God &god, std::shared_ptr<Sentences> sentences, std::shared_ptr<TranslationPieces> pieces);
std::shared_ptr<Histories> TranslationTask(const God &god, std::shared_ptr<Sentences> sentences);
std::shared_ptr<Histories> TranslationTask(const God &god, std::shared_ptr<Sentences> sentences, std::shared_ptr<TranslationPieces> pieces);

}  // namespace amunmt
