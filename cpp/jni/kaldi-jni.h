#ifndef KALDI_JNI_AC_MODEL_WRAPPER_H_
#define KALDI_JNI_AC_MODEL_WRAPPER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/am-nnet-simple.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"

namespace kaldi {
namespace jni {

class DecoderConfig {
 public:

  DecoderConfig(
     const nnet3::AmNnetSimple &am_nnet,
     fst::Fst<fst::StdArc> *decode_fst,
     fst::SymbolTable *word_syms,
     const TransitionModel &trans_model
  ): am_nnet_(am_nnet), decode_fst_(decode_fst), word_syms_(word_syms), trans_model_(trans_model) {}

  const nnet3::AmNnetSimple &GetAmNnet() { return am_nnet_; }

  void Info();

 private:  
  nnet3::AmNnetSimple am_nnet_;
  fst::Fst<fst::StdArc> *decode_fst_;
  fst::SymbolTable *word_syms_;
  const TransitionModel &trans_model_;

};

} // namespace jni 
} // namespace kaldi

#endif // KALDI_JNI_AC_MODEL_WRAPPER_H_
