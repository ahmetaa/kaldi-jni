#include "jni/kaldi-jni.h"
#include "nnet3/nnet-utils.h"
#include <iostream>
#include "hmm/transition-model.h"
#include "nnet3/nnet-utils.h"
#include "jni/kaldijni_KaldiAcousticScorer.h"
#include "util/common-utils.h"

#include "base/kaldi-common.h"
#include "tree/context-dep.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"


namespace kaldi {
namespace jni {

void DecoderConfig::Info() {
  std::cout << "Info will be here" << std::endl;
}


} // namespace jni
} // namespace kaldi

extern "C" {

kaldi::jni::DecoderConfig* instance(jlong handle) {
  return reinterpret_cast<kaldi::jni::DecoderConfig *>(handle); 
}

JNIEXPORT jlong JNICALL Java_kaldijni_KaldiWrapper_initialize
  (JNIEnv *env, jobject obj, jstring model_path_str, jstring fst_path_str, jstring symbol_path_str) {

  using namespace kaldi;

  const char *m_chars = env->GetStringUTFChars(model_path_str, 0);
  const std::string model_path(m_chars);

  std::cout << "Initializing with model " << model_path << std::endl;

  TransitionModel trans_model;
  kaldi::nnet3::AmNnetSimple am_nnet;
  {
    bool binary;
    kaldi::Input ki(model_path, &binary);      
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
  }

  std::cout << "Acoustic Model loaded." << std::endl;

  const char *f_chars = env->GetStringUTFChars(fst_path_str, 0);
  const std::string fst_path(f_chars);

  fst::Fst<fst::StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_path);
  std::cout << "Fst loaded." << std::endl;

  const char *s_chars = env->GetStringUTFChars(symbol_path_str, 0);
  const std::string symbol_path(s_chars);

  fst::SymbolTable *word_syms = fst::SymbolTable::ReadText(symbol_path);
  std::cout << "Word Symbol Table loaded." << std::endl;

  kaldi::jni::DecoderConfig *instance = new kaldi::jni::DecoderConfig(
     am_nnet, decode_fst, word_syms, trans_model);

  return reinterpret_cast<jlong>(instance);
}

JNIEXPORT void JNICALL Java_kaldijni_KaldiWrapper_decode
  (JNIEnv *env, jobject obj, jlong handle, jstring out_path_str, jstring utterance_id_str, jfloatArray feature_arr) {

  kaldi::jni::DecoderConfig *config = instance(handle);

  const char *o_str = env->GetStringUTFChars(out_path_str, 0);
  const std::string out_path(o_str);

  const char *u_str = env->GetStringUTFChars(utterance_id_str, 0);
  const std::string utterance_id(u_str);

  bool allow_partial = true;  
  kaldi::LatticeFasterDecoderConfig decoderConf;
  
  decoderConf.beam = 15.0f;
  decoderConf.max_active = 7000;
  decoderConf.min_active = 200;
  decoderConf.lattice_beam = 8.0f;

  kaldi::nnet3::NnetSimpleComputationOptions nnetOptions;
  nnetOptions.frames_per_chunk = 50;
  nnetOptions.acoustic_scale = 0.1f;
  nnetOptions.extra_left_context = 0;
  nnetOptions.extra_right_context = 0;
  nnetOptions.extra_left_context_initial = -1;
  nnetOptions.extra_right_context_final = -1;

  jfloat *features = env->GetFloatArrayElements(feature_arr, NULL);

  
  
/**
           this.postDecodeAcwt = 1.0f;
            this.numThreads = 1;
            this.extraLeftContext = 0;
            this.extraRightContext = 0;
            this.extraLeftContextInitial = -1;
            this.extraRightContextFinal = -1;
            this.minimize = false;
            this.maxActive = 7000;
            this.minActive = 200;
            this.framesPerChunk = 50;
            this.beam = 15.0f;
            this.latticeBeam = 8.0f;
            this.acousticScale = 0.1f;
*/
  std::cout << "Decoder conf. Beam = " << decoderConf.beam << std::endl; 

}



JNIEXPORT jstring JNICALL Java_kaldijni_KaldiWrapper_modelInfo
  (JNIEnv *env, jobject obj, jlong handle) {
  std::string result = instance(handle)->GetAmNnet().Info();
  return env->NewStringUTF(result.c_str());
}


}
