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

using namespace kaldi;

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
  (JNIEnv *env,
  jobject obj,
  jlong handle,
  jstring out_path_str,
  jstring utterance_id_str,
  jfloatArray feature_arr,
  jint frame_count,
  jint dimension) {

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
  nnetOptions.acoustic_scale = 1.0f;
  nnetOptions.extra_left_context = 0;
  nnetOptions.extra_right_context = 0;
  nnetOptions.extra_left_context_initial = -1;
  nnetOptions.extra_right_context_final = -1;

  jfloat *features = env->GetFloatArrayElements(feature_arr, NULL);

  std::cout << "Decoder conf. Beam = " << decoderConf.beam << std::endl;

}

JNIEXPORT void JNICALL Java_kaldijni_KaldiWrapper_decodeWithFeatureFile
  (JNIEnv *env,
  jobject obj,
  jlong handle,
  jstring out_path_str,
  jstring feature_path_str) {

  kaldi::jni::DecoderConfig *config = instance(handle);

  const char *o_str = env->GetStringUTFChars(out_path_str, 0);
  const std::string out_path(o_str);

  const char *f_str = env->GetStringUTFChars(feature_path_str, 0);
  const std::string feature_file(f_str);

  bool allow_partial = true;
  kaldi::LatticeFasterDecoderConfig decoderConf;

  decoderConf.beam = 15.0f;
  decoderConf.max_active = 7000;
  decoderConf.min_active = 200;
  decoderConf.lattice_beam = 8.0f;
  bool determinize = decoderConf.determinize_lattice;

  kaldi::nnet3::NnetSimpleComputationOptions decodable_opts;
  decodable_opts.frames_per_chunk = 50;
  decodable_opts.acoustic_scale = 1.0f;
  decodable_opts.extra_left_context = 0;
  decodable_opts.extra_right_context = 0;
  decodable_opts.extra_left_context_initial = -1;
  decodable_opts.extra_right_context_final = -1;

  kaldi::SequentialBaseFloatMatrixReader feature_reader(feature_file);

  CompactLatticeWriter compact_lattice_writer;
  LatticeWriter lattice_writer;
  if (! (determinize ? compact_lattice_writer.Open(out_path)
          : lattice_writer.Open(out_path))) {
      KALDI_ERR << "Could not open table for writing lattices: "
                 << out_path;
  }

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

  const TransitionModel &trans_model = config->GetTransitionModel();
  const fst::Fst<fst::StdArc> *decode_fst = config->GetDecodeFst();
  const fst::SymbolTable *word_syms = config->GetSymbolTable();
  const nnet3::AmNnetSimple am_nnet = config->GetAmNnet();


    // this compiler object allows caching of computations across
    // different utterances.
    kaldi::nnet3::CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                       decodable_opts.optimize_config);

    Int32VectorWriter words_writer("");
    Int32VectorWriter alignment_writer("");

  for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const kaldi::Matrix<BaseFloat> &features (feature_reader.Value());
      if (features.NumRows() == 0) {
         KALDI_WARN << "Zero-length utterance: " << utt;
      }

      LatticeFasterDecoder decoder(*decode_fst, decoderConf);
/*
      kaldi::DecodableInterface *nnet_decodable = new
          nnet3::DecodableAmNnetSimpleParallel(
              decodable_opts, trans_model, am_nnet,
              features, NULL, NULL,
              0);
*/

        kaldi::nnet3::DecodableAmNnetSimple nnet_decodable(
            decodable_opts, trans_model, am_nnet,
            features, /*ivector*/ NULL, /*online_ivectors*/ NULL,
            /*online_ivector_period*/ 0, &compiler);

        double like;

        if (DecodeUtteranceLatticeFaster(
                decoder, nnet_decodable, trans_model, word_syms, utt,
                decodable_opts.acoustic_scale, determinize, allow_partial,
                &alignment_writer, &words_writer, &compact_lattice_writer,
                &lattice_writer, &like)) {
          tot_like += like;
          frame_count += nnet_decodable.NumFramesReady();
          num_success++;
        } else num_fail++;

/*      kaldi::DecodeUtteranceLatticeFasterClass *task =
          new DecodeUtteranceLatticeFasterClass(
              decoder, nnet_decodable, // takes ownership of these two.
              trans_model, word_syms, utt, decodable_opts.acoustic_scale,
              determinize, allow_partial, *//**&alignment_writer*//* NULL, *//**&words_writer*//* NULL,
               &compact_lattice_writer, &lattice_writer,
               &tot_like, &frame_count, &num_success, &num_fail, NULL);*/

/*
      (*task)();
      delete task;
*/

  }

  std::cout << "Done" << std::endl;

}

JNIEXPORT jstring JNICALL Java_kaldijni_KaldiWrapper_modelInfo
  (JNIEnv *env, jobject obj, jlong handle) {
  std::string result = instance(handle)->GetAmNnet().Info();
  return env->NewStringUTF(result.c_str());
}

}
