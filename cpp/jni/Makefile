
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDFLAGS += -fPIC -static-libstdc++

JNI_HEADERS_1 = /usr/lib/jvm/java-8-oracle/include

JNI_HEADERS_2 = /usr/lib/jvm/java-8-oracle/include/linux

OBJFILES = kaldi-jni.o

LIBNAME = kaldi-jni

ADDLIBS = ../chain/kaldi-chain.a ../cudamatrix/kaldi-cudamatrix.a \
          ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../nnet3/kaldi-nnet3.a

CXXFLAGS += -I$(JNI_HEADERS_1)

CXXFLAGS += -I$(JNI_HEADERS_2)

CXXFLAGS += -fPIC

include ../makefiles/default_rules.mk
