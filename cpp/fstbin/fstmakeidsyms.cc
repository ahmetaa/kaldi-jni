// Copyright 2012  Vassil Panayotov <vd.panayotov@gmail.com>

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

int main(int argc, char **argv)
{
    using namespace kaldi;
    try {
        std::string sep = "_";
        bool verbose = false;
        bool show_tids = false;
        const char *usage = "Outputs symbolic names for all transition ids"
                "(can be used in graph visualizations)\n"
                "The format of the output is phone_hmm-state_pdfid_transidx tid"
                "(assuming the separator is \'_\')\n\n"
                "Usage: fstmaketidsyms [options] phones model [out_tid_symtab]\n";
        ParseOptions po(usage);
        po.Register("separator", &sep, "The symbol to be used as separator b/w tid's constituents");
        po.Register("verbose-output", &verbose, "Verbose output to stderr?");
        po.Register("show-tids", &show_tids, "Also show the transitions IDs");
        po.Read(argc, argv);
        if (po.NumArgs() < 2 || po.NumArgs() > 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string phnfile = po.GetArg(1);
        std::string mdlfile = po.GetArg(2);
        std::string tidsymfile = po.GetOptArg(3);

        TransitionModel trans_model;
        {
            bool binary;
            Input ki(mdlfile, &binary);
            trans_model.Read(ki.Stream(), binary);
        }

        fst::SymbolTable *phones_symtab = NULL;
        {   // read phone symbol table.
            std::ifstream is(phnfile.c_str());
            phones_symtab = ::fst::SymbolTable::ReadText(is, phnfile);
            if (!phones_symtab) KALDI_ERR << "Could not read phones symbol-table file "<< phnfile;
        }

        if (verbose)
            KALDI_LOG << "#phones: " << trans_model.GetPhones().size();
        fst::SymbolTable tidsymtab("tid-symbol-table");
        kaldi::int64 e = tidsymtab.AddSymbol(
                    phones_symtab->Find((kaldi::int64) 0)); // <eps>
        KALDI_ASSERT(e == 0);
        for (int i = 0; i < trans_model.NumTransitionIds(); i++) {
            std::ostringstream oss;
            kaldi::int32 tid = i+1; // trans-ids are 1-based
            int phnid = trans_model.TransitionIdToPhone(tid);
            oss << phones_symtab->Find(phnid);
            int hmmstate = trans_model.TransitionIdToHmmState(tid);
            int pdf = trans_model.TransitionIdToPdf(tid);
            int trans = trans_model.TransitionIdToTransitionIndex(tid);
            oss << sep << hmmstate << sep << pdf << sep << trans;
            if (show_tids)
                oss << '[' << tid << ']';
            tidsymtab.AddSymbol(oss.str(), tid);
            if (verbose)
                KALDI_LOG << "TransID:" << tid << "; PhoneID:" << phnid <<
                             "; Phone:" << phones_symtab->Find(phnid) <<
                             "; HMM state:" << hmmstate <<
                             "; PDF:" << pdf <<
                             "; trans:" << trans;
        }
        if (tidsymfile == "")
            tidsymtab.WriteText(std::cout);
        else
            tidsymtab.WriteText(tidsymfile);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
}
