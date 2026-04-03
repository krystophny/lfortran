#include <libasr/codegen/asr_to_liric.h>

#ifdef HAVE_LFORTRAN_LIRIC

#include <liric/liric_session.h>

#include <libasr/asr.h>
#include <libasr/asr_utils.h>
#include <libasr/pass/pass_manager.h>

namespace LCompilers {

Result<int> asr_to_liric(ASR::TranslationUnit_t &/*asr*/,
    Allocator &/*al*/, const std::string &/*filename*/,
    CompilerOptions &/*co*/, diag::Diagnostics &diagnostics,
    int /*liric_backend*/)
{
    diagnostics.add(diag::Diagnostic(
        "liric backend not yet implemented",
        diag::Level::Error, diag::Stage::CodeGen, {}));
    return Error();
}

} // namespace LCompilers

#else

namespace LCompilers {

Result<int> asr_to_liric(ASR::TranslationUnit_t &,
    Allocator &, const std::string &,
    CompilerOptions &, diag::Diagnostics &diagnostics, int)
{
    diagnostics.add(diag::Diagnostic(
        "liric backend not enabled; rebuild with -DWITH_LIRIC=ON",
        diag::Level::Error, diag::Stage::CodeGen, {}));
    return Error();
}

} // namespace LCompilers

#endif
