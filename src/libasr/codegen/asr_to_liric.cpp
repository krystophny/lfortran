#include <libasr/codegen/asr_to_liric.h>

#ifdef HAVE_LFORTRAN_LIRIC

#include <liric/liric_session.h>
#include <liric/liric_types.h>

#include <libasr/asr.h>
#include <libasr/asr_utils.h>
#include <libasr/exception.h>
#include <libasr/pass/pass_manager.h>

#include <unordered_map>
#include <vector>
#include <string>

namespace LCompilers {

class CodeGenError {
public:
    diag::Diagnostic d;
    CodeGenError(const std::string &msg)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen)}
    {}
    CodeGenError(const std::string &msg, const Location &loc)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen, {
            diag::Label("", {loc})
        })}
    {}
};

static inline uint64_t get_hash(ASR::asr_t *node) {
    return (uint64_t)node;
}

class ASRToLiricVisitor : public ASR::BaseVisitor<ASRToLiricVisitor> {
public:
    lr_session_t *s;
    Allocator &al;
    CompilerOptions &co;
    diag::Diagnostics &diag;

    uint32_t tmp;               /* current expression result vreg */
    bool is_target;             /* true when visiting assignment target */
    uint32_t proc_return;       /* return block id for current function */

    std::unordered_map<uint64_t, uint32_t> lr_symtab;  /* ASR hash -> vreg */

    std::vector<uint32_t> loop_head_stack;
    std::vector<uint32_t> loop_end_stack;

    ASRToLiricVisitor(lr_session_t *session, Allocator &al_,
                      CompilerOptions &co_, diag::Diagnostics &diag_)
        : s(session), al(al_), co(co_), diag(diag_),
          tmp(0), is_target(false), proc_return(0) {}

    /* ---- Type mapping -------------------------------------------------- */

    lr_type_t *get_type(ASR::ttype_t *t) {
        t = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(t));
        if (ASRUtils::is_integer(*t)) {
            int kind = ASRUtils::extract_kind_from_ttype_t(t);
            switch (kind) {
                case 1: return lr_type_i8_s(s);
                case 2: return lr_type_i16_s(s);
                case 4: return lr_type_i32_s(s);
                case 8: return lr_type_i64_s(s);
            }
        } else if (ASRUtils::is_real(*t)) {
            int kind = ASRUtils::extract_kind_from_ttype_t(t);
            switch (kind) {
                case 4: return lr_type_f32_s(s);
                case 8: return lr_type_f64_s(s);
            }
        } else if (ASRUtils::is_logical(*t)) {
            return lr_type_i1_s(s);
        } else if (ASRUtils::is_character(*t)) {
            return lr_type_ptr_s(s);
        }
        throw CodeGenError("liric: unsupported type");
        return nullptr; /* unreachable, silences warning */
    }

    lr_operand_desc_t V(uint32_t vreg, lr_type_t *ty) {
        return LR_VREG(vreg, ty);
    }

    lr_operand_desc_t I(int64_t val, lr_type_t *ty) {
        return LR_IMM(val, ty);
    }

    lr_operand_desc_t F(double val, lr_type_t *ty) {
        return LR_IMM_F(val, ty);
    }

    /* ---- TranslationUnit ----------------------------------------------- */

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        /* Modules first (in dependency order) */
        std::vector<std::string> build_order =
            ASRUtils::determine_module_dependencies(x);
        for (auto &item : build_order) {
            ASR::symbol_t *mod = x.m_symtab->get_symbol(item);
            if (mod) visit_symbol(*mod);
        }

        /* Then top-level functions */
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Function_t>(*item.second)) {
                visit_symbol(*item.second);
            }
        }

        /* Then the main program */
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Program_t>(*item.second)) {
                visit_symbol(*item.second);
            }
        }
    }

    /* ---- Program (generates main) -------------------------------------- */

    void visit_Program(const ASR::Program_t &x) {
        /* Generate nested functions first */
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Function_t>(*item.second)) {
                visit_Function(*ASR::down_cast<ASR::Function_t>(item.second));
            }
        }

        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *params[2] = {i32, ptr};
        lr_error_t err;

        lr_session_func_begin(s, "main", i32, params, 2, false, &err);
        uint32_t entry = lr_session_block(s);
        lr_session_set_block(s, entry, &err);

        uint32_t argc = lr_session_param(s, 0);
        uint32_t argv = lr_session_param(s, 1);

        /* Call _lpython_call_initial_functions(argc, argv) */
        {
            lr_type_t *void_ty = lr_type_void_s(s);
            lr_type_t *init_params[2] = {i32, ptr};
            lr_session_declare(s, "_lpython_call_initial_functions",
                               void_ty, init_params, 2, false, &err);
            uint32_t init_id = lr_session_intern(s,
                               "_lpython_call_initial_functions");
            lr_operand_desc_t args[2] = {V(argc, i32), V(argv, ptr)};
            lr_emit_call_void(s, LR_GLOBAL(init_id, ptr), args, 2);
        }

        /* Declare local variables */
        declare_vars(x.m_symtab);

        /* Return block */
        proc_return = lr_session_block(s);

        /* Visit program body */
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }

        /* Return 0 */
        lr_emit_br(s, proc_return);
        lr_session_set_block(s, proc_return, &err);
        lr_emit_ret(s, I(0, i32));

        lr_session_func_end_preserve_ir(s, &err);
    }

    /* ---- Function definition ------------------------------------------- */

    std::string get_mangled_name(const ASR::Function_t &x) {
        return std::string(x.m_name);
    }

    void visit_Function(const ASR::Function_t &x) {
        lr_error_t err;

        /* Skip interfaces (declarations without bodies) */
        if (ASRUtils::get_FunctionType(x)->m_deftype ==
                ASR::deftypeType::Interface) {
            return;
        }

        /* Build parameter type array */
        std::vector<lr_type_t *> param_types;
        for (size_t i = 0; i < x.n_args; i++) {
            ASR::Variable_t *arg = ASRUtils::EXPR2VAR(x.m_args[i]);
            lr_type_t *ty = get_type(arg->m_type);
            /* Pass by pointer for intent(in/out/inout) */
            if (arg->m_intent == ASR::intentType::In ||
                arg->m_intent == ASR::intentType::Out ||
                arg->m_intent == ASR::intentType::InOut ||
                arg->m_intent == ASR::intentType::Unspecified) {
                ty = lr_type_ptr_s(s);
            }
            param_types.push_back(ty);
        }

        /* Return type */
        lr_type_t *ret_ty;
        ASR::Variable_t *ret_var = nullptr;
        if (x.m_return_var) {
            ret_var = ASRUtils::EXPR2VAR(x.m_return_var);
            ret_ty = get_type(ret_var->m_type);
        } else {
            ret_ty = lr_type_void_s(s);
        }

        std::string fname = get_mangled_name(x);
        lr_session_func_begin(s, fname.c_str(), ret_ty,
                              param_types.data(), (uint32_t)param_types.size(),
                              false, &err);

        uint32_t entry = lr_session_block(s);
        lr_session_set_block(s, entry, &err);

        /* Save/restore symtab state */
        auto saved_symtab = lr_symtab;

        /* Map parameters to allocas */
        for (size_t i = 0; i < x.n_args; i++) {
            ASR::Variable_t *arg = ASRUtils::EXPR2VAR(x.m_args[i]);
            uint64_t h = get_hash((ASR::asr_t *)arg);
            uint32_t param_vreg = lr_session_param(s, (uint32_t)i);
            /* Parameter is already a pointer, store it for use */
            lr_symtab[h] = param_vreg;
        }

        /* Declare local variables (including return variable) */
        declare_vars(x.m_symtab);

        /* Return block */
        uint32_t saved_proc_return = proc_return;
        proc_return = lr_session_block(s);

        /* Visit function body */
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }

        /* Return */
        lr_emit_br(s, proc_return);
        lr_session_set_block(s, proc_return, &err);

        if (ret_var) {
            uint64_t h = get_hash((ASR::asr_t *)ret_var);
            auto it = lr_symtab.find(h);
            if (it != lr_symtab.end()) {
                uint32_t ret_ptr = it->second;
                uint32_t ret_val = lr_emit_load(s, ret_ty,
                    V(ret_ptr, lr_type_ptr_s(s)));
                lr_emit_ret(s, V(ret_val, ret_ty));
            } else {
                lr_emit_ret_void(s);
            }
        } else {
            lr_emit_ret_void(s);
        }

        lr_session_func_end_preserve_ir(s, &err);

        /* Restore state */
        lr_symtab = saved_symtab;
        proc_return = saved_proc_return;
    }

    /* ---- Function/Subroutine calls ------------------------------------- */

    /* Pass an argument by reference (Fortran convention).
       If the expression is a Var, return its address.
       Otherwise, evaluate, store to a temporary alloca, return the alloca. */
    lr_operand_desc_t emit_arg_by_ref(ASR::expr_t *expr) {
        if (ASR::is_a<ASR::Var_t>(*expr)) {
            is_target = true;
            visit_expr(*expr);
            is_target = false;
            return V(tmp, lr_type_ptr_s(s));
        }
        /* Non-variable: create temporary */
        visit_expr(*expr);
        uint32_t val = tmp;
        lr_type_t *ty = get_type(ASRUtils::expr_type(expr));
        uint32_t tmp_alloca = lr_emit_alloca(s, ty);
        lr_emit_store(s, V(val, ty), V(tmp_alloca, lr_type_ptr_s(s)));
        return V(tmp_alloca, lr_type_ptr_s(s));
    }

    /* Ensure a function is declared (for forward references or externals) */
    uint32_t ensure_function_declared(ASR::Function_t *fn) {
        std::string fname = get_mangled_name(*fn);
        /* Check if already defined as a function in the module */
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, fname.c_str())) {
            return lr_session_intern(s, fname.c_str());
        }
        /* Not yet defined — create a declaration with correct signature */
        lr_error_t err;
        std::vector<lr_type_t *> param_types;
        for (size_t i = 0; i < fn->n_args; i++) {
            ASR::Variable_t *arg = ASRUtils::EXPR2VAR(fn->m_args[i]);
            (void)arg;
            param_types.push_back(lr_type_ptr_s(s));
        }
        lr_type_t *ret_ty;
        if (fn->m_return_var) {
            ret_ty = get_type(ASRUtils::EXPR2VAR(fn->m_return_var)->m_type);
        } else {
            ret_ty = lr_type_void_s(s);
        }
        lr_session_declare(s, fname.c_str(), ret_ty,
                           param_types.data(), (uint32_t)param_types.size(),
                           false, &err);
        return lr_session_intern(s, fname.c_str());
    }

    void visit_SubroutineCall(const ASR::SubroutineCall_t &x) {
        ASR::Function_t *fn = ASR::down_cast<ASR::Function_t>(
            ASRUtils::symbol_get_past_external(x.m_name));
        uint32_t fn_id = ensure_function_declared(fn);

        /* Evaluate arguments (Fortran pass-by-reference) */
        std::vector<lr_operand_desc_t> arg_ops;
        for (size_t i = 0; i < x.n_args; i++) {
            if (x.m_args[i].m_value) {
                arg_ops.push_back(emit_arg_by_ref(x.m_args[i].m_value));
            } else {
                arg_ops.push_back(LR_NULL(lr_type_ptr_s(s)));
            }
        }

        lr_emit_call_void(s, LR_GLOBAL(fn_id, lr_type_ptr_s(s)),
                          arg_ops.data(), (uint32_t)arg_ops.size());
    }

    void visit_FunctionCall(const ASR::FunctionCall_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        ASR::Function_t *fn = ASR::down_cast<ASR::Function_t>(
            ASRUtils::symbol_get_past_external(x.m_name));

        lr_type_t *ret_ty = get_type(x.m_type);
        uint32_t fn_id = ensure_function_declared(fn);

        /* Evaluate arguments (Fortran pass-by-reference) */
        std::vector<lr_operand_desc_t> arg_ops;
        for (size_t i = 0; i < x.n_args; i++) {
            if (x.m_args[i].m_value) {
                arg_ops.push_back(emit_arg_by_ref(x.m_args[i].m_value));
            } else {
                arg_ops.push_back(LR_NULL(lr_type_ptr_s(s)));
            }
        }

        tmp = lr_emit_call(s, ret_ty, LR_GLOBAL(fn_id, lr_type_ptr_s(s)),
                           arg_ops.data(), (uint32_t)arg_ops.size());
    }

    /* ---- Variable declarations ----------------------------------------- */

    void declare_vars(SymbolTable *symtab) {
        for (auto &item : symtab->get_scope()) {
            if (!ASR::is_a<ASR::Variable_t>(*item.second)) continue;
            ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(item.second);
            uint64_t h = get_hash((ASR::asr_t *)v);
            /* Skip if already registered (e.g. function parameters) */
            if (lr_symtab.find(h) != lr_symtab.end()) continue;
            /* Allocate stack space for locals, return vars, and unspecified */
            if (v->m_intent == ASR::intentType::Local ||
                v->m_intent == ASR::intentType::Unspecified ||
                v->m_intent == ASR::intentType::ReturnVar) {
                lr_type_t *ty = get_type(v->m_type);
                uint32_t alloca_vreg = lr_emit_alloca(s, ty);
                lr_symtab[h] = alloca_vreg;
            }
        }
    }

    /* ---- Var ----------------------------------------------------------- */

    void visit_Var(const ASR::Var_t &x) {
        ASR::symbol_t *sym = ASRUtils::symbol_get_past_external(x.m_v);
        if (ASR::is_a<ASR::Variable_t>(*sym)) {
            ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(sym);
            uint64_t h = get_hash((ASR::asr_t *)v);
            auto it = lr_symtab.find(h);
            if (it == lr_symtab.end()) {
                throw CodeGenError("liric: variable not found: "
                                   + std::string(v->m_name));
            }
            uint32_t ptr_vreg = it->second;
            if (is_target) {
                tmp = ptr_vreg;
            } else {
                lr_type_t *ty = get_type(v->m_type);
                tmp = lr_emit_load(s, ty, V(ptr_vreg, lr_type_ptr_s(s)));
            }
        } else {
            throw CodeGenError("liric: unsupported symbol type in Var");
        }
    }

    /* ---- Assignment ---------------------------------------------------- */

    void visit_Assignment(const ASR::Assignment_t &x) {
        /* Get target address */
        is_target = true;
        visit_expr(*x.m_target);
        is_target = false;
        uint32_t target_ptr = tmp;

        /* Evaluate value */
        visit_expr(*x.m_value);
        uint32_t val = tmp;

        lr_type_t *ty = get_type(ASRUtils::expr_type(x.m_value));
        lr_emit_store(s, V(val, ty), V(target_ptr, lr_type_ptr_s(s)));
    }

    /* ---- Integer constants and operations ------------------------------ */

    void visit_IntegerConstant(const ASR::IntegerConstant_t &x) {
        int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        lr_type_t *ty = get_type(x.m_type);
        (void)kind;
        /* Constants are not emitted as instructions in liric.
           We store the vreg from an immediate-to-vreg move. */
        tmp = lr_emit_add(s, ty, I(x.m_n, ty), I(0, ty));
    }

    void visit_IntegerBinOp(const ASR::IntegerBinOp_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *ty = get_type(x.m_type);
        lr_operand_desc_t l = V(left, ty);
        lr_operand_desc_t r = V(right, ty);
        switch (x.m_op) {
            case ASR::binopType::Add:
                tmp = lr_emit_add(s, ty, l, r); break;
            case ASR::binopType::Sub:
                tmp = lr_emit_sub(s, ty, l, r); break;
            case ASR::binopType::Mul:
                tmp = lr_emit_mul(s, ty, l, r); break;
            case ASR::binopType::Div:
                tmp = lr_emit_sdiv(s, ty, l, r); break;
            case ASR::binopType::Pow: {
                /* TODO: handle power operation properly */
                throw CodeGenError("liric: integer Pow not yet implemented");
            }
            default:
                throw CodeGenError("liric: unsupported integer binop");
        }
    }

    void visit_IntegerUnaryMinus(const ASR::IntegerUnaryMinus_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        lr_type_t *ty = get_type(x.m_type);
        tmp = lr_emit_neg(s, ty, V(tmp, ty));
    }

    void visit_IntegerCompare(const ASR::IntegerCompare_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *ty = get_type(ASRUtils::expr_type(x.m_left));
        lr_operand_desc_t l = V(left, ty);
        lr_operand_desc_t r = V(right, ty);
        int pred;
        switch (x.m_op) {
            case ASR::cmpopType::Eq:  pred = LR_CMP_EQ;  break;
            case ASR::cmpopType::NotEq: pred = LR_CMP_NE; break;
            case ASR::cmpopType::Gt:  pred = LR_CMP_SGT; break;
            case ASR::cmpopType::GtE: pred = LR_CMP_SGE; break;
            case ASR::cmpopType::Lt:  pred = LR_CMP_SLT; break;
            case ASR::cmpopType::LtE: pred = LR_CMP_SLE; break;
            default:
                throw CodeGenError("liric: unsupported compare op");
        }
        tmp = lr_emit_icmp(s, pred, l, r);
    }

    /* ---- Real constants and operations --------------------------------- */

    void visit_RealConstant(const ASR::RealConstant_t &x) {
        lr_type_t *ty = get_type(x.m_type);
        tmp = lr_emit_fadd(s, ty, F(x.m_r, ty), F(0.0, ty));
    }

    void visit_RealBinOp(const ASR::RealBinOp_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *ty = get_type(x.m_type);
        lr_operand_desc_t l = V(left, ty);
        lr_operand_desc_t r = V(right, ty);
        switch (x.m_op) {
            case ASR::binopType::Add:
                tmp = lr_emit_fadd(s, ty, l, r); break;
            case ASR::binopType::Sub:
                tmp = lr_emit_fsub(s, ty, l, r); break;
            case ASR::binopType::Mul:
                tmp = lr_emit_fmul(s, ty, l, r); break;
            case ASR::binopType::Div:
                tmp = lr_emit_fdiv(s, ty, l, r); break;
            case ASR::binopType::Pow:
                throw CodeGenError("liric: real Pow not yet implemented");
            default:
                throw CodeGenError("liric: unsupported real binop");
        }
    }

    void visit_RealCompare(const ASR::RealCompare_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *ty = get_type(ASRUtils::expr_type(x.m_left));
        lr_operand_desc_t l = V(left, ty);
        lr_operand_desc_t r = V(right, ty);
        int pred;
        switch (x.m_op) {
            case ASR::cmpopType::Eq:  pred = LR_FCMP_OEQ; break;
            case ASR::cmpopType::NotEq: pred = LR_FCMP_ONE; break;
            case ASR::cmpopType::Gt:  pred = LR_FCMP_OGT; break;
            case ASR::cmpopType::GtE: pred = LR_FCMP_OGE; break;
            case ASR::cmpopType::Lt:  pred = LR_FCMP_OLT; break;
            case ASR::cmpopType::LtE: pred = LR_FCMP_OLE; break;
            default:
                throw CodeGenError("liric: unsupported real compare op");
        }
        tmp = lr_emit_fcmp(s, pred, l, r);
    }

    void visit_RealUnaryMinus(const ASR::RealUnaryMinus_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        lr_type_t *ty = get_type(x.m_type);
        tmp = lr_emit_fneg(s, ty, V(tmp, ty));
    }

    /* ---- Logical ------------------------------------------------------- */

    void visit_LogicalConstant(const ASR::LogicalConstant_t &x) {
        lr_type_t *i1 = lr_type_i1_s(s);
        tmp = lr_emit_add(s, i1, I(x.m_value ? 1 : 0, i1), I(0, i1));
    }

    void visit_LogicalBinOp(const ASR::LogicalBinOp_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *i1 = lr_type_i1_s(s);
        lr_operand_desc_t l = V(left, i1);
        lr_operand_desc_t r = V(right, i1);
        switch (x.m_op) {
            case ASR::logicalbinopType::And:
                tmp = lr_emit_and(s, i1, l, r); break;
            case ASR::logicalbinopType::Or:
                tmp = lr_emit_or(s, i1, l, r); break;
            case ASR::logicalbinopType::Xor:
                tmp = lr_emit_xor(s, i1, l, r); break;
            case ASR::logicalbinopType::NEqv:
                tmp = lr_emit_xor(s, i1, l, r); break;
            case ASR::logicalbinopType::Eqv:
                tmp = lr_emit_icmp(s, LR_CMP_EQ, l, r); break;
            default:
                throw CodeGenError("liric: unsupported logical binop");
        }
    }

    void visit_LogicalNot(const ASR::LogicalNot_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        lr_type_t *i1 = lr_type_i1_s(s);
        tmp = lr_emit_xor(s, i1, V(tmp, i1), I(1, i1));
    }

    /* ---- Control flow -------------------------------------------------- */

    void visit_If(const ASR::If_t &x) {
        visit_expr(*x.m_test);
        uint32_t cond = tmp;
        lr_type_t *i1 = lr_type_i1_s(s);
        lr_error_t err;

        uint32_t then_block = lr_session_block(s);
        uint32_t else_block = lr_session_block(s);
        uint32_t merge_block = lr_session_block(s);

        lr_emit_condbr(s, V(cond, i1),
                       x.n_orelse > 0 ? then_block : then_block,
                       x.n_orelse > 0 ? else_block : merge_block);

        lr_session_set_block(s, then_block, &err);
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }
        lr_emit_br(s, merge_block);

        if (x.n_orelse > 0) {
            lr_session_set_block(s, else_block, &err);
            for (size_t i = 0; i < x.n_orelse; i++) {
                visit_stmt(*x.m_orelse[i]);
            }
            lr_emit_br(s, merge_block);
        }

        lr_session_set_block(s, merge_block, &err);
    }

    void visit_WhileLoop(const ASR::WhileLoop_t &x) {
        lr_error_t err;
        lr_type_t *i1 = lr_type_i1_s(s);

        uint32_t cond_block = lr_session_block(s);
        uint32_t body_block = lr_session_block(s);
        uint32_t end_block = lr_session_block(s);

        loop_head_stack.push_back(cond_block);
        loop_end_stack.push_back(end_block);

        lr_emit_br(s, cond_block);
        lr_session_set_block(s, cond_block, &err);
        visit_expr(*x.m_test);
        lr_emit_condbr(s, V(tmp, i1), body_block, end_block);

        lr_session_set_block(s, body_block, &err);
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }
        lr_emit_br(s, cond_block);

        lr_session_set_block(s, end_block, &err);
        loop_head_stack.pop_back();
        loop_end_stack.pop_back();
    }

    void visit_DoLoop(const ASR::DoLoop_t &x) {
        lr_error_t err;
        lr_type_t *i1 = lr_type_i1_s(s);

        /* Evaluate loop bounds */
        visit_expr(*x.m_head.m_start);
        uint32_t start_val = tmp;
        visit_expr(*x.m_head.m_end);
        uint32_t end_val = tmp;
        lr_type_t *loop_ty = get_type(ASRUtils::expr_type(x.m_head.m_v));

        uint32_t step_val;
        if (x.m_head.m_increment) {
            visit_expr(*x.m_head.m_increment);
            step_val = tmp;
        } else {
            step_val = lr_emit_add(s, loop_ty, I(1, loop_ty), I(0, loop_ty));
        }

        /* Store start value to loop variable */
        is_target = true;
        visit_expr(*x.m_head.m_v);
        is_target = false;
        uint32_t loop_var_ptr = tmp;
        lr_emit_store(s, V(start_val, loop_ty), V(loop_var_ptr, lr_type_ptr_s(s)));

        uint32_t cond_block = lr_session_block(s);
        uint32_t body_block = lr_session_block(s);
        uint32_t incr_block = lr_session_block(s);
        uint32_t end_block = lr_session_block(s);

        loop_head_stack.push_back(incr_block);
        loop_end_stack.push_back(end_block);

        /* Branch to condition check */
        lr_emit_br(s, cond_block);

        /* Condition: loop_var <= end (or >= end if step < 0) */
        lr_session_set_block(s, cond_block, &err);
        uint32_t cur_val = lr_emit_load(s, loop_ty,
                                        V(loop_var_ptr, lr_type_ptr_s(s)));
        uint32_t cond = lr_emit_icmp(s, LR_CMP_SLE,
                                     V(cur_val, loop_ty),
                                     V(end_val, loop_ty));
        lr_emit_condbr(s, V(cond, i1), body_block, end_block);

        /* Body */
        lr_session_set_block(s, body_block, &err);
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }
        lr_emit_br(s, incr_block);

        /* Increment */
        lr_session_set_block(s, incr_block, &err);
        uint32_t next_val = lr_emit_load(s, loop_ty,
                                         V(loop_var_ptr, lr_type_ptr_s(s)));
        next_val = lr_emit_add(s, loop_ty, V(next_val, loop_ty),
                               V(step_val, loop_ty));
        lr_emit_store(s, V(next_val, loop_ty),
                      V(loop_var_ptr, lr_type_ptr_s(s)));
        lr_emit_br(s, cond_block);

        /* End */
        lr_session_set_block(s, end_block, &err);
        loop_head_stack.pop_back();
        loop_end_stack.pop_back();
    }

    void visit_Exit(const ASR::Exit_t & /* x */) {
        if (loop_end_stack.empty()) {
            throw CodeGenError("liric: exit outside loop");
        }
        lr_emit_br(s, loop_end_stack.back());
        uint32_t dead = lr_session_block(s);
        lr_session_set_block(s, dead, NULL);
    }

    void visit_Cycle(const ASR::Cycle_t & /* x */) {
        if (loop_head_stack.empty()) {
            throw CodeGenError("liric: cycle outside loop");
        }
        lr_emit_br(s, loop_head_stack.back());
        uint32_t dead = lr_session_block(s);
        lr_session_set_block(s, dead, NULL);
    }

    void visit_Return(const ASR::Return_t & /* x */) {
        lr_emit_br(s, proc_return);
        uint32_t dead = lr_session_block(s);
        lr_session_set_block(s, dead, NULL);
    }

    /* ---- Print (basic: calls _lfortran_printf) ------------------------- */

    void visit_Print(const ASR::Print_t &x) {
        if (!x.m_text) return;
        visit_expr(*x.m_text);
        /* tmp now holds the formatted string.
           For now, we handle only StringFormat with integer values. */
        /* TODO: implement full Print via _lfortran_printf */
        throw CodeGenError("liric: Print not yet implemented "
                           "(need StringFormat support)");
    }

    void visit_StringFormat(const ASR::StringFormat_t & /* x */) {
        /* TODO: implement StringFormat */
        throw CodeGenError("liric: StringFormat not yet implemented");
    }

    /* ---- Stop / ErrorStop ---------------------------------------------- */

    void visit_Stop(const ASR::Stop_t & /* x */) {
        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_error_t err;

        lr_type_t *exit_params[1] = {i32};
        lr_session_declare(s, "exit", lr_type_void_s(s),
                           exit_params, 1, false, &err);
        uint32_t exit_id = lr_session_intern(s, "exit");
        lr_operand_desc_t args[1] = {I(0, i32)};
        lr_emit_call_void(s, LR_GLOBAL(exit_id, ptr), args, 1);
        lr_emit_unreachable(s);
    }

    void visit_ErrorStop(const ASR::ErrorStop_t & /* x */) {
        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_error_t err;

        lr_type_t *exit_params[1] = {i32};
        lr_session_declare(s, "exit", lr_type_void_s(s),
                           exit_params, 1, false, &err);
        uint32_t exit_id = lr_session_intern(s, "exit");
        lr_operand_desc_t args[1] = {I(1, i32)};
        lr_emit_call_void(s, LR_GLOBAL(exit_id, ptr), args, 1);
        lr_emit_unreachable(s);
    }

    /* ---- Cast ---------------------------------------------------------- */

    void visit_Cast(const ASR::Cast_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        uint32_t arg = tmp;
        lr_type_t *src_ty = get_type(ASRUtils::expr_type(x.m_arg));
        lr_type_t *dst_ty = get_type(x.m_type);
        lr_operand_desc_t v = V(arg, src_ty);

        switch (x.m_kind) {
            case ASR::cast_kindType::IntegerToReal:
                tmp = lr_emit_sitofp(s, dst_ty, v); break;
            case ASR::cast_kindType::RealToInteger:
                tmp = lr_emit_fptosi(s, dst_ty, v); break;
            case ASR::cast_kindType::IntegerToInteger:
                tmp = lr_emit_sextortrunc(s, dst_ty, v); break;
            case ASR::cast_kindType::RealToReal:
                if (dst_ty == src_ty) { /* no-op */ }
                else if (lr_type_width(s, dst_ty) > lr_type_width(s, src_ty))
                    tmp = lr_emit_fpext(s, dst_ty, v);
                else
                    tmp = lr_emit_fptrunc(s, dst_ty, v);
                break;
            case ASR::cast_kindType::IntegerToLogical:
                tmp = lr_emit_icmp(s, LR_CMP_NE, v, I(0, src_ty));
                break;
            case ASR::cast_kindType::LogicalToInteger:
                tmp = lr_emit_zext(s, dst_ty, v); break;
            default:
                throw CodeGenError("liric: unsupported cast kind");
        }
    }
};

/* ---- Entry point ------------------------------------------------------- */

Result<int> asr_to_liric(ASR::TranslationUnit_t &asr,
    Allocator &al, const std::string &filename,
    CompilerOptions &co, diag::Diagnostics &diagnostics,
    int liric_backend)
{
    lr_session_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = LR_MODE_DIRECT;
    cfg.backend = (lr_session_backend_t)liric_backend;
    lr_error_t err;

    lr_session_t *session = lr_session_create(&cfg, &err);
    if (!session) {
        diagnostics.add(diag::Diagnostic(
            "liric: failed to create session: " + std::string(err.msg),
            diag::Level::Error, diag::Stage::CodeGen, {}));
        return Error();
    }

    try {
        ASRToLiricVisitor v(session, al, co, diagnostics);
        v.visit_asr((ASR::asr_t &)asr);
    } catch (const CodeGenError &e) {
        lr_session_destroy(session);
        diagnostics.diagnostics.push_back(e.d);
        return Error();
    }

    if (lr_session_emit_object(session, filename.c_str(), &err) != 0) {
        lr_session_destroy(session);
        diagnostics.add(diag::Diagnostic(
            "liric: failed to emit object file: " + std::string(err.msg),
            diag::Level::Error, diag::Stage::CodeGen, {}));
        return Error();
    }

    lr_session_destroy(session);
    return 0;
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
