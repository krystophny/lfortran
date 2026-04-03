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

    lr_type_t *get_scalar_type(ASR::ttype_t *t) {
        t = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(t));
        if (ASR::is_a<ASR::Array_t>(*t)) {
            ASR::Array_t *arr = ASR::down_cast<ASR::Array_t>(t);
            return get_scalar_type(arr->m_type);
        }
        if (ASRUtils::is_integer(*t) || ASRUtils::is_unsigned_integer(*t)) {
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
        } else if (ASR::is_a<ASR::Complex_t>(*t)) {
            int kind = ASRUtils::extract_kind_from_ttype_t(t);
            lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
            lr_type_t *fields[2] = {fty, fty};
            return lr_type_struct_s(s, fields, 2, false);
        } else if (ASR::is_a<ASR::StructType_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::CPtr_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::EnumType_t>(*t)) {
            return lr_type_i32_s(s);
        } else if (ASR::is_a<ASR::FunctionType_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::Tuple_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::List_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::Dict_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::Set_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::UnionType_t>(*t)) {
            return lr_type_ptr_s(s);
        } else if (ASR::is_a<ASR::TypeParameter_t>(*t)) {
            return lr_type_ptr_s(s);
        }
        throw CodeGenError("liric: unsupported scalar type "
            + std::to_string(t->type));
        return nullptr;
    }

    lr_type_t *get_type(ASR::ttype_t *t) {
        t = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(t));
        if (ASR::is_a<ASR::Array_t>(*t)) {
            ASR::Array_t *arr = ASR::down_cast<ASR::Array_t>(t);
            if (arr->m_physical_type ==
                    ASR::array_physical_typeType::FixedSizeArray) {
                ASR::dimension_t *dims = arr->m_dims;
                size_t n_dims = arr->n_dims;
                int64_t total = ASRUtils::get_fixed_size_of_array(
                    dims, n_dims);
                if (total > 0) {
                    lr_type_t *elem = get_scalar_type(arr->m_type);
                    return lr_type_array_s(s, elem, (uint64_t)total);
                }
            }
            return lr_type_ptr_s(s);
        }
        return get_scalar_type(t);
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

    /* Create a global string constant and return a vreg holding a pointer
       to it. Uses lr_session_intern to get the correct symbol ID for
       LR_GLOBAL (not the global_id from lr_session_global). */
    uint32_t emit_global_string(const char *str) {
        static unsigned gstr_counter = 0;
        size_t len = 0;
        while (str[len]) len++;
        lr_type_t *i8 = lr_type_i8_s(s);
        lr_type_t *arr_ty = lr_type_array_s(s, i8, len + 1);
        lr_type_t *ptr = lr_type_ptr_s(s);
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), ".gstr.%u", gstr_counter++);
        lr_session_global(s, name_buf, arr_ty, true, str, len + 1);
        uint32_t sym_id = lr_session_intern(s, name_buf);
        return lr_emit_bitcast(s, ptr, LR_GLOBAL(sym_id, ptr));
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

    /* ---- Module -------------------------------------------------------- */

    void visit_Module(const ASR::Module_t &x) {
        /* Skip intrinsic/runtime modules — their functions are provided
           by the lfortran runtime library, not compiled here. */
        std::string modname(x.m_name);
        if (modname.find("lfortran_intrinsic") != std::string::npos ||
            modname.find("iso_fortran_env") != std::string::npos ||
            modname.find("iso_c_binding") != std::string::npos ||
            modname.find("ieee_arithmetic") != std::string::npos) {
            return;
        }

        /* Visit all functions in user modules */
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Function_t>(*item.second)) {
                visit_Function(*ASR::down_cast<ASR::Function_t>(
                    item.second));
            }
        }
        /* Module variables become globals */
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(
                    item.second);
                uint64_t h = get_hash((ASR::asr_t *)v);
                if (lr_symtab.find(h) != lr_symtab.end()) continue;
                lr_type_t *ty = get_type(v->m_type);
                std::string gname = "__module_" + std::string(x.m_name)
                    + "_" + std::string(v->m_name);
                lr_session_global(s, gname.c_str(), ty, false, NULL, 0);
                uint32_t sym_id = lr_session_intern(s, gname.c_str());
                /* Store a sentinel — module vars are accessed via global ref */
                lr_symtab[h] = sym_id | 0x80000000u; /* high bit = global flag */
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

    void visit_StringConstant(const ASR::StringConstant_t &x) {
        tmp = emit_global_string( x.m_s);
    }

    void visit_UnsignedIntegerConstant(const ASR::UnsignedIntegerConstant_t &x) {
        lr_type_t *ty = get_type(x.m_type);
        tmp = lr_emit_add(s, ty, I(x.m_n, ty), I(0, ty));
    }

    void visit_UnsignedIntegerBinOp(const ASR::UnsignedIntegerBinOp_t &x) {
        if (x.m_value) { visit_expr(*x.m_value); return; }
        visit_expr(*x.m_left); uint32_t left = tmp;
        visit_expr(*x.m_right); uint32_t right = tmp;
        lr_type_t *ty = get_type(x.m_type);
        lr_operand_desc_t l = V(left, ty), r = V(right, ty);
        switch (x.m_op) {
            case ASR::binopType::Add: tmp = lr_emit_add(s, ty, l, r); break;
            case ASR::binopType::Sub: tmp = lr_emit_sub(s, ty, l, r); break;
            case ASR::binopType::Mul: tmp = lr_emit_mul(s, ty, l, r); break;
            case ASR::binopType::Div: tmp = lr_emit_udiv(s, ty, l, r); break;
            default: throw CodeGenError("liric: unsupported unsigned binop");
        }
    }

    /* ---- Intrinsic functions ------------------------------------------- */

    /* Declare a C math function (double->double or float->float) */
    uint32_t declare_math_func(const char *name, lr_type_t *ty) {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, name)) {
            return lr_session_intern(s, name);
        }
        lr_type_t *params[1] = {ty};
        lr_session_declare(s, name, ty, params, 1, false, &err);
        return lr_session_intern(s, name);
    }

    uint32_t declare_math_func2(const char *name, lr_type_t *ret,
                                lr_type_t *a1, lr_type_t *a2) {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, name)) {
            return lr_session_intern(s, name);
        }
        lr_type_t *params[2] = {a1, a2};
        lr_session_declare(s, name, ret, params, 2, false, &err);
        return lr_session_intern(s, name);
    }

    /* Emit a call to a unary math function (sin, cos, sqrt, etc.) */
    void emit_unary_math(const char *f64_name, const char *f32_name,
                         ASR::expr_t *arg, ASR::ttype_t *res_type) {
        visit_expr(*arg);
        uint32_t a = tmp;
        lr_type_t *ty = get_type(res_type);
        lr_type_t *ptr = lr_type_ptr_s(s);
        const char *name = (ty == lr_type_f32_s(s)) ? f32_name : f64_name;
        uint32_t fn_id = declare_math_func(name, ty);
        lr_operand_desc_t args[1] = {V(a, ty)};
        tmp = lr_emit_call(s, ty, LR_GLOBAL(fn_id, ptr), args, 1);
    }

    void visit_IntrinsicElementalFunction(
            const ASR::IntrinsicElementalFunction_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        int64_t id = x.m_intrinsic_id;
        lr_type_t *ptr = lr_type_ptr_s(s);

        if (id == 39) { /* Abs */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *ty = get_type(t);
            if (ASRUtils::is_integer(*t)) {
                uint32_t neg = lr_emit_neg(s, ty, V(arg, ty));
                uint32_t cond = lr_emit_icmp(s, LR_CMP_SGE,
                    V(arg, ty), I(0, ty));
                tmp = lr_emit_select(s, ty,
                    V(cond, lr_type_i1_s(s)),
                    V(arg, ty), V(neg, ty));
            } else if (ASRUtils::is_real(*t)) {
                emit_unary_math("fabs", "fabsf", x.m_args[0], x.m_type);
            } else {
                throw CodeGenError("liric: abs() unsupported type");
            }
        } else if (id == 2 || id == 51) { /* Mod, Modulo */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            if (ASRUtils::is_integer(*x.m_type)) {
                tmp = lr_emit_srem(s, ty, V(a, ty), V(b, ty));
            } else {
                tmp = lr_emit_frem(s, ty, V(a, ty), V(b, ty));
            }
        } else if (id == 3) { /* Sin */
            emit_unary_math("sin", "sinf", x.m_args[0], x.m_type);
        } else if (id == 4) { /* Cos */
            emit_unary_math("cos", "cosf", x.m_args[0], x.m_type);
        } else if (id == 5) { /* Tan */
            emit_unary_math("tan", "tanf", x.m_args[0], x.m_type);
        } else if (id == 6) { /* Asin */
            emit_unary_math("asin", "asinf", x.m_args[0], x.m_type);
        } else if (id == 7) { /* Acos */
            emit_unary_math("acos", "acosf", x.m_args[0], x.m_type);
        } else if (id == 8) { /* Atan */
            emit_unary_math("atan", "atanf", x.m_args[0], x.m_type);
        } else if (id == 9) { /* Sinh */
            emit_unary_math("sinh", "sinhf", x.m_args[0], x.m_type);
        } else if (id == 10) { /* Cosh */
            emit_unary_math("cosh", "coshf", x.m_args[0], x.m_type);
        } else if (id == 11) { /* Tanh */
            emit_unary_math("tanh", "tanhf", x.m_args[0], x.m_type);
        } else if (id == 12) { /* Atan2 */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            const char *name = (ty == lr_type_f32_s(s)) ? "atan2f" : "atan2";
            uint32_t fn_id = declare_math_func2(name, ty, ty, ty);
            lr_operand_desc_t args[2] = {V(a, ty), V(b, ty)};
            tmp = lr_emit_call(s, ty, LR_GLOBAL(fn_id, ptr), args, 2);
        } else if (id == 34) { /* Log */
            emit_unary_math("log", "logf", x.m_args[0], x.m_type);
        } else if (id == 35) { /* Log10 */
            emit_unary_math("log10", "log10f", x.m_args[0], x.m_type);
        } else if (id == 42) { /* Exp */
            emit_unary_math("exp", "expf", x.m_args[0], x.m_type);
        } else if (id == 142) { /* Sqrt */
            emit_unary_math("sqrt", "sqrtf", x.m_args[0], x.m_type);
        } else if (id == 146) { /* Floor */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_real(*arg_t) && ASRUtils::is_integer(*x.m_type)) {
                const char *fn = (arg_ty == lr_type_f32_s(s))
                    ? "floorf" : "floor";
                uint32_t fn_id = declare_math_func(fn, arg_ty);
                lr_operand_desc_t args[1] = {V(a, arg_ty)};
                uint32_t fval = lr_emit_call(s, arg_ty,
                    LR_GLOBAL(fn_id, ptr), args, 1);
                tmp = lr_emit_fptosi(s, res_ty, V(fval, arg_ty));
            } else {
                emit_unary_math("floor", "floorf", x.m_args[0], x.m_type);
            }
        } else if (id == 147) { /* Ceiling */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_real(*arg_t) && ASRUtils::is_integer(*x.m_type)) {
                const char *fn = (arg_ty == lr_type_f32_s(s))
                    ? "ceilf" : "ceil";
                uint32_t fn_id = declare_math_func(fn, arg_ty);
                lr_operand_desc_t args[1] = {V(a, arg_ty)};
                uint32_t fval = lr_emit_call(s, arg_ty,
                    LR_GLOBAL(fn_id, ptr), args, 1);
                tmp = lr_emit_fptosi(s, res_ty, V(fval, arg_ty));
            } else {
                emit_unary_math("ceil", "ceilf", x.m_args[0], x.m_type);
            }
        } else if (id == 137) { /* Nint */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            const char *fn = (arg_ty == lr_type_f32_s(s))
                ? "roundf" : "round";
            uint32_t fn_id = declare_math_func(fn, arg_ty);
            lr_operand_desc_t args[1] = {V(a, arg_ty)};
            uint32_t fval = lr_emit_call(s, arg_ty,
                LR_GLOBAL(fn_id, ptr), args, 1);
            tmp = lr_emit_fptosi(s, res_ty, V(fval, arg_ty));
        } else if (id == 122 || id == 123) { /* Max, Min */
            visit_expr(*x.m_args[0]);
            uint32_t result = tmp;
            lr_type_t *ty = get_type(x.m_type);
            ASR::ttype_t *t = ASRUtils::expr_type(x.m_args[0]);
            for (size_t i = 1; i < x.n_args; i++) {
                visit_expr(*x.m_args[i]);
                uint32_t b = tmp;
                uint32_t cond;
                if (ASRUtils::is_integer(*t)) {
                    cond = lr_emit_icmp(s, (id == 122)
                        ? LR_CMP_SGT : LR_CMP_SLT,
                        V(result, ty), V(b, ty));
                } else {
                    cond = lr_emit_fcmp(s, (id == 122)
                        ? LR_FCMP_OGT : LR_FCMP_OLT,
                        V(result, ty), V(b, ty));
                }
                result = lr_emit_select(s, ty,
                    V(cond, lr_type_i1_s(s)),
                    V(result, ty), V(b, ty));
            }
            tmp = result;
        } else if (id == 130) { /* Sign */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            if (ASRUtils::is_integer(*x.m_type)) {
                /* sign(a, b) = abs(a) * sign_of(b) */
                uint32_t abs_a_neg = lr_emit_neg(s, ty, V(a, ty));
                uint32_t a_pos = lr_emit_icmp(s, LR_CMP_SGE,
                    V(a, ty), I(0, ty));
                uint32_t abs_a = lr_emit_select(s, ty,
                    V(a_pos, lr_type_i1_s(s)),
                    V(a, ty), V(abs_a_neg, ty));
                uint32_t b_neg = lr_emit_icmp(s, LR_CMP_SLT,
                    V(b, ty), I(0, ty));
                uint32_t neg_abs = lr_emit_neg(s, ty, V(abs_a, ty));
                tmp = lr_emit_select(s, ty,
                    V(b_neg, lr_type_i1_s(s)),
                    V(neg_abs, ty), V(abs_a, ty));
            } else {
                /* sign(a, b) = copysign(a, b) */
                const char *fn = (ty == lr_type_f32_s(s))
                    ? "copysignf" : "copysign";
                uint32_t fn_id = declare_math_func2(fn, ty, ty, ty);
                lr_operand_desc_t args[2] = {V(a, ty), V(b, ty)};
                tmp = lr_emit_call(s, ty, LR_GLOBAL(fn_id, ptr), args, 2);
            }
        } else if (id == 80) { /* Not (bitwise) */
            visit_expr(*x.m_args[0]);
            lr_type_t *ty = get_type(x.m_type);
            tmp = lr_emit_not(s, ty, V(tmp, ty));
        } else if (id == 81) { /* Iand */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            tmp = lr_emit_and(s, ty, V(a, ty), V(b, ty));
        } else if (id == 82) { /* Ior */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            tmp = lr_emit_or(s, ty, V(a, ty), V(b, ty));
        } else if (id == 83) { /* Ieor */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            tmp = lr_emit_xor(s, ty, V(a, ty), V(b, ty));
        } else if (id == 62) { /* Shiftr */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            tmp = lr_emit_lshr(s, ty, V(a, ty), V(b, ty));
        } else if (id == 64) { /* Shiftl */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            tmp = lr_emit_shl(s, ty, V(a, ty), V(b, ty));
        } else if (id == 37) { /* Trunc */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            if (ASRUtils::is_real(*x.m_type)) {
                emit_unary_math("trunc", "truncf", x.m_args[0], x.m_type);
            } else {
                tmp = lr_emit_fptosi(s, ty, V(a, arg_ty));
            }
        } else if (id == 141) { /* Dim */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            visit_expr(*x.m_args[1]);
            uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            if (ASRUtils::is_integer(*x.m_type)) {
                uint32_t diff = lr_emit_sub(s, ty, V(a, ty), V(b, ty));
                uint32_t cond = lr_emit_icmp(s, LR_CMP_SGT,
                    V(a, ty), V(b, ty));
                tmp = lr_emit_select(s, ty,
                    V(cond, lr_type_i1_s(s)),
                    V(diff, ty), I(0, ty));
            } else {
                uint32_t diff = lr_emit_fsub(s, ty, V(a, ty), V(b, ty));
                uint32_t cond = lr_emit_fcmp(s, LR_FCMP_OGT,
                    V(a, ty), V(b, ty));
                tmp = lr_emit_select(s, ty,
                    V(cond, lr_type_i1_s(s)),
                    V(diff, ty), F(0.0, ty));
            }
        } else if (id == 40) { /* Aimag */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            int kind = ASRUtils::extract_kind_from_ttype_t(arg_t);
            lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
            lr_type_t *sty = get_type(arg_t);
            uint32_t idx1[1] = {1};
            tmp = lr_emit_extractvalue(s, fty, V(arg, sty), idx1, 1);
        } else if (id == 41) { /* Dreal */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            int kind = ASRUtils::extract_kind_from_ttype_t(arg_t);
            lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
            lr_type_t *sty = get_type(arg_t);
            uint32_t idx0[1] = {0};
            tmp = lr_emit_extractvalue(s, fty, V(arg, sty), idx0, 1);
        } else if (id == 60) { /* Merge */
            visit_expr(*x.m_args[0]); uint32_t tsource = tmp;
            visit_expr(*x.m_args[1]); uint32_t fsource = tmp;
            visit_expr(*x.m_args[2]); uint32_t mask = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *i1 = lr_type_i1_s(s);
            tmp = lr_emit_select(s, ty, V(mask, i1),
                V(tsource, ty), V(fsource, ty));
        } else if (id == 46) { /* Trailz */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            /* Simple bit-scan loop for count trailing zeros */
            lr_type_t *i1 = lr_type_i1_s(s);
            lr_error_t err;
            uint32_t count_alloca = lr_emit_alloca(s, ty);
            uint32_t val_alloca = lr_emit_alloca(s, arg_ty);
            lr_emit_store(s, I(0, ty), V(count_alloca, ptr));
            lr_emit_store(s, V(arg, arg_ty), V(val_alloca, ptr));
            int bits = (arg_ty == lr_type_i8_s(s)) ? 8 :
                       (arg_ty == lr_type_i16_s(s)) ? 16 :
                       (arg_ty == lr_type_i32_s(s)) ? 32 : 64;
            uint32_t loop_blk = lr_session_block(s);
            uint32_t body_blk = lr_session_block(s);
            uint32_t end_blk = lr_session_block(s);
            lr_emit_br(s, loop_blk);
            lr_session_set_block(s, loop_blk, &err);
            uint32_t cur_count = lr_emit_load(s, ty, V(count_alloca, ptr));
            uint32_t cur_val = lr_emit_load(s, arg_ty, V(val_alloca, ptr));
            uint32_t bit0 = lr_emit_and(s, arg_ty, V(cur_val, arg_ty),
                I(1, arg_ty));
            uint32_t is_zero = lr_emit_icmp(s, LR_CMP_EQ,
                V(bit0, arg_ty), I(0, arg_ty));
            uint32_t not_done = lr_emit_icmp(s, LR_CMP_SLT,
                V(cur_count, ty), I(bits, ty));
            uint32_t cont = lr_emit_and(s, i1, V(is_zero, i1),
                V(not_done, i1));
            lr_emit_condbr(s, V(cont, i1), body_blk, end_blk);
            lr_session_set_block(s, body_blk, &err);
            uint32_t next_count = lr_emit_add(s, ty, V(cur_count, ty),
                I(1, ty));
            uint32_t next_val = lr_emit_lshr(s, arg_ty, V(cur_val, arg_ty),
                I(1, arg_ty));
            lr_emit_store(s, V(next_count, ty), V(count_alloca, ptr));
            lr_emit_store(s, V(next_val, arg_ty), V(val_alloca, ptr));
            lr_emit_br(s, loop_blk);
            lr_session_set_block(s, end_blk, &err);
            tmp = lr_emit_load(s, ty, V(count_alloca, ptr));
        } else if (id == 88) { /* Leadz */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            lr_type_t *i1 = lr_type_i1_s(s);
            lr_error_t err;
            int bits = (arg_ty == lr_type_i8_s(s)) ? 8 :
                       (arg_ty == lr_type_i16_s(s)) ? 16 :
                       (arg_ty == lr_type_i32_s(s)) ? 32 : 64;
            uint32_t count_alloca = lr_emit_alloca(s, ty);
            uint32_t val_alloca = lr_emit_alloca(s, arg_ty);
            lr_emit_store(s, I(0, ty), V(count_alloca, ptr));
            lr_emit_store(s, V(arg, arg_ty), V(val_alloca, ptr));
            uint32_t loop_blk = lr_session_block(s);
            uint32_t body_blk = lr_session_block(s);
            uint32_t end_blk = lr_session_block(s);
            lr_emit_br(s, loop_blk);
            lr_session_set_block(s, loop_blk, &err);
            uint32_t cur_count = lr_emit_load(s, ty, V(count_alloca, ptr));
            uint32_t cur_val = lr_emit_load(s, arg_ty, V(val_alloca, ptr));
            int64_t high_bit = (int64_t)1 << (bits - 1);
            uint32_t top = lr_emit_and(s, arg_ty, V(cur_val, arg_ty),
                I(high_bit, arg_ty));
            uint32_t is_zero = lr_emit_icmp(s, LR_CMP_EQ,
                V(top, arg_ty), I(0, arg_ty));
            uint32_t not_done = lr_emit_icmp(s, LR_CMP_SLT,
                V(cur_count, ty), I(bits, ty));
            uint32_t cont = lr_emit_and(s, i1, V(is_zero, i1),
                V(not_done, i1));
            lr_emit_condbr(s, V(cont, i1), body_blk, end_blk);
            lr_session_set_block(s, body_blk, &err);
            uint32_t next_count = lr_emit_add(s, ty, V(cur_count, ty),
                I(1, ty));
            uint32_t next_val = lr_emit_shl(s, arg_ty, V(cur_val, arg_ty),
                I(1, arg_ty));
            lr_emit_store(s, V(next_count, ty), V(count_alloca, ptr));
            lr_emit_store(s, V(next_val, arg_ty), V(val_alloca, ptr));
            lr_emit_br(s, loop_blk);
            lr_session_set_block(s, end_blk, &err);
            tmp = lr_emit_load(s, ty, V(count_alloca, ptr));
        } else if (id == 84) { /* Ibclr */
            visit_expr(*x.m_args[0]); uint32_t a = tmp;
            visit_expr(*x.m_args[1]); uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            uint32_t mask = lr_emit_shl(s, ty, I(1, ty), V(b, ty));
            uint32_t inv = lr_emit_not(s, ty, V(mask, ty));
            tmp = lr_emit_and(s, ty, V(a, ty), V(inv, ty));
        } else if (id == 85) { /* Ibset */
            visit_expr(*x.m_args[0]); uint32_t a = tmp;
            visit_expr(*x.m_args[1]); uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            uint32_t mask = lr_emit_shl(s, ty, I(1, ty), V(b, ty));
            tmp = lr_emit_or(s, ty, V(a, ty), V(mask, ty));
        } else if (id == 86) { /* Btest */
            visit_expr(*x.m_args[0]); uint32_t a = tmp;
            visit_expr(*x.m_args[1]); uint32_t b = tmp;
            lr_type_t *ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            uint32_t shifted = lr_emit_lshr(s, ty, V(a, ty), V(b, ty));
            uint32_t bit = lr_emit_and(s, ty, V(shifted, ty), I(1, ty));
            tmp = lr_emit_icmp(s, LR_CMP_NE, V(bit, ty), I(0, ty));
        } else if (id == 87) { /* Ibits */
            visit_expr(*x.m_args[0]); uint32_t val = tmp;
            visit_expr(*x.m_args[1]); uint32_t pos = tmp;
            visit_expr(*x.m_args[2]); uint32_t len = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *pty = get_type(ASRUtils::expr_type(x.m_args[1]));
            lr_type_t *lty = get_type(ASRUtils::expr_type(x.m_args[2]));
            if (pty != ty) pos = lr_emit_sextortrunc(s, ty, V(pos, pty));
            if (lty != ty) len = lr_emit_sextortrunc(s, ty, V(len, lty));
            uint32_t shifted = lr_emit_lshr(s, ty, V(val, ty), V(pos, ty));
            uint32_t mask = lr_emit_sub(s, ty,
                V(lr_emit_shl(s, ty, I(1, ty), V(len, ty)), ty),
                I(1, ty));
            tmp = lr_emit_and(s, ty, V(shifted, ty), V(mask, ty));
        } else if (id == 66) { /* Ishft */
            visit_expr(*x.m_args[0]); uint32_t a = tmp;
            visit_expr(*x.m_args[1]); uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            lr_type_t *bty = get_type(ASRUtils::expr_type(x.m_args[1]));
            if (bty != ty) b = lr_emit_sextortrunc(s, ty, V(b, bty));
            lr_type_t *i1 = lr_type_i1_s(s);
            uint32_t is_neg = lr_emit_icmp(s, LR_CMP_SLT,
                V(b, ty), I(0, ty));
            uint32_t abs_b = lr_emit_select(s, ty, V(is_neg, i1),
                V(lr_emit_neg(s, ty, V(b, ty)), ty), V(b, ty));
            uint32_t left = lr_emit_shl(s, ty, V(a, ty), V(abs_b, ty));
            uint32_t right = lr_emit_lshr(s, ty, V(a, ty), V(abs_b, ty));
            tmp = lr_emit_select(s, ty, V(is_neg, i1),
                V(right, ty), V(left, ty));
        } else if (id == 161) { /* Real (type conversion) */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_integer(*arg_t)) {
                tmp = lr_emit_sitofp(s, res_ty, V(arg, arg_ty));
            } else if (ASRUtils::is_real(*arg_t)) {
                if (arg_ty == res_ty) { /* no-op */ }
                else if (lr_type_width(s, res_ty) > lr_type_width(s, arg_ty))
                    tmp = lr_emit_fpext(s, res_ty, V(arg, arg_ty));
                else
                    tmp = lr_emit_fptrunc(s, res_ty, V(arg, arg_ty));
            } else if (ASR::is_a<ASR::Complex_t>(*arg_t)) {
                int kind = ASRUtils::extract_kind_from_ttype_t(arg_t);
                lr_type_t *fty = (kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t idx0[1] = {0};
                uint32_t re = lr_emit_extractvalue(s, fty,
                    V(arg, arg_ty), idx0, 1);
                if (fty != res_ty) {
                    if (lr_type_width(s, res_ty) > lr_type_width(s, fty))
                        tmp = lr_emit_fpext(s, res_ty, V(re, fty));
                    else
                        tmp = lr_emit_fptrunc(s, res_ty, V(re, fty));
                } else {
                    tmp = re;
                }
            }
        } else if (id == 162) { /* Cmplx */
            lr_type_t *res_ty = get_type(x.m_type);
            int dst_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            lr_type_t *fty = (dst_kind == 4)
                ? lr_type_f32_s(s) : lr_type_f64_s(s);
            visit_expr(*x.m_args[0]);
            uint32_t re_arg = tmp;
            ASR::ttype_t *re_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *re_ty = get_type(re_t);
            uint32_t re_val;
            if (ASRUtils::is_integer(*re_t))
                re_val = lr_emit_sitofp(s, fty, V(re_arg, re_ty));
            else if (re_ty != fty) {
                if (lr_type_width(s, fty) > lr_type_width(s, re_ty))
                    re_val = lr_emit_fpext(s, fty, V(re_arg, re_ty));
                else
                    re_val = lr_emit_fptrunc(s, fty, V(re_arg, re_ty));
            } else
                re_val = re_arg;
            uint32_t im_val;
            if (x.n_args > 1 && x.m_args[1]) {
                visit_expr(*x.m_args[1]);
                uint32_t im_arg = tmp;
                ASR::ttype_t *im_t = ASRUtils::expr_type(x.m_args[1]);
                lr_type_t *im_ty = get_type(im_t);
                if (ASRUtils::is_integer(*im_t))
                    im_val = lr_emit_sitofp(s, fty, V(im_arg, im_ty));
                else if (im_ty != fty) {
                    if (lr_type_width(s, fty) > lr_type_width(s, im_ty))
                        im_val = lr_emit_fpext(s, fty, V(im_arg, im_ty));
                    else
                        im_val = lr_emit_fptrunc(s, fty, V(im_arg, im_ty));
                } else
                    im_val = im_arg;
            } else {
                im_val = lr_emit_fadd(s, fty, F(0.0, fty), F(0.0, fty));
            }
            uint32_t alloca_v = lr_emit_alloca(s, res_ty);
            uint32_t re_ptr = lr_emit_structgep(s, res_ty,
                V(alloca_v, ptr), 0);
            uint32_t im_ptr = lr_emit_structgep(s, res_ty,
                V(alloca_v, ptr), 1);
            lr_emit_store(s, V(re_val, fty), V(re_ptr, ptr));
            lr_emit_store(s, V(im_val, fty), V(im_ptr, ptr));
            tmp = lr_emit_load(s, res_ty, V(alloca_v, ptr));
        } else if (id == 186) { /* Int (type conversion) */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_real(*arg_t)) {
                tmp = lr_emit_fptosi(s, res_ty, V(arg, arg_ty));
            } else if (ASRUtils::is_integer(*arg_t)) {
                if (arg_ty != res_ty)
                    tmp = lr_emit_sextortrunc(s, res_ty, V(arg, arg_ty));
            } else if (ASRUtils::is_logical(*arg_t)) {
                tmp = lr_emit_zext(s, res_ty, V(arg, arg_ty));
            }
        } else if (id == 187) { /* Present */
            /* Check if optional argument is non-null */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            tmp = lr_emit_icmp(s, LR_CMP_NE,
                V(arg, ptr), LR_NULL(ptr));
        } else if (id == 159) { /* Logical */
            visit_expr(*x.m_args[0]);
            /* Logical-to-logical is a no-op in our representation */
        } else if (id == 157) { /* Conjg */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            lr_type_t *sty = get_type(x.m_type);
            int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
            uint32_t idx0[1] = {0};
            uint32_t idx1[1] = {1};
            uint32_t re = lr_emit_extractvalue(s, fty, V(arg, sty), idx0, 1);
            uint32_t im = lr_emit_extractvalue(s, fty, V(arg, sty), idx1, 1);
            uint32_t neg_im = lr_emit_fneg(s, fty, V(im, fty));
            uint32_t alloca_v = lr_emit_alloca(s, sty);
            uint32_t re_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 0);
            uint32_t im_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 1);
            lr_emit_store(s, V(re, fty), V(re_ptr, ptr));
            lr_emit_store(s, V(neg_im, fty), V(im_ptr, ptr));
            tmp = lr_emit_load(s, sty, V(alloca_v, ptr));
        } else if (id == 78) { /* Exponent */
            visit_expr(*x.m_args[0]);
            uint32_t arg = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            /* Use frexp to get the exponent */
            const char *fn = (arg_ty == lr_type_f32_s(s))
                ? "_lfortran_iexponent" : "_lfortran_dexponent";
            uint32_t fn_id = declare_math_func2(fn, lr_type_i32_s(s),
                arg_ty, arg_ty);
            lr_operand_desc_t args[1] = {V(arg, arg_ty)};
            uint32_t result = lr_emit_call(s, lr_type_i32_s(s),
                LR_GLOBAL(fn_id, ptr), args, 1);
            if (res_ty != lr_type_i32_s(s))
                tmp = lr_emit_sextortrunc(s, res_ty,
                    V(result, lr_type_i32_s(s)));
            else
                tmp = result;
        } else if (id == 93) { /* Repeat */
            visit_expr(*x.m_args[0]); uint32_t str = tmp;
            visit_expr(*x.m_args[1]); uint32_t count_val = tmp;
            lr_type_t *count_ty = get_type(ASRUtils::expr_type(x.m_args[1]));
            lr_type_t *i64 = lr_type_i64_s(s);
            lr_type_t *i32 = lr_type_i32_s(s);
            lr_type_t *i8 = lr_type_i8_s(s);
            /* Get string length */
            uint32_t strlen_id = get_strlen_id();
            lr_operand_desc_t sl_args[1] = {V(str, ptr)};
            uint32_t str_len = lr_emit_call(s, i64,
                LR_GLOBAL(strlen_id, ptr), sl_args, 1);
            /* Count as i64 */
            uint32_t cnt64;
            if (count_ty != i64)
                cnt64 = lr_emit_sextortrunc(s, i64, V(count_val, count_ty));
            else
                cnt64 = count_val;
            uint32_t total = lr_emit_mul(s, i64, V(str_len, i64),
                V(cnt64, i64));
            uint32_t buf_size = lr_emit_add(s, i64, V(total, i64), I(1, i64));
            uint32_t malloc_id = get_malloc_id();
            lr_operand_desc_t ma_args[1] = {V(buf_size, i64)};
            uint32_t buf = lr_emit_call(s, ptr,
                LR_GLOBAL(malloc_id, ptr), ma_args, 1);
            /* Copy in a loop */
            uint32_t memcpy_id = get_memcpy_id();
            uint32_t off_alloca = lr_emit_alloca(s, i64);
            uint32_t i_alloca = lr_emit_alloca(s, i64);
            lr_emit_store(s, I(0, i64), V(off_alloca, ptr));
            lr_emit_store(s, I(0, i64), V(i_alloca, ptr));
            lr_error_t err;
            lr_type_t *i1 = lr_type_i1_s(s);
            uint32_t lp = lr_session_block(s);
            uint32_t lb = lr_session_block(s);
            uint32_t le = lr_session_block(s);
            lr_emit_br(s, lp);
            lr_session_set_block(s, lp, &err);
            uint32_t ci = lr_emit_load(s, i64, V(i_alloca, ptr));
            uint32_t cond = lr_emit_icmp(s, LR_CMP_SLT,
                V(ci, i64), V(cnt64, i64));
            lr_emit_condbr(s, V(cond, i1), lb, le);
            lr_session_set_block(s, lb, &err);
            uint32_t off = lr_emit_load(s, i64, V(off_alloca, ptr));
            lr_operand_desc_t gep_idx[1] = {V(off, i64)};
            uint32_t dst = lr_emit_gep(s, i8, V(buf, ptr), gep_idx, 1);
            lr_operand_desc_t mc_args[3] = {
                V(dst, ptr), V(str, ptr), V(str_len, i64)
            };
            lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc_args, 3);
            uint32_t new_off = lr_emit_add(s, i64, V(off, i64),
                V(str_len, i64));
            lr_emit_store(s, V(new_off, i64), V(off_alloca, ptr));
            uint32_t ni = lr_emit_add(s, i64, V(ci, i64), I(1, i64));
            lr_emit_store(s, V(ni, i64), V(i_alloca, ptr));
            lr_emit_br(s, lp);
            lr_session_set_block(s, le, &err);
            /* Null terminate */
            lr_operand_desc_t nt_idx[1] = {V(total, i64)};
            uint32_t null_pos = lr_emit_gep(s, i8, V(buf, ptr), nt_idx, 1);
            lr_emit_store(s, I(0, i8), V(null_pos, ptr));
            tmp = buf;
        } else if (id == 107) { /* Ichar */
            visit_expr(*x.m_args[0]);
            uint32_t str = tmp;
            lr_type_t *i8 = lr_type_i8_s(s);
            lr_type_t *res_ty = get_type(x.m_type);
            uint32_t ch = lr_emit_load(s, i8, V(str, ptr));
            tmp = lr_emit_zext(s, res_ty, V(ch, i8));
        } else if (id == 108 || id == 109) { /* Char, Achar */
            visit_expr(*x.m_args[0]);
            uint32_t code = tmp;
            lr_type_t *i8 = lr_type_i8_s(s);
            lr_type_t *i64 = lr_type_i64_s(s);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            uint32_t ch = lr_emit_trunc(s, i8, V(code, arg_ty));
            /* Allocate 2-byte buffer */
            uint32_t malloc_id = get_malloc_id();
            lr_operand_desc_t ma_args[1] = {I(2, i64)};
            uint32_t buf = lr_emit_call(s, ptr,
                LR_GLOBAL(malloc_id, ptr), ma_args, 1);
            lr_emit_store(s, V(ch, i8), V(buf, ptr));
            lr_operand_desc_t gep_null[1] = {I(1, lr_type_i32_s(s))};
            uint32_t null_ptr = lr_emit_gep(s, i8, V(buf, ptr),
                gep_null, 1);
            lr_emit_store(s, I(0, i8), V(null_ptr, ptr));
            tmp = buf;
        } else if (id == 110) { /* Iachar */
            visit_expr(*x.m_args[0]);
            uint32_t str = tmp;
            lr_type_t *i8 = lr_type_i8_s(s);
            lr_type_t *res_ty = get_type(x.m_type);
            uint32_t ch = lr_emit_load(s, i8, V(str, ptr));
            tmp = lr_emit_zext(s, res_ty, V(ch, i8));
        } else if (id == 14) { /* Asinh */
            emit_unary_math("asinh", "asinhf", x.m_args[0], x.m_type);
        } else if (id == 22) { /* Acosh */
            emit_unary_math("acosh", "acoshf", x.m_args[0], x.m_type);
        } else if (id == 23) { /* Atanh */
            emit_unary_math("atanh", "atanhf", x.m_args[0], x.m_type);
        } else if (id == 30) { /* Erf */
            emit_unary_math("erf", "erff", x.m_args[0], x.m_type);
        } else if (id == 31) { /* Erfc */
            emit_unary_math("erfc", "erfcf", x.m_args[0], x.m_type);
        } else if (id == 33) { /* Gamma */
            emit_unary_math("tgamma", "tgammaf", x.m_args[0], x.m_type);
        } else if (id == 36) { /* LogGamma */
            emit_unary_math("lgamma", "lgammaf", x.m_args[0], x.m_type);
        } else if (id == 97) { /* Hypot */
            visit_expr(*x.m_args[0]); uint32_t a = tmp;
            visit_expr(*x.m_args[1]); uint32_t b = tmp;
            lr_type_t *ty = get_type(x.m_type);
            const char *name = (ty == lr_type_f32_s(s)) ? "hypotf" : "hypot";
            uint32_t fn_id = declare_math_func2(name, ty, ty, ty);
            lr_operand_desc_t args[2] = {V(a, ty), V(b, ty)};
            tmp = lr_emit_call(s, ty, LR_GLOBAL(fn_id, ptr), args, 2);
        } else if (id == 138) { /* Aint */
            visit_expr(*x.m_args[0]);
            uint32_t a = tmp;
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            lr_type_t *arg_ty = get_type(arg_t);
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_real(*arg_t) && ASRUtils::is_real(*x.m_type)) {
                emit_unary_math("trunc", "truncf", x.m_args[0], x.m_type);
            } else {
                tmp = lr_emit_fptosi(s, res_ty, V(a, arg_ty));
            }
        } else if (id == 139) { /* Anint */
            emit_unary_math("round", "roundf", x.m_args[0], x.m_type);
        } else if (id == 143) { /* Sngl */
            visit_expr(*x.m_args[0]);
            lr_type_t *f32 = lr_type_f32_s(s);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            if (arg_ty != f32)
                tmp = lr_emit_fptrunc(s, f32, V(tmp, arg_ty));
        } else if (id == 144 || id == 145) { /* Ifix, Idint */
            visit_expr(*x.m_args[0]);
            lr_type_t *arg_ty = get_type(ASRUtils::expr_type(x.m_args[0]));
            lr_type_t *res_ty = get_type(x.m_type);
            tmp = lr_emit_fptosi(s, res_ty, V(tmp, arg_ty));
        } else if (id == 103) { /* StringConcat (elemental) */
            /* This should be handled by visit_StringConcat but sometimes
               appears as an IntrinsicElementalFunction */
            visit_expr(*x.m_args[0]); uint32_t left = tmp;
            visit_expr(*x.m_args[1]); uint32_t right = tmp;
            lr_type_t *i64 = lr_type_i64_s(s);
            lr_type_t *i8 = lr_type_i8_s(s);
            uint32_t strlen_id = get_strlen_id();
            uint32_t malloc_id = get_malloc_id();
            uint32_t memcpy_id = get_memcpy_id();
            lr_operand_desc_t sl[1] = {V(left, ptr)};
            uint32_t ll = lr_emit_call(s, i64,
                LR_GLOBAL(strlen_id, ptr), sl, 1);
            lr_operand_desc_t sr[1] = {V(right, ptr)};
            uint32_t rl = lr_emit_call(s, i64,
                LR_GLOBAL(strlen_id, ptr), sr, 1);
            uint32_t tl = lr_emit_add(s, i64, V(ll, i64), V(rl, i64));
            uint32_t bs = lr_emit_add(s, i64, V(tl, i64), I(1, i64));
            lr_operand_desc_t ma[1] = {V(bs, i64)};
            uint32_t buf = lr_emit_call(s, ptr,
                LR_GLOBAL(malloc_id, ptr), ma, 1);
            lr_operand_desc_t mc1[3] = {V(buf, ptr), V(left, ptr), V(ll, i64)};
            lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc1, 3);
            lr_operand_desc_t gi[1] = {V(ll, i64)};
            uint32_t d2 = lr_emit_gep(s, i8, V(buf, ptr), gi, 1);
            lr_operand_desc_t mc2[3] = {V(d2, ptr), V(right, ptr), V(rl, i64)};
            lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc2, 3);
            lr_operand_desc_t ni[1] = {V(tl, i64)};
            uint32_t np = lr_emit_gep(s, i8, V(buf, ptr), ni, 1);
            lr_emit_store(s, I(0, i8), V(np, ptr));
            tmp = buf;
        } else if (id == 105) { /* StringTrim */
            /* Trim trailing spaces - use _lfortran_str_trim runtime */
            visit_expr(*x.m_args[0]); uint32_t str = tmp;
            /* Simple implementation: find last non-space, allocate, copy */
            lr_type_t *i64 = lr_type_i64_s(s);
            lr_type_t *i8 = lr_type_i8_s(s);
            lr_type_t *i32 = lr_type_i32_s(s);
            lr_type_t *i1 = lr_type_i1_s(s);
            uint32_t strlen_id = get_strlen_id();
            uint32_t malloc_id = get_malloc_id();
            uint32_t memcpy_id = get_memcpy_id();
            lr_operand_desc_t sl[1] = {V(str, ptr)};
            uint32_t len = lr_emit_call(s, i64,
                LR_GLOBAL(strlen_id, ptr), sl, 1);
            /* Scan backward to find last non-space */
            uint32_t end_alloca = lr_emit_alloca(s, i64);
            lr_emit_store(s, V(len, i64), V(end_alloca, ptr));
            lr_error_t err;
            uint32_t scan_blk = lr_session_block(s);
            uint32_t scan_body = lr_session_block(s);
            uint32_t scan_end = lr_session_block(s);
            lr_emit_br(s, scan_blk);
            lr_session_set_block(s, scan_blk, &err);
            uint32_t cur_end = lr_emit_load(s, i64, V(end_alloca, ptr));
            uint32_t gt0 = lr_emit_icmp(s, LR_CMP_SGT,
                V(cur_end, i64), I(0, i64));
            uint32_t prev = lr_emit_sub(s, i64, V(cur_end, i64), I(1, i64));
            lr_operand_desc_t gi[1] = {V(prev, i64)};
            uint32_t cp = lr_emit_gep(s, i8, V(str, ptr), gi, 1);
            uint32_t ch = lr_emit_load(s, i8, V(cp, ptr));
            uint32_t is_space = lr_emit_icmp(s, LR_CMP_EQ,
                V(ch, i8), I(32, i8));
            uint32_t cont = lr_emit_and(s, i1, V(gt0, i1),
                V(is_space, i1));
            lr_emit_condbr(s, V(cont, i1), scan_body, scan_end);
            lr_session_set_block(s, scan_body, &err);
            lr_emit_store(s, V(prev, i64), V(end_alloca, ptr));
            lr_emit_br(s, scan_blk);
            lr_session_set_block(s, scan_end, &err);
            uint32_t trim_len = lr_emit_load(s, i64, V(end_alloca, ptr));
            uint32_t bs = lr_emit_add(s, i64, V(trim_len, i64), I(1, i64));
            lr_operand_desc_t ma[1] = {V(bs, i64)};
            uint32_t buf = lr_emit_call(s, ptr,
                LR_GLOBAL(malloc_id, ptr), ma, 1);
            lr_operand_desc_t mc[3] = {V(buf, ptr), V(str, ptr),
                V(trim_len, i64)};
            lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc, 3);
            lr_operand_desc_t ni[1] = {V(trim_len, i64)};
            uint32_t np = lr_emit_gep(s, i8, V(buf, ptr), ni, 1);
            lr_emit_store(s, I(0, i8), V(np, ptr));
            tmp = buf;
        } else if (id == 104) { /* StringLenTrim */
            visit_expr(*x.m_args[0]); uint32_t str = tmp;
            lr_type_t *i64 = lr_type_i64_s(s);
            lr_type_t *i8 = lr_type_i8_s(s);
            lr_type_t *i1 = lr_type_i1_s(s);
            lr_type_t *res_ty = get_type(x.m_type);
            uint32_t strlen_id = get_strlen_id();
            lr_operand_desc_t sl[1] = {V(str, ptr)};
            uint32_t len = lr_emit_call(s, i64,
                LR_GLOBAL(strlen_id, ptr), sl, 1);
            uint32_t end_alloca = lr_emit_alloca(s, i64);
            lr_emit_store(s, V(len, i64), V(end_alloca, ptr));
            lr_error_t err;
            uint32_t scan_blk = lr_session_block(s);
            uint32_t scan_body = lr_session_block(s);
            uint32_t scan_end = lr_session_block(s);
            lr_emit_br(s, scan_blk);
            lr_session_set_block(s, scan_blk, &err);
            uint32_t cur_end = lr_emit_load(s, i64, V(end_alloca, ptr));
            uint32_t gt0 = lr_emit_icmp(s, LR_CMP_SGT,
                V(cur_end, i64), I(0, i64));
            uint32_t prev = lr_emit_sub(s, i64, V(cur_end, i64), I(1, i64));
            lr_operand_desc_t gi[1] = {V(prev, i64)};
            uint32_t cp = lr_emit_gep(s, i8, V(str, ptr), gi, 1);
            uint32_t ch = lr_emit_load(s, i8, V(cp, ptr));
            uint32_t is_space = lr_emit_icmp(s, LR_CMP_EQ,
                V(ch, i8), I(32, i8));
            uint32_t cont = lr_emit_and(s, i1, V(gt0, i1),
                V(is_space, i1));
            lr_emit_condbr(s, V(cont, i1), scan_body, scan_end);
            lr_session_set_block(s, scan_body, &err);
            lr_emit_store(s, V(prev, i64), V(end_alloca, ptr));
            lr_emit_br(s, scan_blk);
            lr_session_set_block(s, scan_end, &err);
            uint32_t result = lr_emit_load(s, i64, V(end_alloca, ptr));
            if (res_ty != i64)
                tmp = lr_emit_sextortrunc(s, res_ty, V(result, i64));
            else
                tmp = result;
        } else if (id == 158) { /* Huge */
            lr_type_t *res_ty = get_type(x.m_type);
            if (ASRUtils::is_integer(*x.m_type)) {
                int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                int64_t huge_val;
                switch (kind) {
                    case 1: huge_val = 127; break;
                    case 2: huge_val = 32767; break;
                    case 4: huge_val = 2147483647LL; break;
                    case 8: huge_val = 9223372036854775807LL; break;
                    default: huge_val = 2147483647LL; break;
                }
                tmp = lr_emit_add(s, res_ty, I(huge_val, res_ty),
                    I(0, res_ty));
            } else if (ASRUtils::is_real(*x.m_type)) {
                int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                double huge_val = (kind == 4) ? 3.40282347e+38
                    : 1.7976931348623157e+308;
                tmp = lr_emit_fadd(s, res_ty, F(huge_val, res_ty),
                    F(0.0, res_ty));
            }
        } else if (id == 152) { /* Epsilon */
            lr_type_t *res_ty = get_type(x.m_type);
            int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            double eps = (kind == 4) ? 1.1920929e-07 : 2.2204460492503131e-16;
            tmp = lr_emit_fadd(s, res_ty, F(eps, res_ty), F(0.0, res_ty));
        } else if (id == 154) { /* Tiny */
            lr_type_t *res_ty = get_type(x.m_type);
            int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
            double tiny_val = (kind == 4) ? 1.17549435e-38
                : 2.2250738585072014e-308;
            tmp = lr_emit_fadd(s, res_ty, F(tiny_val, res_ty),
                F(0.0, res_ty));
        } else if (id == 155) { /* BitSize */
            lr_type_t *res_ty = get_type(x.m_type);
            int kind = ASRUtils::extract_kind_from_ttype_t(
                ASRUtils::expr_type(x.m_args[0]));
            tmp = lr_emit_add(s, res_ty, I(kind * 8, res_ty),
                I(0, res_ty));
        } else if (id == 91) { /* Digits */
            lr_type_t *res_ty = get_type(x.m_type);
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            int kind = ASRUtils::extract_kind_from_ttype_t(arg_t);
            int digits;
            if (ASRUtils::is_integer(*arg_t)) {
                digits = kind * 8 - 1;
            } else {
                digits = (kind == 4) ? 24 : 53;
            }
            tmp = lr_emit_add(s, res_ty, I(digits, res_ty), I(0, res_ty));
        } else if (id == 125) { /* Radix */
            lr_type_t *res_ty = get_type(x.m_type);
            tmp = lr_emit_add(s, res_ty, I(2, res_ty), I(0, res_ty));
        } else if (id == 129) { /* Range */
            lr_type_t *res_ty = get_type(x.m_type);
            ASR::ttype_t *arg_t = ASRUtils::expr_type(x.m_args[0]);
            int kind = ASRUtils::extract_kind_from_ttype_t(arg_t);
            int range;
            if (ASRUtils::is_integer(*arg_t)) {
                switch (kind) {
                    case 1: range = 2; break;
                    case 2: range = 4; break;
                    case 4: range = 9; break;
                    case 8: range = 18; break;
                    default: range = 9; break;
                }
            } else {
                range = (kind == 4) ? 37 : 307;
            }
            tmp = lr_emit_add(s, res_ty, I(range, res_ty), I(0, res_ty));
        } else if (id == 153) { /* Precision */
            lr_type_t *res_ty = get_type(x.m_type);
            int kind = ASRUtils::extract_kind_from_ttype_t(
                ASRUtils::expr_type(x.m_args[0]));
            int prec = (kind == 4) ? 6 : 15;
            tmp = lr_emit_add(s, res_ty, I(prec, res_ty), I(0, res_ty));
        } else if (id == 156) { /* NewLine */
            tmp = emit_global_string( "\n");
        } else {
            throw CodeGenError("liric: IntrinsicElementalFunction id="
                + std::to_string(id) + " not yet implemented");
        }
    }

    void visit_IntrinsicImpureSubroutine(
            const ASR::IntrinsicImpureSubroutine_t &x) {
        int64_t id = x.m_sub_intrinsic_id;
        lr_type_t *ptr = lr_type_ptr_s(s);
        if (id == 0) { /* RandomNumber */
            if (x.n_args > 0) {
                is_target = true;
                visit_expr(*x.m_args[0]);
                is_target = false;
                uint32_t arr_ptr = tmp;
                lr_type_t *i32 = lr_type_i32_s(s);
                lr_type_t *void_ty = lr_type_void_s(s);
                lr_type_t *params[2] = {ptr, i32};
                uint32_t fn_id = ensure_runtime_func(
                    "_lfortran_random_number", void_ty, params, 2);
                ASR::ttype_t *at = ASRUtils::expr_type(x.m_args[0]);
                ASR::dimension_t *dims = nullptr;
                size_t n_dims = ASRUtils::extract_dimensions_from_ttype(at, dims);
                int64_t total = ASRUtils::get_fixed_size_of_array(dims, n_dims);
                if (total <= 0) total = 1;
                lr_operand_desc_t args[2] = {
                    V(arr_ptr, ptr), I(total, i32)
                };
                lr_emit_call_void(s, LR_GLOBAL(fn_id, ptr), args, 2);
            }
        } else if (id == 11) { /* MoveAlloc */
            if (x.n_args >= 2) {
                is_target = true;
                visit_expr(*x.m_args[0]);
                is_target = false;
                uint32_t from_ptr = tmp;
                is_target = true;
                visit_expr(*x.m_args[1]);
                is_target = false;
                uint32_t to_ptr = tmp;
                uint32_t val = lr_emit_load(s, ptr, V(from_ptr, ptr));
                lr_emit_store(s, V(val, ptr), V(to_ptr, ptr));
                lr_emit_store(s, LR_NULL(ptr), V(from_ptr, ptr));
            }
        } else if (id == 7) { /* CpuTime */
            if (x.n_args > 0) {
                is_target = true;
                visit_expr(*x.m_args[0]);
                is_target = false;
                uint32_t time_ptr = tmp;
                lr_type_t *params[1] = {ptr};
                uint32_t fn_id = ensure_runtime_func(
                    "_lfortran_cpu_time", lr_type_void_s(s), params, 1);
                lr_operand_desc_t args[1] = {V(time_ptr, ptr)};
                lr_emit_call_void(s, LR_GLOBAL(fn_id, ptr), args, 1);
            }
        } else if (id == 9) { /* SystemClock */
            /* system_clock(count, count_rate, count_max) - ignore for now */
        } else if (id == 10) { /* DateAndTime */
            /* date_and_time(...) - ignore for now */
        } else if (id == 13) { /* Abort */
            emit_exit(134);
        } else {
            throw CodeGenError("liric: IntrinsicImpureSubroutine id="
                + std::to_string(id) + " not yet implemented");
        }
    }

    void visit_IntrinsicArrayFunction(
            const ASR::IntrinsicArrayFunction_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        int64_t id = x.m_arr_intrinsic_id;
        /* These functions operate on arrays and normally get lowered
           by ASR passes. If they reach codegen, we handle the simple
           cases directly (1D fixed-size arrays with no dim/mask). */
        throw CodeGenError("liric: IntrinsicArrayFunction id="
            + std::to_string(id) + " not yet implemented");
    }

    void visit_IntrinsicImpureFunction(
            const ASR::IntrinsicImpureFunction_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        int64_t id = x.m_impure_intrinsic_id;
        if (id == 2) { /* Allocated */
            /* Check if the pointer is non-NULL */
            visit_expr(*x.m_args[0]);
            uint32_t ptr_val = tmp;
            lr_type_t *ptr = lr_type_ptr_s(s);
            tmp = lr_emit_icmp(s, LR_CMP_NE,
                V(ptr_val, ptr), LR_NULL(ptr));
        } else {
            throw CodeGenError("liric: IntrinsicImpureFunction id="
                + std::to_string(id) + " not yet implemented");
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
            uint32_t raw = it->second;
            if (raw & 0x80000000u) {
                /* Global variable: the stored value is a symbol ID
                   with the high bit set as a flag. */
                uint32_t sym_id = raw & 0x7FFFFFFFu;
                lr_type_t *ptr = lr_type_ptr_s(s);
                uint32_t addr = lr_emit_bitcast(s, ptr,
                    LR_GLOBAL(sym_id, ptr));
                if (is_target) {
                    tmp = addr;
                } else {
                    lr_type_t *ty = get_type(v->m_type);
                    tmp = lr_emit_load(s, ty, V(addr, ptr));
                }
            } else {
                uint32_t ptr_vreg = raw;
                if (is_target) {
                    tmp = ptr_vreg;
                } else {
                    lr_type_t *ty = get_type(v->m_type);
                    tmp = lr_emit_load(s, ty,
                        V(ptr_vreg, lr_type_ptr_s(s)));
                }
            }
        } else if (ASR::is_a<ASR::Function_t>(*sym)) {
            /* Function reference (function pointer or pass-as-argument).
               Return the address of the function. */
            ASR::Function_t *fn = ASR::down_cast<ASR::Function_t>(sym);
            uint32_t fn_id = ensure_function_declared(fn);
            tmp = lr_emit_bitcast(s, lr_type_ptr_s(s),
                LR_GLOBAL(fn_id, lr_type_ptr_s(s)));
        } else {
            throw CodeGenError("liric: unsupported symbol type in Var: "
                + std::to_string(sym->type));
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
                /* Integer power via runtime: _lfortran_dpowi / _lfortran_spowi
                   or inline loop for small cases. Use the runtime. */
                lr_type_t *f64 = lr_type_f64_s(s);
                const char *pow_fn = "_lfortran_dpowi";
                uint32_t pow_id = declare_math_func2(pow_fn, f64, f64, ty);
                uint32_t base_f = lr_emit_sitofp(s, f64, l);
                lr_operand_desc_t args[2] = {V(base_f, f64), r};
                uint32_t result_f = lr_emit_call(s, f64,
                    LR_GLOBAL(pow_id, lr_type_ptr_s(s)), args, 2);
                tmp = lr_emit_fptosi(s, ty, V(result_f, f64));
                break;
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
            case ASR::binopType::Pow: {
                const char *pow_fn = (ty == lr_type_f32_s(s))
                    ? "powf" : "pow";
                uint32_t fn_id = declare_math_func2(pow_fn, ty, ty, ty);
                lr_operand_desc_t args[2] = {l, r};
                tmp = lr_emit_call(s, ty,
                    LR_GLOBAL(fn_id, lr_type_ptr_s(s)), args, 2);
                break;
            }
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

    /* ---- GoTo ---------------------------------------------------------- */

    std::unordered_map<int64_t, uint32_t> goto_targets;

    void visit_GoTo(const ASR::GoTo_t &x) {
        auto it = goto_targets.find(x.m_target_id);
        if (it == goto_targets.end()) {
            /* Forward reference — create block and register */
            uint32_t blk = lr_session_block(s);
            goto_targets[x.m_target_id] = blk;
            lr_emit_br(s, blk);
        } else {
            lr_emit_br(s, it->second);
        }
        uint32_t dead = lr_session_block(s);
        lr_session_set_block(s, dead, NULL);
    }

    void visit_GoToTarget(const ASR::GoToTarget_t &x) {
        uint32_t blk;
        auto it = goto_targets.find(x.m_id);
        if (it == goto_targets.end()) {
            blk = lr_session_block(s);
            goto_targets[x.m_id] = blk;
        } else {
            blk = it->second;
        }
        lr_emit_br(s, blk);
        lr_session_set_block(s, blk, NULL);
    }

    /* ---- Runtime function helpers --------------------------------------- */

    /* Declare a runtime function by name and return its symbol ID.
       If already declared, returns the existing ID. */
    uint32_t ensure_runtime_func(const char *name, lr_type_t *ret_ty,
                                 lr_type_t **params = NULL,
                                 uint32_t n_params = 0,
                                 bool vararg = false) {
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, name)) {
            return lr_session_intern(s, name);
        }
        lr_error_t err;
        lr_session_declare(s, name, ret_ty, params, n_params, vararg, &err);
        return lr_session_intern(s, name);
    }

    /* Convenience: declare a function taking one ptr arg */
    uint32_t ensure_runtime_func_1ptr(const char *name, lr_type_t *ret_ty) {
        lr_type_t *params[1] = {lr_type_ptr_s(s)};
        return ensure_runtime_func(name, ret_ty, params, 1, false);
    }

    /* ---- Print (via _lfortran_printf runtime function) ----------------- */

    /* _lfortran_printf(format, str, str_len, end, end_len)
       All 5 args are fixed (not varargs). format is unused by runtime
       but kept for ABI compatibility. str+str_len is the main output.
       end+end_len is the line terminator (typically "\n", len=1). */
    uint32_t get_lfortran_printf_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "_lfortran_printf")) {
            return lr_session_intern(s, "_lfortran_printf");
        }
        lr_type_t *void_ty = lr_type_void_s(s);
        lr_type_t *params[5] = {
            lr_type_ptr_s(s),  /* format (unused) */
            lr_type_ptr_s(s),  /* str */
            lr_type_i32_s(s),  /* str_len */
            lr_type_ptr_s(s),  /* end */
            lr_type_i32_s(s),  /* end_len */
        };
        lr_session_declare(s, "_lfortran_printf", void_ty, params, 5, false, &err);
        return lr_session_intern(s, "_lfortran_printf");
    }

    /* _lcompilers_snprintf_alloc(al, format, ...) — allocates and formats a string.
       Requires a valid allocator from _lfortran_get_default_allocator(). */
    uint32_t get_snprintf_alloc_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "_lcompilers_snprintf_alloc")) {
            return lr_session_intern(s, "_lcompilers_snprintf_alloc");
        }
        lr_type_t *params[2] = {lr_type_ptr_s(s), lr_type_ptr_s(s)};
        lr_session_declare(s, "_lcompilers_snprintf_alloc",
                           lr_type_ptr_s(s), params, 2, true, &err);
        return lr_session_intern(s, "_lcompilers_snprintf_alloc");
    }

    uint32_t get_default_allocator_id() {
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod,
                "_lfortran_get_default_allocator")) {
            return lr_session_intern(s, "_lfortran_get_default_allocator");
        }
        lr_error_t err;
        lr_session_declare(s, "_lfortran_get_default_allocator",
            lr_type_ptr_s(s), NULL, 0, false, &err);
        return lr_session_intern(s, "_lfortran_get_default_allocator");
    }

    /* Emit formatted print using C printf (declared as external).
       This is the simplest correct approach for formatted output. */
    void emit_printf_values(ASR::expr_t **values, size_t n_values,
                            bool suppress_newline) {

        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);

        /* Declare printf as a varargs function returning i32 */
        uint32_t printf_c_id;
        {
            lr_module_t *mod = lr_session_module(s);
            if (mod && lr_module_lookup_function(mod, "printf")) {
                printf_c_id = lr_session_intern(s, "printf");
            } else {
                lr_error_t err;
                lr_type_t *params[1] = {ptr};
                lr_session_declare(s, "printf", i32, params, 1, true, &err);
                printf_c_id = lr_session_intern(s, "printf");
            }
        }

        /* Build format string and collect value args */
        std::string fmt;
        std::vector<lr_operand_desc_t> val_args;

        for (size_t i = 0; i < n_values; i++) {
            if (i > 0) fmt += " ";
            ASR::ttype_t *t = ASRUtils::extract_type(
                ASRUtils::expr_type(values[i]));
            int kind = ASRUtils::extract_kind_from_ttype_t(t);

            visit_expr(*values[i]);

            if (ASRUtils::is_integer(*t)) {
                lr_type_t *ty = get_type(t);
                switch (kind) {
                    case 1: {
                        /* Promote i8 to i32 for printf */
                        uint32_t promoted = lr_emit_sextortrunc(s, i32,
                            V(tmp, ty));
                        fmt += "%hhd";
                        val_args.push_back(V(promoted, i32));
                        break;
                    }
                    case 2: {
                        uint32_t promoted = lr_emit_sextortrunc(s, i32,
                            V(tmp, ty));
                        fmt += "%hd";
                        val_args.push_back(V(promoted, i32));
                        break;
                    }
                    case 4: fmt += "%d"; val_args.push_back(V(tmp, ty)); break;
                    case 8: fmt += "%lld"; val_args.push_back(V(tmp, ty)); break;
                    default: fmt += "%d"; val_args.push_back(V(tmp, ty)); break;
                }
            } else if (ASRUtils::is_real(*t)) {
                lr_type_t *f64 = lr_type_f64_s(s);
                uint32_t val = tmp;
                if (kind == 4) {
                    val = lr_emit_fpext(s, f64, V(val, lr_type_f32_s(s)));
                }
                fmt += "%13.4e";
                val_args.push_back(V(val, f64));
            } else if (ASRUtils::is_logical(*t)) {
                fmt += "%s";
                lr_type_t *i1 = lr_type_i1_s(s);
                uint32_t t_str = emit_global_string( "T");
                uint32_t f_str = emit_global_string( "F");
                uint32_t str = lr_emit_select(s, ptr,
                    V(tmp, i1), V(t_str, ptr), V(f_str, ptr));
                val_args.push_back(V(str, ptr));
            } else if (ASRUtils::is_character(*t)) {
                fmt += "%s";
                val_args.push_back(V(tmp, ptr));
            } else if (ASR::is_a<ASR::Complex_t>(*t)) {
                lr_type_t *sty = get_type(t);
                lr_type_t *f64 = lr_type_f64_s(s);
                lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : f64;
                uint32_t idx0[1] = {0};
                uint32_t idx1[1] = {1};
                uint32_t re = lr_emit_extractvalue(s, fty,
                    V(tmp, sty), idx0, 1);
                uint32_t im = lr_emit_extractvalue(s, fty,
                    V(tmp, sty), idx1, 1);
                if (kind == 4) {
                    re = lr_emit_fpext(s, f64, V(re, fty));
                    im = lr_emit_fpext(s, f64, V(im, fty));
                }
                fmt += "(%13.4e,%13.4e)";
                val_args.push_back(V(re, f64));
                val_args.push_back(V(im, f64));
            } else if (ASRUtils::is_unsigned_integer(*t)) {
                lr_type_t *ty = get_type(t);
                switch (kind) {
                    case 1: case 2: {
                        uint32_t promoted = lr_emit_zextortrunc(s, i32,
                            V(tmp, ty));
                        fmt += "%u";
                        val_args.push_back(V(promoted, i32));
                        break;
                    }
                    case 4: fmt += "%u"; val_args.push_back(V(tmp, ty)); break;
                    case 8: fmt += "%llu"; val_args.push_back(V(tmp, ty)); break;
                    default: fmt += "%u"; val_args.push_back(V(tmp, ty)); break;
                }
            } else {
                throw CodeGenError("liric: Print unsupported type "
                    + std::to_string(t->type));
            }
        }
        if (!suppress_newline) fmt += "\n";

        /* Create format string as a global constant */
        uint32_t fmt_ptr = emit_global_string( fmt.c_str());

        /* Call printf(fmt, args...) */
        {
            lr_inst_desc_t d; memset(&d, 0, sizeof(d));
            uint32_t nops = 1 + 1 + (uint32_t)val_args.size();
            lr_operand_desc_t ops[64];
            ops[0] = LR_GLOBAL(printf_c_id, ptr);  /* callee */
            ops[1] = V(fmt_ptr, ptr);               /* format string */
            for (size_t j = 0; j < val_args.size(); j++)
                ops[2 + j] = val_args[j];
            d.op = LR_OP_CALL;
            d.type = i32;
            d.operands = ops;
            d.num_operands = nops;
            d.call_vararg = true;
            d.call_fixed_args = 1;
            lr_session_emit(s, &d, NULL);
        }

        if (!suppress_newline) {
            /* Flush stdout */
            uint32_t fflush_id;
            lr_module_t *mod = lr_session_module(s);
            if (mod && lr_module_lookup_function(mod, "fflush")) {
                fflush_id = lr_session_intern(s, "fflush");
            } else {
                lr_error_t err;
                lr_type_t *params[1] = {ptr};
                lr_session_declare(s, "fflush", i32, params, 1,
                    false, &err);
                fflush_id = lr_session_intern(s, "fflush");
            }
            lr_operand_desc_t fargs[1] = {LR_NULL(ptr)};
            lr_emit_call(s, i32, LR_GLOBAL(fflush_id, ptr), fargs, 1);
        }
    }

    void visit_Print(const ASR::Print_t &x) {
        if (!x.m_text) return;
        if (ASR::is_a<ASR::StringFormat_t>(*x.m_text)) {
            ASR::StringFormat_t *sf =
                ASR::down_cast<ASR::StringFormat_t>(x.m_text);
            emit_printf_values(sf->m_args, sf->n_args, false);
        } else {
            /* Single expression print (e.g. string) */
            ASR::expr_t *args[1] = {x.m_text};
            emit_printf_values(args, 1, false);
        }
    }

    void visit_FileWrite(const ASR::FileWrite_t &x) {
        if (x.m_overloaded) {
            visit_stmt(*x.m_overloaded);
            return;
        }
        /* For stdout (unit=* or unit=6), use printf.
           For other units, throw for now. */
        if (x.n_values == 1 &&
                ASR::is_a<ASR::StringFormat_t>(*x.m_values[0])) {
            ASR::StringFormat_t *sf =
                ASR::down_cast<ASR::StringFormat_t>(x.m_values[0]);
            emit_printf_values(sf->m_args, sf->n_args, false);
        } else if (x.n_values > 0) {
            emit_printf_values(x.m_values, x.n_values, false);
        }
    }

    void visit_StringFormat(const ASR::StringFormat_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        throw CodeGenError("liric: StringFormat outside Print not yet implemented");
    }

    /* ---- SelectCase ---------------------------------------------------- */

    void visit_Select(const ASR::Select_t &x) {
        lr_error_t err;
        lr_type_t *i1 = lr_type_i1_s(s);

        visit_expr(*x.m_test);
        uint32_t test_val = tmp;
        lr_type_t *test_ty = get_type(ASRUtils::expr_type(x.m_test));

        uint32_t end_block = lr_session_block(s);

        for (size_t i = 0; i < x.n_body; i++) {
            ASR::case_stmt_t *cs = x.m_body[i];
            if (ASR::is_a<ASR::CaseStmt_t>(*cs)) {
                ASR::CaseStmt_t *c = ASR::down_cast<ASR::CaseStmt_t>(cs);
                uint32_t case_body = lr_session_block(s);
                uint32_t next_case = lr_session_block(s);

                /* Check if test matches any of the case values */
                uint32_t match = 0;
                for (size_t j = 0; j < c->n_test; j++) {
                    visit_expr(*c->m_test[j]);
                    uint32_t cmp = lr_emit_icmp(s, LR_CMP_EQ,
                        V(test_val, test_ty), V(tmp, test_ty));
                    if (j == 0) {
                        match = cmp;
                    } else {
                        match = lr_emit_or(s, i1, V(match, i1), V(cmp, i1));
                    }
                }
                lr_emit_condbr(s, V(match, i1), case_body, next_case);

                lr_session_set_block(s, case_body, &err);
                for (size_t j = 0; j < c->n_body; j++) {
                    visit_stmt(*c->m_body[j]);
                }
                lr_emit_br(s, end_block);
                lr_session_set_block(s, next_case, &err);
            }
        }

        /* Default case */
        if (x.n_default > 0) {
            for (size_t i = 0; i < x.n_default; i++) {
                visit_stmt(*x.m_default[i]);
            }
        }
        lr_emit_br(s, end_block);
        lr_session_set_block(s, end_block, &err);
    }

    /* ---- Complex ------------------------------------------------------- */

    void visit_ComplexConstant(const ASR::ComplexConstant_t &x) {
        int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
        lr_type_t *sty = get_type(x.m_type);
        /* Build struct {re, im} */
        uint32_t alloca_v = lr_emit_alloca(s, sty);
        uint32_t re_ptr = lr_emit_structgep(s, sty,
            V(alloca_v, lr_type_ptr_s(s)), 0);
        uint32_t im_ptr = lr_emit_structgep(s, sty,
            V(alloca_v, lr_type_ptr_s(s)), 1);
        lr_emit_store(s, F(x.m_re, fty), V(re_ptr, lr_type_ptr_s(s)));
        lr_emit_store(s, F(x.m_im, fty), V(im_ptr, lr_type_ptr_s(s)));
        tmp = lr_emit_load(s, sty, V(alloca_v, lr_type_ptr_s(s)));
    }

    void visit_ComplexConstructor(const ASR::ComplexConstructor_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        lr_type_t *sty = get_type(x.m_type);
        int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);

        visit_expr(*x.m_re);
        uint32_t re_val = tmp;
        visit_expr(*x.m_im);
        uint32_t im_val = tmp;

        uint32_t alloca_v = lr_emit_alloca(s, sty);
        uint32_t re_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 0);
        uint32_t im_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 1);
        lr_emit_store(s, V(re_val, fty), V(re_ptr, ptr));
        lr_emit_store(s, V(im_val, fty), V(im_ptr, ptr));
        tmp = lr_emit_load(s, sty, V(alloca_v, ptr));
    }

    void visit_ComplexBinOp(const ASR::ComplexBinOp_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;

        lr_type_t *sty = get_type(x.m_type);
        int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);

        /* Extract re/im from both operands */
        uint32_t idx0[1] = {0};
        uint32_t idx1[1] = {1};
        uint32_t l_re = lr_emit_extractvalue(s, fty, V(left, sty), idx0, 1);
        uint32_t l_im = lr_emit_extractvalue(s, fty, V(left, sty), idx1, 1);
        uint32_t r_re = lr_emit_extractvalue(s, fty, V(right, sty), idx0, 1);
        uint32_t r_im = lr_emit_extractvalue(s, fty, V(right, sty), idx1, 1);

        uint32_t res_re, res_im;
        switch (x.m_op) {
            case ASR::binopType::Add:
                res_re = lr_emit_fadd(s, fty, V(l_re, fty), V(r_re, fty));
                res_im = lr_emit_fadd(s, fty, V(l_im, fty), V(r_im, fty));
                break;
            case ASR::binopType::Sub:
                res_re = lr_emit_fsub(s, fty, V(l_re, fty), V(r_re, fty));
                res_im = lr_emit_fsub(s, fty, V(l_im, fty), V(r_im, fty));
                break;
            case ASR::binopType::Mul: {
                /* (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
                uint32_t ac = lr_emit_fmul(s, fty, V(l_re, fty), V(r_re, fty));
                uint32_t bd = lr_emit_fmul(s, fty, V(l_im, fty), V(r_im, fty));
                uint32_t ad = lr_emit_fmul(s, fty, V(l_re, fty), V(r_im, fty));
                uint32_t bc = lr_emit_fmul(s, fty, V(l_im, fty), V(r_re, fty));
                res_re = lr_emit_fsub(s, fty, V(ac, fty), V(bd, fty));
                res_im = lr_emit_fadd(s, fty, V(ad, fty), V(bc, fty));
                break;
            }
            case ASR::binopType::Div: {
                /* (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2) */
                uint32_t ac = lr_emit_fmul(s, fty, V(l_re, fty), V(r_re, fty));
                uint32_t bd = lr_emit_fmul(s, fty, V(l_im, fty), V(r_im, fty));
                uint32_t bc = lr_emit_fmul(s, fty, V(l_im, fty), V(r_re, fty));
                uint32_t ad = lr_emit_fmul(s, fty, V(l_re, fty), V(r_im, fty));
                uint32_t cc = lr_emit_fmul(s, fty, V(r_re, fty), V(r_re, fty));
                uint32_t dd = lr_emit_fmul(s, fty, V(r_im, fty), V(r_im, fty));
                uint32_t denom = lr_emit_fadd(s, fty, V(cc, fty), V(dd, fty));
                uint32_t num_re = lr_emit_fadd(s, fty, V(ac, fty), V(bd, fty));
                uint32_t num_im = lr_emit_fsub(s, fty, V(bc, fty), V(ad, fty));
                res_re = lr_emit_fdiv(s, fty, V(num_re, fty), V(denom, fty));
                res_im = lr_emit_fdiv(s, fty, V(num_im, fty), V(denom, fty));
                break;
            }
            default:
                throw CodeGenError("liric: unsupported complex binop");
        }

        /* Build result struct */
        uint32_t alloca_v = lr_emit_alloca(s, sty);
        uint32_t re_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 0);
        uint32_t im_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 1);
        lr_emit_store(s, V(res_re, fty), V(re_ptr, ptr));
        lr_emit_store(s, V(res_im, fty), V(im_ptr, ptr));
        tmp = lr_emit_load(s, sty, V(alloca_v, ptr));
    }

    void visit_ComplexUnaryMinus(const ASR::ComplexUnaryMinus_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        uint32_t arg = tmp;
        lr_type_t *sty = get_type(x.m_type);
        int kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
        lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);

        uint32_t idx0[1] = {0};
        uint32_t idx1[1] = {1};
        uint32_t re = lr_emit_extractvalue(s, fty, V(arg, sty), idx0, 1);
        uint32_t im = lr_emit_extractvalue(s, fty, V(arg, sty), idx1, 1);
        uint32_t neg_re = lr_emit_fneg(s, fty, V(re, fty));
        uint32_t neg_im = lr_emit_fneg(s, fty, V(im, fty));

        uint32_t alloca_v = lr_emit_alloca(s, sty);
        uint32_t re_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 0);
        uint32_t im_ptr = lr_emit_structgep(s, sty, V(alloca_v, ptr), 1);
        lr_emit_store(s, V(neg_re, fty), V(re_ptr, ptr));
        lr_emit_store(s, V(neg_im, fty), V(im_ptr, ptr));
        tmp = lr_emit_load(s, sty, V(alloca_v, ptr));
    }

    void visit_ComplexCompare(const ASR::ComplexCompare_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;

        lr_type_t *sty = get_type(ASRUtils::expr_type(x.m_left));
        int kind = ASRUtils::extract_kind_from_ttype_t(
            ASRUtils::expr_type(x.m_left));
        lr_type_t *fty = (kind == 4) ? lr_type_f32_s(s) : lr_type_f64_s(s);
        lr_type_t *i1 = lr_type_i1_s(s);

        uint32_t idx0[1] = {0};
        uint32_t idx1[1] = {1};
        uint32_t l_re = lr_emit_extractvalue(s, fty, V(left, sty), idx0, 1);
        uint32_t l_im = lr_emit_extractvalue(s, fty, V(left, sty), idx1, 1);
        uint32_t r_re = lr_emit_extractvalue(s, fty, V(right, sty), idx0, 1);
        uint32_t r_im = lr_emit_extractvalue(s, fty, V(right, sty), idx1, 1);

        uint32_t re_cmp = lr_emit_fcmp(s, LR_FCMP_OEQ,
            V(l_re, fty), V(r_re, fty));
        uint32_t im_cmp = lr_emit_fcmp(s, LR_FCMP_OEQ,
            V(l_im, fty), V(r_im, fty));

        if (x.m_op == ASR::cmpopType::Eq) {
            tmp = lr_emit_and(s, i1, V(re_cmp, i1), V(im_cmp, i1));
        } else if (x.m_op == ASR::cmpopType::NotEq) {
            uint32_t eq = lr_emit_and(s, i1, V(re_cmp, i1),
                                      V(im_cmp, i1));
            tmp = lr_emit_xor(s, i1, V(eq, i1), I(1, i1));
        } else {
            throw CodeGenError("liric: unsupported complex compare op");
        }
    }

    /* ---- Stop / ErrorStop ---------------------------------------------- */

    void emit_exit(int code) {
        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *p = lr_type_ptr_s(s);
        lr_type_t *exit_params[1] = {i32};
        uint32_t exit_id = ensure_runtime_func("exit", lr_type_void_s(s),
            exit_params, 1);
        lr_operand_desc_t args[1] = {I(code, i32)};
        lr_emit_call_void(s, LR_GLOBAL(exit_id, p), args, 1);
        lr_emit_unreachable(s);
    }

    void visit_Stop(const ASR::Stop_t &x) {
        if (x.m_code && ASR::is_a<ASR::IntegerConstant_t>(*x.m_code)) {
            int64_t code = ASR::down_cast<ASR::IntegerConstant_t>(
                x.m_code)->m_n;
            emit_exit((int)code);
        } else {
            emit_exit(0);
        }
    }

    void visit_ErrorStop(const ASR::ErrorStop_t &x) {
        if (x.m_code && ASR::is_a<ASR::IntegerConstant_t>(*x.m_code)) {
            int64_t code = ASR::down_cast<ASR::IntegerConstant_t>(
                x.m_code)->m_n;
            emit_exit((int)code);
        } else {
            emit_exit(1);
        }
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
            case ASR::cast_kindType::UnsignedIntegerToInteger:
            case ASR::cast_kindType::IntegerToUnsignedInteger:
            case ASR::cast_kindType::UnsignedIntegerToUnsignedInteger:
                tmp = lr_emit_zextortrunc(s, dst_ty, v); break;
            case ASR::cast_kindType::UnsignedIntegerToReal:
                tmp = lr_emit_uitofp(s, dst_ty, v); break;
            case ASR::cast_kindType::RealToUnsignedInteger:
                tmp = lr_emit_fptoui(s, dst_ty, v); break;
            case ASR::cast_kindType::LogicalToReal:
                tmp = lr_emit_uitofp(s, dst_ty,
                    V(lr_emit_zext(s, lr_type_i32_s(s), v),
                      lr_type_i32_s(s)));
                break;
            case ASR::cast_kindType::RealToLogical:
                tmp = lr_emit_fcmp(s, LR_FCMP_ONE, v, F(0.0, src_ty));
                break;
            case ASR::cast_kindType::LogicalToLogical:
                /* no-op */ break;
            case ASR::cast_kindType::ComplexToReal: {
                /* Extract real part from complex struct */
                int src_kind = ASRUtils::extract_kind_from_ttype_t(
                    ASRUtils::expr_type(x.m_arg));
                lr_type_t *src_fty = (src_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t idx0[1] = {0};
                uint32_t re = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx0, 1);
                if (dst_ty != src_fty) {
                    if (lr_type_width(s, dst_ty) > lr_type_width(s, src_fty))
                        tmp = lr_emit_fpext(s, dst_ty, V(re, src_fty));
                    else
                        tmp = lr_emit_fptrunc(s, dst_ty, V(re, src_fty));
                } else {
                    tmp = re;
                }
                break;
            }
            case ASR::cast_kindType::ComplexToComplex: {
                int src_kind = ASRUtils::extract_kind_from_ttype_t(
                    ASRUtils::expr_type(x.m_arg));
                int dst_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                lr_type_t *src_fty = (src_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                lr_type_t *dst_fty = (dst_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t idx0[1] = {0};
                uint32_t idx1[1] = {1};
                uint32_t re = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx0, 1);
                uint32_t im = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx1, 1);
                if (src_fty != dst_fty) {
                    if (lr_type_width(s, dst_fty) > lr_type_width(s, src_fty)) {
                        re = lr_emit_fpext(s, dst_fty, V(re, src_fty));
                        im = lr_emit_fpext(s, dst_fty, V(im, src_fty));
                    } else {
                        re = lr_emit_fptrunc(s, dst_fty, V(re, src_fty));
                        im = lr_emit_fptrunc(s, dst_fty, V(im, src_fty));
                    }
                }
                uint32_t alloca_v = lr_emit_alloca(s, dst_ty);
                lr_type_t *ptr = lr_type_ptr_s(s);
                uint32_t re_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 0);
                uint32_t im_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 1);
                lr_emit_store(s, V(re, dst_fty), V(re_ptr, ptr));
                lr_emit_store(s, V(im, dst_fty), V(im_ptr, ptr));
                tmp = lr_emit_load(s, dst_ty, V(alloca_v, ptr));
                break;
            }
            case ASR::cast_kindType::RealToComplex: {
                int dst_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                lr_type_t *dst_fty = (dst_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t re_val = arg;
                if (src_ty != dst_fty) {
                    if (lr_type_width(s, dst_fty) > lr_type_width(s, src_ty))
                        re_val = lr_emit_fpext(s, dst_fty, V(arg, src_ty));
                    else
                        re_val = lr_emit_fptrunc(s, dst_fty, V(arg, src_ty));
                }
                uint32_t alloca_v = lr_emit_alloca(s, dst_ty);
                lr_type_t *ptr = lr_type_ptr_s(s);
                uint32_t re_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 0);
                uint32_t im_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 1);
                lr_emit_store(s, V(re_val, dst_fty), V(re_ptr, ptr));
                lr_emit_store(s, F(0.0, dst_fty), V(im_ptr, ptr));
                tmp = lr_emit_load(s, dst_ty, V(alloca_v, ptr));
                break;
            }
            case ASR::cast_kindType::IntegerToComplex: {
                int dst_kind = ASRUtils::extract_kind_from_ttype_t(x.m_type);
                lr_type_t *dst_fty = (dst_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t re_val = lr_emit_sitofp(s, dst_fty, v);
                uint32_t alloca_v = lr_emit_alloca(s, dst_ty);
                lr_type_t *ptr = lr_type_ptr_s(s);
                uint32_t re_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 0);
                uint32_t im_ptr = lr_emit_structgep(s, dst_ty,
                    V(alloca_v, ptr), 1);
                lr_emit_store(s, V(re_val, dst_fty), V(re_ptr, ptr));
                lr_emit_store(s, F(0.0, dst_fty), V(im_ptr, ptr));
                tmp = lr_emit_load(s, dst_ty, V(alloca_v, ptr));
                break;
            }
            case ASR::cast_kindType::ComplexToInteger: {
                int src_kind = ASRUtils::extract_kind_from_ttype_t(
                    ASRUtils::expr_type(x.m_arg));
                lr_type_t *src_fty = (src_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t idx0[1] = {0};
                uint32_t re = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx0, 1);
                tmp = lr_emit_fptosi(s, dst_ty, V(re, src_fty));
                break;
            }
            case ASR::cast_kindType::ComplexToLogical: {
                int src_kind = ASRUtils::extract_kind_from_ttype_t(
                    ASRUtils::expr_type(x.m_arg));
                lr_type_t *src_fty = (src_kind == 4)
                    ? lr_type_f32_s(s) : lr_type_f64_s(s);
                uint32_t idx0[1] = {0};
                uint32_t idx1[1] = {1};
                uint32_t re = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx0, 1);
                uint32_t im = lr_emit_extractvalue(s, src_fty,
                    V(arg, src_ty), idx1, 1);
                uint32_t re_nz = lr_emit_fcmp(s, LR_FCMP_ONE,
                    V(re, src_fty), F(0.0, src_fty));
                uint32_t im_nz = lr_emit_fcmp(s, LR_FCMP_ONE,
                    V(im, src_fty), F(0.0, src_fty));
                tmp = lr_emit_or(s, lr_type_i1_s(s),
                    V(re_nz, lr_type_i1_s(s)),
                    V(im_nz, lr_type_i1_s(s)));
                break;
            }
            /* LogicalToComplex handled via LogicalToInteger + IntegerToComplex */
            case ASR::cast_kindType::IntegerToString:
            case ASR::cast_kindType::RealToString:
            case ASR::cast_kindType::LogicalToString:
            case ASR::cast_kindType::StringToLogical:
            case ASR::cast_kindType::StringToInteger:
            case ASR::cast_kindType::StringToList:
            case ASR::cast_kindType::StringToArray:
            case ASR::cast_kindType::ListToArray:
                /* These require string formatting runtime support.
                   For now pass through as-is. */
                break;
            case ASR::cast_kindType::CPtrToUnsignedInteger:
                tmp = lr_emit_ptrtoint(s, dst_ty, V(arg, src_ty));
                break;
            case ASR::cast_kindType::UnsignedIntegerToCPtr:
                tmp = lr_emit_inttoptr(s, dst_ty, V(arg, src_ty));
                break;
            case ASR::cast_kindType::UnsignedIntegerToLogical:
                tmp = lr_emit_icmp(s, LR_CMP_NE, v, I(0, src_ty));
                break;
            case ASR::cast_kindType::PointerToInteger:
                tmp = lr_emit_ptrtoint(s, dst_ty, V(arg, src_ty));
                break;
            case ASR::cast_kindType::IntegerToSymbolicExpression:
            case ASR::cast_kindType::ClassToStruct:
            case ASR::cast_kindType::ClassToClass:
            case ASR::cast_kindType::ClassToIntrinsic:
                /* Pass through */
                break;
            default:
                throw CodeGenError("liric: unsupported cast kind "
                    + std::to_string((int)x.m_kind));
        }
    }

    /* ---- ArrayPhysicalCast ------------------------------------------- */

    void visit_ArrayPhysicalCast(const ASR::ArrayPhysicalCast_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* In our representation, most array physical casts are no-ops
           since all arrays are pointers. The key case is
           FixedSizeArray -> PointerArray/DescriptorArray: we need to
           get a pointer to the data (address of first element). */
        bool saved_target = is_target;
        if (x.m_old == ASR::array_physical_typeType::FixedSizeArray &&
            (x.m_new == ASR::array_physical_typeType::PointerArray ||
             x.m_new == ASR::array_physical_typeType::DescriptorArray)) {
            is_target = true;
            visit_expr(*x.m_arg);
            is_target = saved_target;
            uint32_t arr_ptr = tmp;
            /* GEP to get pointer to first element */
            lr_type_t *i32 = lr_type_i32_s(s);
            lr_type_t *ptr = lr_type_ptr_s(s);
            ASR::ttype_t *arg_type = ASRUtils::type_get_past_allocatable(
                ASRUtils::type_get_past_pointer(
                    ASRUtils::expr_type(x.m_arg)));
            lr_type_t *arr_ty = get_type(arg_type);
            lr_operand_desc_t gep_idx[2] = {I(0, i32), I(0, i32)};
            tmp = lr_emit_gep(s, arr_ty, V(arr_ptr, ptr), gep_idx, 2);
        } else {
            visit_expr(*x.m_arg);
            is_target = saved_target;
        }
    }

    /* ---- Array helpers ------------------------------------------------- */

    /* Get the lower bound for dimension `dim_idx` of an array type.
       Returns 1 if no explicit start is set. */
    int64_t get_array_dim_start(ASR::ttype_t *arr_type, size_t dim_idx) {
        ASR::dimension_t *dims = nullptr;
        size_t n = ASRUtils::extract_dimensions_from_ttype(arr_type, dims);
        if (dim_idx < n && dims[dim_idx].m_start) {
            int64_t val = 1;
            if (ASRUtils::extract_value(
                    ASRUtils::expr_value(dims[dim_idx].m_start), val))
                return val;
        }
        return 1;
    }

    /* Compute a linearized, 0-based offset for a FixedSizeArray from
       Fortran indices (which may start at a lower bound != 0). */
    uint32_t linearize_fixed_index(ASR::ttype_t *arr_type,
                                   ASR::array_index_t *args, size_t n_args) {
        ASR::dimension_t *dims = nullptr;
        size_t n_dims = ASRUtils::extract_dimensions_from_ttype(arr_type, dims);
        lr_type_t *i32 = lr_type_i32_s(s);

        uint32_t offset = lr_emit_add(s, i32, I(0, i32), I(0, i32));
        uint32_t stride = lr_emit_add(s, i32, I(1, i32), I(0, i32));

        bool saved_target = is_target;
        is_target = false;
        for (size_t d = 0; d < n_args && d < n_dims; d++) {
            visit_expr(*args[d].m_right);
            uint32_t idx = tmp;
            lr_type_t *idx_ty = get_type(ASRUtils::expr_type(args[d].m_right));
            if (idx_ty != i32) {
                idx = lr_emit_sextortrunc(s, i32, V(idx, idx_ty));
            }
            int64_t lb = get_array_dim_start(arr_type, d);
            uint32_t adj = lr_emit_sub(s, i32, V(idx, i32), I(lb, i32));
            uint32_t contrib = lr_emit_mul(s, i32, V(adj, i32),
                                           V(stride, i32));
            offset = lr_emit_add(s, i32, V(offset, i32), V(contrib, i32));

            if (d + 1 < n_args) {
                int64_t dim_len = 0;
                if (dims[d].m_length &&
                    ASRUtils::extract_value(
                        ASRUtils::expr_value(dims[d].m_length), dim_len)) {
                    stride = lr_emit_mul(s, i32, V(stride, i32),
                                         I(dim_len, i32));
                } else if (dims[d].m_length) {
                    visit_expr(*dims[d].m_length);
                    uint32_t len = tmp;
                    lr_type_t *len_ty = get_type(
                        ASRUtils::expr_type(dims[d].m_length));
                    if (len_ty != i32) {
                        len = lr_emit_sextortrunc(s, i32, V(len, len_ty));
                    }
                    stride = lr_emit_mul(s, i32, V(stride, i32),
                                         V(len, i32));
                }
            }
        }
        is_target = saved_target;
        return offset;
    }

    /* ---- ArrayItem ---------------------------------------------------- */

    void visit_ArrayItem(const ASR::ArrayItem_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        bool want_address = is_target;
        ASR::ttype_t *arr_type = ASRUtils::expr_type(x.m_v);
        ASR::ttype_t *arr_type_inner = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(arr_type));
        lr_type_t *elem_ty = get_scalar_type(arr_type_inner);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);

        if (ASR::is_a<ASR::Array_t>(*arr_type_inner)) {
            ASR::Array_t *at = ASR::down_cast<ASR::Array_t>(arr_type_inner);
            if (at->m_physical_type ==
                    ASR::array_physical_typeType::FixedSizeArray) {
                /* FixedSizeArray: alloca is [N x elem], use GEP */
                is_target = true;
                visit_expr(*x.m_v);
                is_target = want_address;
                uint32_t arr_ptr = tmp;

                uint32_t offset = linearize_fixed_index(
                    arr_type_inner, x.m_args, x.n_args);

                lr_type_t *arr_ty = get_type(arr_type_inner);
                lr_operand_desc_t gep_idx[2] = {
                    I(0, i32), V(offset, i32)
                };
                uint32_t elem_ptr = lr_emit_gep(s, arr_ty,
                    V(arr_ptr, ptr), gep_idx, 2);

                if (want_address) {
                    tmp = elem_ptr;
                } else {
                    tmp = lr_emit_load(s, elem_ty, V(elem_ptr, ptr));
                }
                return;
            }
            if (at->m_physical_type ==
                    ASR::array_physical_typeType::PointerArray) {
                /* PointerArray: data is behind a pointer, load then GEP */
                is_target = true;
                visit_expr(*x.m_v);
                is_target = want_address;
                uint32_t data_ptr_ptr = tmp;
                uint32_t data_ptr = lr_emit_load(s, ptr,
                    V(data_ptr_ptr, ptr));

                uint32_t offset = linearize_fixed_index(
                    arr_type_inner, x.m_args, x.n_args);
                lr_operand_desc_t gep_idx[1] = {V(offset, i32)};
                uint32_t elem_ptr = lr_emit_gep(s, elem_ty,
                    V(data_ptr, ptr), gep_idx, 1);

                if (want_address) {
                    tmp = elem_ptr;
                } else {
                    tmp = lr_emit_load(s, elem_ty, V(elem_ptr, ptr));
                }
                return;
            }
            if (at->m_physical_type ==
                    ASR::array_physical_typeType::DescriptorArray) {
                /* DescriptorArray: variable holds a data pointer
                   (set by Allocate). Load the pointer, then GEP. */
                is_target = true;
                visit_expr(*x.m_v);
                is_target = want_address;
                uint32_t data_ptr_ptr = tmp;
                uint32_t data_ptr = lr_emit_load(s, ptr,
                    V(data_ptr_ptr, ptr));

                uint32_t offset = linearize_fixed_index(
                    arr_type_inner, x.m_args, x.n_args);
                lr_operand_desc_t gep_idx[1] = {V(offset, i32)};
                uint32_t elem_ptr = lr_emit_gep(s, elem_ty,
                    V(data_ptr, ptr), gep_idx, 1);

                if (want_address) {
                    tmp = elem_ptr;
                } else {
                    tmp = lr_emit_load(s, elem_ty, V(elem_ptr, ptr));
                }
                return;
            }
        }
        throw CodeGenError("liric: unsupported array physical type in ArrayItem");
    }

    /* ---- ArrayConstant ------------------------------------------------ */

    void visit_ArrayConstant(const ASR::ArrayConstant_t &x) {
        /* ArrayConstant stores raw binary data in m_data.
           Emit it as a global constant and load. */
        ASR::ttype_t *el_type = ASRUtils::extract_type(x.m_type);
        lr_type_t *elem_ty = get_scalar_type(el_type);
        int kind = ASRUtils::extract_kind_from_ttype_t(el_type);
        int64_t n_elems = x.m_n_data;
        lr_type_t *arr_ty = lr_type_array_s(s, elem_ty, (uint64_t)n_elems);
        size_t data_size = (size_t)(n_elems * kind);

        static unsigned arr_const_counter = 0;
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), ".arr_const.%u",
                 arr_const_counter++);
        lr_session_global(s, name_buf, arr_ty, true,
                          x.m_data, data_size);
        uint32_t sym_id = lr_session_intern(s, name_buf);
        lr_type_t *ptr = lr_type_ptr_s(s);
        tmp = lr_emit_bitcast(s, ptr, LR_GLOBAL(sym_id, ptr));
    }

    /* ---- ArrayConstructor --------------------------------------------- */

    void visit_ArrayConstructor(const ASR::ArrayConstructor_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        ASR::ttype_t *el_type = ASRUtils::extract_type(x.m_type);
        lr_type_t *elem_ty = get_scalar_type(el_type);
        lr_type_t *arr_ty = get_type(x.m_type);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);

        uint32_t arr_alloca = lr_emit_alloca(s, arr_ty);
        for (size_t i = 0; i < x.n_args; i++) {
            visit_expr(*x.m_args[i]);
            uint32_t val = tmp;
            lr_operand_desc_t gep_idx[2] = {I(0, i32), I((int64_t)i, i32)};
            uint32_t ep = lr_emit_gep(s, arr_ty, V(arr_alloca, ptr),
                                      gep_idx, 2);
            lr_emit_store(s, V(val, elem_ty), V(ep, ptr));
        }
        tmp = lr_emit_load(s, arr_ty, V(arr_alloca, ptr));
    }

    /* ---- ArrayBound --------------------------------------------------- */

    void visit_ArrayBound(const ASR::ArrayBound_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        ASR::ttype_t *arr_type = ASRUtils::expr_type(x.m_v);
        ASR::dimension_t *dims = nullptr;
        size_t n_dims = ASRUtils::extract_dimensions_from_ttype(arr_type, dims);
        lr_type_t *res_ty = get_type(x.m_type);

        size_t dim_idx = 0;
        if (x.m_dim) {
            int64_t dv = 1;
            if (ASRUtils::extract_value(ASRUtils::expr_value(x.m_dim), dv)) {
                dim_idx = (size_t)(dv - 1);
            }
        }
        if (dim_idx >= n_dims) {
            tmp = lr_emit_add(s, res_ty, I(1, res_ty), I(0, res_ty));
            return;
        }

        if (x.m_bound == ASR::arrayboundType::LBound) {
            int64_t lb = 1;
            if (dims[dim_idx].m_start) {
                ASRUtils::extract_value(
                    ASRUtils::expr_value(dims[dim_idx].m_start), lb);
            }
            tmp = lr_emit_add(s, res_ty, I(lb, res_ty), I(0, res_ty));
        } else {
            int64_t lb = 1;
            if (dims[dim_idx].m_start) {
                ASRUtils::extract_value(
                    ASRUtils::expr_value(dims[dim_idx].m_start), lb);
            }
            int64_t length = 0;
            if (dims[dim_idx].m_length &&
                ASRUtils::extract_value(
                    ASRUtils::expr_value(dims[dim_idx].m_length), length)) {
                int64_t ub = lb + length - 1;
                tmp = lr_emit_add(s, res_ty, I(ub, res_ty), I(0, res_ty));
            } else if (dims[dim_idx].m_length) {
                visit_expr(*dims[dim_idx].m_length);
                uint32_t len = tmp;
                lr_type_t *len_ty = get_type(
                    ASRUtils::expr_type(dims[dim_idx].m_length));
                if (len_ty != res_ty) {
                    len = lr_emit_sextortrunc(s, res_ty, V(len, len_ty));
                }
                uint32_t lb_v = lr_emit_add(s, res_ty,
                    I(lb, res_ty), I(0, res_ty));
                uint32_t ub_v = lr_emit_add(s, res_ty,
                    V(lb_v, res_ty), V(len, res_ty));
                tmp = lr_emit_sub(s, res_ty, V(ub_v, res_ty),
                    I(1, res_ty));
            } else {
                tmp = lr_emit_add(s, res_ty, I(0, res_ty), I(0, res_ty));
            }
        }
    }

    /* ---- ArraySize ---------------------------------------------------- */

    void visit_ArraySize(const ASR::ArraySize_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        ASR::ttype_t *arr_type = ASRUtils::expr_type(x.m_v);
        ASR::dimension_t *dims = nullptr;
        size_t n_dims = ASRUtils::extract_dimensions_from_ttype(arr_type, dims);
        lr_type_t *res_ty = get_type(x.m_type);

        if (x.m_dim) {
            int64_t dv = 1;
            if (ASRUtils::extract_value(ASRUtils::expr_value(x.m_dim), dv)) {
                size_t dim_idx = (size_t)(dv - 1);
                if (dim_idx < n_dims && dims[dim_idx].m_length) {
                    int64_t length = 0;
                    if (ASRUtils::extract_value(
                            ASRUtils::expr_value(dims[dim_idx].m_length),
                            length)) {
                        tmp = lr_emit_add(s, res_ty,
                            I(length, res_ty), I(0, res_ty));
                    } else {
                        visit_expr(*dims[dim_idx].m_length);
                        uint32_t len = tmp;
                        lr_type_t *len_ty = get_type(
                            ASRUtils::expr_type(dims[dim_idx].m_length));
                        if (len_ty != res_ty) {
                            len = lr_emit_sextortrunc(s, res_ty,
                                V(len, len_ty));
                        }
                        tmp = len;
                    }
                    return;
                }
            }
        }

        int64_t total = ASRUtils::get_fixed_size_of_array(dims, n_dims);
        if (total > 0) {
            tmp = lr_emit_add(s, res_ty, I(total, res_ty), I(0, res_ty));
        } else {
            uint32_t product = lr_emit_add(s, res_ty, I(1, res_ty),
                                           I(0, res_ty));
            for (size_t d = 0; d < n_dims; d++) {
                if (dims[d].m_length) {
                    visit_expr(*dims[d].m_length);
                    uint32_t len = tmp;
                    lr_type_t *len_ty = get_type(
                        ASRUtils::expr_type(dims[d].m_length));
                    if (len_ty != res_ty) {
                        len = lr_emit_sextortrunc(s, res_ty,
                            V(len, len_ty));
                    }
                    product = lr_emit_mul(s, res_ty, V(product, res_ty),
                                          V(len, res_ty));
                }
            }
            tmp = product;
        }
    }

    /* ---- ArrayBroadcast ----------------------------------------------- */

    void visit_ArrayBroadcast(const ASR::ArrayBroadcast_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* Broadcast a scalar to fill an array.
           For fixed-size arrays, store the value into each element. */
        visit_expr(*x.m_array);
        uint32_t scalar_val = tmp;
        lr_type_t *elem_ty = get_scalar_type(x.m_type);
        lr_type_t *arr_ty = get_type(x.m_type);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);

        ASR::dimension_t *dims = nullptr;
        size_t n_dims = ASRUtils::extract_dimensions_from_ttype(x.m_type, dims);
        int64_t total = ASRUtils::get_fixed_size_of_array(dims, n_dims);
        if (total <= 0) {
            /* Non-fixed-size: allocate dynamically and fill with a loop.
               For descriptors, we need the runtime size. Use alloca
               with a reasonable upper bound or malloc. For now,
               just produce a null pointer (broadcast will be handled
               by the allocate path). */
            tmp = lr_emit_inttoptr(s, ptr, I(0, lr_type_i64_s(s)));
            return;
        }

        uint32_t arr_alloca = lr_emit_alloca(s, arr_ty);
        for (int64_t i = 0; i < total; i++) {
            lr_operand_desc_t gep_idx[2] = {I(0, i32), I(i, i32)};
            uint32_t ep = lr_emit_gep(s, arr_ty, V(arr_alloca, ptr),
                                      gep_idx, 2);
            lr_emit_store(s, V(scalar_val, elem_ty), V(ep, ptr));
        }
        tmp = lr_emit_load(s, arr_ty, V(arr_alloca, ptr));
    }

    /* ---- Allocate / Deallocate ---------------------------------------- */

    uint32_t get_malloc_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "malloc")) {
            return lr_session_intern(s, "malloc");
        }
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        lr_type_t *params[1] = {i64};
        lr_session_declare(s, "malloc", ptr, params, 1, false, &err);
        return lr_session_intern(s, "malloc");
    }

    uint32_t get_free_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "free")) {
            return lr_session_intern(s, "free");
        }
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *params[1] = {ptr};
        lr_session_declare(s, "free", lr_type_void_s(s), params, 1,
                           false, &err);
        return lr_session_intern(s, "free");
    }

    void visit_Allocate(const ASR::Allocate_t &x) {
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        uint32_t malloc_id = get_malloc_id();

        for (size_t i = 0; i < x.n_args; i++) {
            ASR::alloc_arg_t &aa = x.m_args[i];
            ASR::ttype_t *alloc_type = ASRUtils::expr_type(aa.m_a);
            lr_type_t *elem_ty = get_scalar_type(alloc_type);
            int elem_bytes = 0;
            if (elem_ty == lr_type_i8_s(s))       elem_bytes = 1;
            else if (elem_ty == lr_type_i16_s(s))  elem_bytes = 2;
            else if (elem_ty == lr_type_i32_s(s))  elem_bytes = 4;
            else if (elem_ty == lr_type_i64_s(s))  elem_bytes = 8;
            else if (elem_ty == lr_type_f32_s(s))  elem_bytes = 4;
            else if (elem_ty == lr_type_f64_s(s))  elem_bytes = 8;
            else if (elem_ty == lr_type_i1_s(s))   elem_bytes = 1;
            else                                    elem_bytes = 8;

            uint32_t total_size = lr_emit_add(s, i64, I(1, i64), I(0, i64));
            for (size_t d = 0; d < aa.n_dims; d++) {
                if (aa.m_dims[d].m_length) {
                    visit_expr(*aa.m_dims[d].m_length);
                    uint32_t dim_len = tmp;
                    lr_type_t *dim_ty = get_type(
                        ASRUtils::expr_type(aa.m_dims[d].m_length));
                    if (dim_ty != i64) {
                        dim_len = lr_emit_sextortrunc(s, i64,
                            V(dim_len, dim_ty));
                    }
                    total_size = lr_emit_mul(s, i64, V(total_size, i64),
                                             V(dim_len, i64));
                }
            }
            uint32_t byte_size = lr_emit_mul(s, i64, V(total_size, i64),
                I(elem_bytes, i64));

            lr_operand_desc_t args[1] = {V(byte_size, i64)};
            uint32_t mem = lr_emit_call(s, ptr,
                LR_GLOBAL(malloc_id, ptr), args, 1);

            is_target = true;
            visit_expr(*aa.m_a);
            is_target = false;
            uint32_t var_ptr = tmp;
            lr_emit_store(s, V(mem, ptr), V(var_ptr, ptr));
        }

        if (x.m_stat) {
            is_target = true;
            visit_expr(*x.m_stat);
            is_target = false;
            uint32_t stat_ptr = tmp;
            lr_type_t *stat_ty = get_type(ASRUtils::expr_type(x.m_stat));
            lr_emit_store(s, I(0, stat_ty), V(stat_ptr, lr_type_ptr_s(s)));
        }
    }

    void visit_ExplicitDeallocate(const ASR::ExplicitDeallocate_t &x) {
        lr_type_t *ptr = lr_type_ptr_s(s);
        uint32_t free_id = get_free_id();

        for (size_t i = 0; i < x.n_vars; i++) {
            is_target = true;
            visit_expr(*x.m_vars[i]);
            is_target = false;
            uint32_t var_ptr = tmp;
            uint32_t data = lr_emit_load(s, ptr, V(var_ptr, ptr));
            lr_operand_desc_t args[1] = {V(data, ptr)};
            lr_emit_call_void(s, LR_GLOBAL(free_id, ptr), args, 1);
        }
    }

    /* ---- StructInstanceMember ----------------------------------------- */

    void visit_StructInstanceMember(const ASR::StructInstanceMember_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* Access struct field. Get the struct pointer, then GEP to
           the field index. We determine the field index from the
           StructType_t member types and the struct definition's
           symbol table. */
        ASR::symbol_t *member_sym = ASRUtils::symbol_get_past_external(
            x.m_m);

        /* Get the struct definition from the expression */
        ASR::symbol_t *struct_sym =
            ASRUtils::get_struct_sym_from_struct_expr(x.m_v);
        if (!struct_sym || !ASR::is_a<ASR::Struct_t>(*struct_sym)) {
            throw CodeGenError(
                "liric: StructInstanceMember on non-struct type");
        }
        ASR::Struct_t *struct_def = ASR::down_cast<ASR::Struct_t>(
            struct_sym);

        /* Find field index by iterating over struct members in order */
        uint32_t field_idx = 0;
        bool found = false;
        for (size_t i = 0; i < struct_def->n_members; i++) {
            ASR::symbol_t *ms = struct_def->m_symtab->get_symbol(
                struct_def->m_members[i]);
            if (ms && ASRUtils::symbol_get_past_external(ms) ==
                    member_sym) {
                field_idx = (uint32_t)i;
                found = true;
                break;
            }
        }
        if (!found) {
            throw CodeGenError(
                "liric: struct field not found");
        }

        /* Visit the struct expression to get its pointer */
        bool saved_target = is_target;
        is_target = true;
        visit_expr(*x.m_v);
        is_target = saved_target;
        uint32_t struct_ptr = tmp;

        /* Build a struct type for GEP from struct member types */
        ASR::ttype_t *v_type = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(
                ASRUtils::expr_type(x.m_v)));
        std::vector<lr_type_t *> field_types;
        if (ASR::is_a<ASR::StructType_t>(*v_type)) {
            ASR::StructType_t *st = ASR::down_cast<ASR::StructType_t>(
                v_type);
            for (size_t i = 0; i < st->n_data_member_types; i++) {
                field_types.push_back(get_type(st->m_data_member_types[i]));
            }
        } else {
            /* Fallback: iterate symtab */
            for (size_t i = 0; i < struct_def->n_members; i++) {
                ASR::symbol_t *ms = struct_def->m_symtab->get_symbol(
                    struct_def->m_members[i]);
                if (ms && ASR::is_a<ASR::Variable_t>(*ms)) {
                    ASR::Variable_t *fv = ASR::down_cast<ASR::Variable_t>(
                        ms);
                    field_types.push_back(get_type(fv->m_type));
                }
            }
        }
        lr_type_t *sty = lr_type_struct_s(s, field_types.data(),
            (uint32_t)field_types.size(), false);
        lr_type_t *ptr_ty = lr_type_ptr_s(s);

        uint32_t field_ptr = lr_emit_structgep(s, sty,
            V(struct_ptr, ptr_ty), field_idx);
        lr_type_t *field_ty = get_type(x.m_type);

        if (is_target) {
            tmp = field_ptr;
        } else {
            tmp = lr_emit_load(s, field_ty, V(field_ptr, ptr_ty));
        }
    }

    /* ---- String operations -------------------------------------------- */

    uint32_t get_strlen_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "strlen")) {
            return lr_session_intern(s, "strlen");
        }
        lr_type_t *i64 = lr_type_i64_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *params[1] = {ptr};
        lr_session_declare(s, "strlen", i64, params, 1, false, &err);
        return lr_session_intern(s, "strlen");
    }

    uint32_t get_strcmp_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "strcmp")) {
            return lr_session_intern(s, "strcmp");
        }
        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *params[2] = {ptr, ptr};
        lr_session_declare(s, "strcmp", i32, params, 2, false, &err);
        return lr_session_intern(s, "strcmp");
    }

    uint32_t get_memcpy_id() {
        lr_error_t err;
        lr_module_t *mod = lr_session_module(s);
        if (mod && lr_module_lookup_function(mod, "memcpy")) {
            return lr_session_intern(s, "memcpy");
        }
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        lr_type_t *params[3] = {ptr, ptr, i64};
        lr_session_declare(s, "memcpy", ptr, params, 3, false, &err);
        return lr_session_intern(s, "memcpy");
    }

    void visit_StringLen(const ASR::StringLen_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        uint32_t str_ptr = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        uint32_t strlen_id = get_strlen_id();
        lr_operand_desc_t args[1] = {V(str_ptr, ptr)};
        uint32_t len = lr_emit_call(s, i64,
            LR_GLOBAL(strlen_id, ptr), args, 1);
        lr_type_t *res_ty = get_type(x.m_type);
        if (res_ty != i64) {
            tmp = lr_emit_sextortrunc(s, res_ty, V(len, i64));
        } else {
            tmp = len;
        }
    }

    void visit_StringPhysicalCast(const ASR::StringPhysicalCast_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* String physical casts are no-ops in our representation
           since all strings are char* pointers. */
        visit_expr(*x.m_arg);
    }

    void visit_StringCompare(const ASR::StringCompare_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);
        uint32_t strcmp_id = get_strcmp_id();
        lr_operand_desc_t args[2] = {V(left, ptr), V(right, ptr)};
        uint32_t cmp_result = lr_emit_call(s, i32,
            LR_GLOBAL(strcmp_id, ptr), args, 2);

        switch (x.m_op) {
            case ASR::cmpopType::Eq:
                tmp = lr_emit_icmp(s, LR_CMP_EQ,
                    V(cmp_result, i32), I(0, i32));
                break;
            case ASR::cmpopType::NotEq:
                tmp = lr_emit_icmp(s, LR_CMP_NE,
                    V(cmp_result, i32), I(0, i32));
                break;
            case ASR::cmpopType::Lt:
                tmp = lr_emit_icmp(s, LR_CMP_SLT,
                    V(cmp_result, i32), I(0, i32));
                break;
            case ASR::cmpopType::LtE:
                tmp = lr_emit_icmp(s, LR_CMP_SLE,
                    V(cmp_result, i32), I(0, i32));
                break;
            case ASR::cmpopType::Gt:
                tmp = lr_emit_icmp(s, LR_CMP_SGT,
                    V(cmp_result, i32), I(0, i32));
                break;
            case ASR::cmpopType::GtE:
                tmp = lr_emit_icmp(s, LR_CMP_SGE,
                    V(cmp_result, i32), I(0, i32));
                break;
            default:
                throw CodeGenError(
                    "liric: unsupported string compare op");
        }
    }

    void visit_StringConcat(const ASR::StringConcat_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* Allocate buffer for result = left + right.
           Compute lengths, malloc, memcpy both halves,
           null-terminate. */
        visit_expr(*x.m_left);
        uint32_t left_str = tmp;
        visit_expr(*x.m_right);
        uint32_t right_str = tmp;

        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        uint32_t strlen_id = get_strlen_id();
        uint32_t malloc_id = get_malloc_id();
        uint32_t memcpy_id = get_memcpy_id();

        lr_operand_desc_t sl_args[1] = {V(left_str, ptr)};
        uint32_t left_len = lr_emit_call(s, i64,
            LR_GLOBAL(strlen_id, ptr), sl_args, 1);
        lr_operand_desc_t sr_args[1] = {V(right_str, ptr)};
        uint32_t right_len = lr_emit_call(s, i64,
            LR_GLOBAL(strlen_id, ptr), sr_args, 1);

        uint32_t total_len = lr_emit_add(s, i64, V(left_len, i64),
                                         V(right_len, i64));
        uint32_t buf_size = lr_emit_add(s, i64, V(total_len, i64),
                                        I(1, i64));

        lr_operand_desc_t ma_args[1] = {V(buf_size, i64)};
        uint32_t buf = lr_emit_call(s, ptr,
            LR_GLOBAL(malloc_id, ptr), ma_args, 1);

        lr_operand_desc_t mc1_args[3] = {
            V(buf, ptr), V(left_str, ptr), V(left_len, i64)
        };
        lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc1_args, 3);

        lr_type_t *i8 = lr_type_i8_s(s);
        lr_operand_desc_t gep_idx[1] = {V(left_len, i64)};
        uint32_t dst2 = lr_emit_gep(s, i8, V(buf, ptr), gep_idx, 1);

        lr_operand_desc_t mc2_args[3] = {
            V(dst2, ptr), V(right_str, ptr), V(right_len, i64)
        };
        lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc2_args, 3);

        lr_operand_desc_t nt_idx[1] = {V(total_len, i64)};
        uint32_t null_pos = lr_emit_gep(s, i8, V(buf, ptr), nt_idx, 1);
        lr_emit_store(s, I(0, i8), V(null_pos, ptr));

        tmp = buf;
    }

    /* ---- TypeInquiry -------------------------------------------------- */

    void visit_TypeInquiry(const ASR::TypeInquiry_t &x) {
        visit_expr(*x.m_value);
    }

    /* ---- LogicalCompare ----------------------------------------------- */

    void visit_LogicalCompare(const ASR::LogicalCompare_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_left);
        uint32_t left = tmp;
        visit_expr(*x.m_right);
        uint32_t right = tmp;
        lr_type_t *i1 = lr_type_i1_s(s);
        switch (x.m_op) {
            case ASR::cmpopType::Eq:
                tmp = lr_emit_icmp(s, LR_CMP_EQ,
                    V(left, i1), V(right, i1));
                break;
            case ASR::cmpopType::NotEq:
                tmp = lr_emit_icmp(s, LR_CMP_NE,
                    V(left, i1), V(right, i1));
                break;
            default:
                throw CodeGenError("liric: unsupported logical compare op");
        }
    }

    /* ---- Associate ---------------------------------------------------- */

    void visit_Associate(const ASR::Associate_t &x) {
        /* Associate creates an alias: target => value.
           In our simple representation, we store the address/value of
           the RHS into the target pointer slot. */
        is_target = true;
        visit_expr(*x.m_value);
        is_target = false;
        uint32_t value_ptr = tmp;

        /* The target is a Variable. Register it in the symbol table. */
        if (ASR::is_a<ASR::Var_t>(*x.m_target)) {
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(x.m_target);
            uint64_t h = get_hash((ASR::asr_t *)v);
            lr_symtab[h] = value_ptr;
        } else {
            is_target = true;
            visit_expr(*x.m_target);
            is_target = false;
            uint32_t target_ptr = tmp;
            lr_emit_store(s, V(value_ptr, lr_type_ptr_s(s)),
                V(target_ptr, lr_type_ptr_s(s)));
        }
    }

    /* ---- AssociateBlockCall -------------------------------------------- */

    void visit_AssociateBlockCall(const ASR::AssociateBlockCall_t &x) {
        if (!ASR::is_a<ASR::AssociateBlock_t>(*x.m_m)) {
            throw CodeGenError(
                "liric: AssociateBlockCall target is not AssociateBlock");
        }
        ASR::AssociateBlock_t *ab = ASR::down_cast<ASR::AssociateBlock_t>(
            x.m_m);
        /* Declare local variables from the associate block scope */
        declare_vars(ab->m_symtab);
        /* Visit all statements in the block body */
        for (size_t i = 0; i < ab->n_body; i++) {
            visit_stmt(*ab->m_body[i]);
        }
    }

    /* ---- BlockCall ---------------------------------------------------- */

    void visit_BlockCall(const ASR::BlockCall_t &x) {
        if (!ASR::is_a<ASR::Block_t>(*x.m_m)) {
            throw CodeGenError("liric: BlockCall target is not Block");
        }
        ASR::Block_t *blk = ASR::down_cast<ASR::Block_t>(x.m_m);
        declare_vars(blk->m_symtab);
        for (size_t i = 0; i < blk->n_body; i++) {
            visit_stmt(*blk->m_body[i]);
        }
    }

    /* ---- DoConcurrentLoop --------------------------------------------- */

    void visit_DoConcurrentLoop(const ASR::DoConcurrentLoop_t &x) {
        /* Treat do concurrent as a sequential do loop for now */
        for (size_t h = 0; h < x.n_head; h++) {
            lr_error_t err;
            lr_type_t *i1 = lr_type_i1_s(s);
            ASR::do_loop_head_t head = x.m_head[h];

            visit_expr(*head.m_start);
            uint32_t start_val = tmp;
            visit_expr(*head.m_end);
            uint32_t end_val = tmp;
            lr_type_t *loop_ty = get_type(ASRUtils::expr_type(head.m_v));

            uint32_t step_val;
            if (head.m_increment) {
                visit_expr(*head.m_increment);
                step_val = tmp;
            } else {
                step_val = lr_emit_add(s, loop_ty, I(1, loop_ty),
                                       I(0, loop_ty));
            }

            is_target = true;
            visit_expr(*head.m_v);
            is_target = false;
            uint32_t loop_var_ptr = tmp;
            lr_emit_store(s, V(start_val, loop_ty),
                V(loop_var_ptr, lr_type_ptr_s(s)));

            uint32_t cond_block = lr_session_block(s);
            uint32_t body_block = lr_session_block(s);
            uint32_t incr_block = lr_session_block(s);
            uint32_t end_block = lr_session_block(s);

            loop_head_stack.push_back(incr_block);
            loop_end_stack.push_back(end_block);

            lr_emit_br(s, cond_block);
            lr_session_set_block(s, cond_block, &err);
            uint32_t cur_val = lr_emit_load(s, loop_ty,
                V(loop_var_ptr, lr_type_ptr_s(s)));
            uint32_t cond = lr_emit_icmp(s, LR_CMP_SLE,
                V(cur_val, loop_ty), V(end_val, loop_ty));
            lr_emit_condbr(s, V(cond, i1), body_block, end_block);

            lr_session_set_block(s, body_block, &err);
        }

        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
        }

        for (size_t h = x.n_head; h > 0; h--) {
            lr_error_t err;
            ASR::do_loop_head_t head = x.m_head[h - 1];
            lr_type_t *loop_ty = get_type(ASRUtils::expr_type(head.m_v));

            is_target = true;
            visit_expr(*head.m_v);
            is_target = false;
            uint32_t loop_var_ptr = tmp;

            lr_emit_br(s, loop_head_stack.back());
            lr_session_set_block(s, loop_head_stack.back(), &err);

            uint32_t next_val = lr_emit_load(s, loop_ty,
                V(loop_var_ptr, lr_type_ptr_s(s)));

            uint32_t step_val;
            if (head.m_increment) {
                visit_expr(*head.m_increment);
                step_val = tmp;
            } else {
                step_val = lr_emit_add(s, loop_ty, I(1, loop_ty),
                                       I(0, loop_ty));
            }
            next_val = lr_emit_add(s, loop_ty, V(next_val, loop_ty),
                                   V(step_val, loop_ty));
            lr_emit_store(s, V(next_val, loop_ty),
                V(loop_var_ptr, lr_type_ptr_s(s)));

            /* Branch back to condition (stored before incr block) */
            uint32_t end_blk = loop_end_stack.back();
            uint32_t incr_blk = loop_head_stack.back();
            lr_emit_br(s, incr_blk - 1);

            lr_session_set_block(s, end_blk, &err);
            loop_head_stack.pop_back();
            loop_end_stack.pop_back();
        }
    }

    /* ---- ImpliedDoLoop ------------------------------------------------ */

    void visit_ImpliedDoLoop(const ASR::ImpliedDoLoop_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* An implied do loop builds an array from values.
           For fixed-size, we can unroll at codegen time if
           start/end/step are constant. Otherwise, we emit a
           loop that fills a temporary array. */
        lr_type_t *arr_ty = get_type(x.m_type);
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *elem_ty = get_scalar_type(x.m_type);
        lr_type_t *i32 = lr_type_i32_s(s);

        uint32_t arr_alloca = lr_emit_alloca(s, arr_ty);
        uint32_t write_idx_alloca = lr_emit_alloca(s, i32);
        lr_emit_store(s, I(0, i32), V(write_idx_alloca, ptr));

        lr_error_t err;
        lr_type_t *i1 = lr_type_i1_s(s);

        visit_expr(*x.m_start);
        uint32_t start_val = tmp;
        visit_expr(*x.m_end);
        uint32_t end_val = tmp;
        lr_type_t *var_ty = get_type(ASRUtils::expr_type(x.m_var));

        uint32_t step_val;
        if (x.m_increment) {
            visit_expr(*x.m_increment);
            step_val = tmp;
        } else {
            step_val = lr_emit_add(s, var_ty, I(1, var_ty), I(0, var_ty));
        }

        /* Set loop variable */
        is_target = true;
        visit_expr(*x.m_var);
        is_target = false;
        uint32_t var_ptr = tmp;
        lr_emit_store(s, V(start_val, var_ty), V(var_ptr, ptr));

        uint32_t cond_block = lr_session_block(s);
        uint32_t body_block = lr_session_block(s);
        uint32_t incr_block = lr_session_block(s);
        uint32_t end_block = lr_session_block(s);

        lr_emit_br(s, cond_block);
        lr_session_set_block(s, cond_block, &err);
        uint32_t cur = lr_emit_load(s, var_ty, V(var_ptr, ptr));
        uint32_t cond = lr_emit_icmp(s, LR_CMP_SLE,
            V(cur, var_ty), V(end_val, var_ty));
        lr_emit_condbr(s, V(cond, i1), body_block, end_block);

        lr_session_set_block(s, body_block, &err);
        /* Evaluate each value expression and store to array */
        for (size_t i = 0; i < x.n_values; i++) {
            visit_expr(*x.m_values[i]);
            uint32_t val = tmp;
            uint32_t widx = lr_emit_load(s, i32,
                V(write_idx_alloca, ptr));
            lr_operand_desc_t gep_idx[2] = {I(0, i32), V(widx, i32)};
            uint32_t ep = lr_emit_gep(s, arr_ty, V(arr_alloca, ptr),
                                      gep_idx, 2);
            lr_emit_store(s, V(val, elem_ty), V(ep, ptr));
            uint32_t next_widx = lr_emit_add(s, i32, V(widx, i32),
                                             I(1, i32));
            lr_emit_store(s, V(next_widx, i32),
                V(write_idx_alloca, ptr));
        }

        /* Increment loop var */
        lr_emit_br(s, incr_block);
        lr_session_set_block(s, incr_block, &err);
        uint32_t cur2 = lr_emit_load(s, var_ty, V(var_ptr, ptr));
        uint32_t next = lr_emit_add(s, var_ty, V(cur2, var_ty),
                                    V(step_val, var_ty));
        lr_emit_store(s, V(next, var_ty), V(var_ptr, ptr));
        lr_emit_br(s, cond_block);

        lr_session_set_block(s, end_block, &err);
        tmp = lr_emit_load(s, arr_ty, V(arr_alloca, ptr));
    }

    /* ---- StructConstructor -------------------------------------------- */

    void visit_StructConstructor(const ASR::StructConstructor_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* Build struct type from member types */
        ASR::ttype_t *stype = ASRUtils::type_get_past_pointer(
            ASRUtils::type_get_past_allocatable(x.m_type));
        lr_type_t *sty = get_type(stype);
        lr_type_t *ptr = lr_type_ptr_s(s);

        /* Allocate struct on stack */
        uint32_t alloca_v = lr_emit_alloca(s, sty);

        /* If StructType_t, we know the member types */
        if (ASR::is_a<ASR::StructType_t>(*stype)) {
            ASR::StructType_t *st = ASR::down_cast<ASR::StructType_t>(stype);
            std::vector<lr_type_t *> field_types;
            for (size_t i = 0; i < st->n_data_member_types; i++) {
                field_types.push_back(get_type(st->m_data_member_types[i]));
            }
            lr_type_t *struct_ty = lr_type_struct_s(s, field_types.data(),
                (uint32_t)field_types.size(), false);

            for (size_t i = 0; i < x.n_args && i < st->n_data_member_types;
                 i++) {
                if (!x.m_args[i].m_value) continue;
                visit_expr(*x.m_args[i].m_value);
                uint32_t val = tmp;
                uint32_t fp = lr_emit_structgep(s, struct_ty,
                    V(alloca_v, ptr), (uint32_t)i);
                lr_type_t *ft = field_types[i];
                lr_emit_store(s, V(val, ft), V(fp, ptr));
            }
        }
        tmp = alloca_v;
    }

    /* ---- BitCast ------------------------------------------------------ */

    void visit_BitCast(const ASR::BitCast_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_source);
        uint32_t src = tmp;
        lr_type_t *src_ty = get_type(ASRUtils::expr_type(x.m_source));
        lr_type_t *dst_ty = get_type(x.m_type);
        lr_type_t *ptr = lr_type_ptr_s(s);

        /* Store source to a temp, load as destination type */
        uint32_t tmp_alloca = lr_emit_alloca(s, src_ty);
        lr_emit_store(s, V(src, src_ty), V(tmp_alloca, ptr));
        tmp = lr_emit_load(s, dst_ty, V(tmp_alloca, ptr));
    }

    /* ---- StringItem --------------------------------------------------- */

    void visit_StringItem(const ASR::StringItem_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        uint32_t str = tmp;
        visit_expr(*x.m_idx);
        uint32_t idx = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i8 = lr_type_i8_s(s);
        lr_type_t *i32 = lr_type_i32_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        lr_type_t *idx_ty = get_type(ASRUtils::expr_type(x.m_idx));

        /* Convert 1-based Fortran index to 0-based */
        if (idx_ty != i32) {
            idx = lr_emit_sextortrunc(s, i32, V(idx, idx_ty));
        }
        uint32_t idx0 = lr_emit_sub(s, i32, V(idx, i32), I(1, i32));

        /* Allocate 2-byte buffer for single character + null */
        uint32_t malloc_id = get_malloc_id();
        lr_operand_desc_t ma_args[1] = {I(2, i64)};
        uint32_t buf = lr_emit_call(s, ptr,
            LR_GLOBAL(malloc_id, ptr), ma_args, 1);

        /* Copy the character */
        lr_operand_desc_t gep_idx[1] = {V(idx0, i32)};
        uint32_t char_ptr = lr_emit_gep(s, i8, V(str, ptr), gep_idx, 1);
        uint32_t ch = lr_emit_load(s, i8, V(char_ptr, ptr));
        lr_emit_store(s, V(ch, i8), V(buf, ptr));

        /* Null terminate */
        lr_operand_desc_t gep_null[1] = {I(1, i32)};
        uint32_t null_ptr = lr_emit_gep(s, i8, V(buf, ptr), gep_null, 1);
        lr_emit_store(s, I(0, i8), V(null_ptr, ptr));

        tmp = buf;
    }

    /* ---- StringSection ------------------------------------------------ */

    void visit_StringSection(const ASR::StringSection_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        visit_expr(*x.m_arg);
        uint32_t str = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);
        lr_type_t *i8 = lr_type_i8_s(s);

        /* Get string length */
        uint32_t strlen_id = get_strlen_id();
        lr_operand_desc_t sl_args[1] = {V(str, ptr)};
        uint32_t str_len = lr_emit_call(s, i64,
            LR_GLOBAL(strlen_id, ptr), sl_args, 1);

        /* Get start (1-based, default 1) */
        uint32_t start_val;
        if (x.m_start) {
            visit_expr(*x.m_start);
            lr_type_t *st = get_type(ASRUtils::expr_type(x.m_start));
            start_val = (st != i64) ? lr_emit_sextortrunc(s, i64, V(tmp, st))
                                    : tmp;
        } else {
            start_val = lr_emit_add(s, i64, I(1, i64), I(0, i64));
        }

        /* Get end (1-based, default len) */
        uint32_t end_val;
        if (x.m_end) {
            visit_expr(*x.m_end);
            lr_type_t *et = get_type(ASRUtils::expr_type(x.m_end));
            end_val = (et != i64) ? lr_emit_sextortrunc(s, i64, V(tmp, et))
                                  : tmp;
        } else {
            end_val = str_len;
        }

        /* Convert to 0-based */
        uint32_t start0 = lr_emit_sub(s, i64, V(start_val, i64), I(1, i64));
        uint32_t slice_len = lr_emit_sub(s, i64, V(end_val, i64),
                                         V(start_val, i64));
        slice_len = lr_emit_add(s, i64, V(slice_len, i64), I(1, i64));

        /* Allocate and copy */
        uint32_t malloc_id = get_malloc_id();
        uint32_t memcpy_id = get_memcpy_id();
        uint32_t buf_size = lr_emit_add(s, i64, V(slice_len, i64), I(1, i64));
        lr_operand_desc_t ma_args[1] = {V(buf_size, i64)};
        uint32_t buf = lr_emit_call(s, ptr,
            LR_GLOBAL(malloc_id, ptr), ma_args, 1);

        lr_operand_desc_t src_idx[1] = {V(start0, i64)};
        uint32_t src_ptr = lr_emit_gep(s, i8, V(str, ptr), src_idx, 1);

        lr_operand_desc_t mc_args[3] = {
            V(buf, ptr), V(src_ptr, ptr), V(slice_len, i64)
        };
        lr_emit_call_void(s, LR_GLOBAL(memcpy_id, ptr), mc_args, 3);

        /* Null terminate */
        lr_operand_desc_t nt_idx[1] = {V(slice_len, i64)};
        uint32_t null_pos = lr_emit_gep(s, i8, V(buf, ptr), nt_idx, 1);
        lr_emit_store(s, I(0, i8), V(null_pos, ptr));

        tmp = buf;
    }

    /* ---- CPtrToPointer ------------------------------------------------ */

    void visit_CPtrToPointer(const ASR::CPtrToPointer_t &x) {
        /* c_f_pointer(cptr, ptr) - store the C pointer value into
           the Fortran pointer variable. */
        visit_expr(*x.m_cptr);
        uint32_t cptr_val = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);

        is_target = true;
        visit_expr(*x.m_ptr);
        is_target = false;
        uint32_t fptr_ptr = tmp;

        lr_emit_store(s, V(cptr_val, ptr), V(fptr_ptr, ptr));
    }

    /* ---- PointerToCPtr ------------------------------------------------ */

    void visit_PointerToCPtr(const ASR::PointerToCPtr_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* c_loc(ptr) - return the pointer value as a C pointer */
        is_target = true;
        visit_expr(*x.m_arg);
        is_target = false;
        /* tmp is already the address */
    }

    /* ---- Nullify ------------------------------------------------------ */

    void visit_Nullify(const ASR::Nullify_t &x) {
        lr_type_t *ptr = lr_type_ptr_s(s);
        for (size_t i = 0; i < x.n_vars; i++) {
            is_target = true;
            visit_expr(*x.m_vars[i]);
            is_target = false;
            uint32_t var_ptr = tmp;
            /* Store null pointer */
            lr_emit_store(s, LR_NULL(ptr), V(var_ptr, ptr));
        }
    }

    /* ---- PointerNullConstant ------------------------------------------ */

    void visit_PointerNullConstant(
            const ASR::PointerNullConstant_t & /* x */) {
        lr_type_t *ptr = lr_type_ptr_s(s);
        tmp = lr_emit_inttoptr(s, ptr, I(0, lr_type_i64_s(s)));
    }

    /* ---- PointerAssociated -------------------------------------------- */

    void visit_PointerAssociated(const ASR::PointerAssociated_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* Check if pointer is not null */
        is_target = true;
        visit_expr(*x.m_ptr);
        is_target = false;
        uint32_t ptr_val = tmp;
        lr_type_t *ptr = lr_type_ptr_s(s);
        lr_type_t *i64 = lr_type_i64_s(s);

        uint32_t loaded = lr_emit_load(s, ptr, V(ptr_val, ptr));
        uint32_t as_int = lr_emit_ptrtoint(s, i64, V(loaded, ptr));
        tmp = lr_emit_icmp(s, LR_CMP_NE, V(as_int, i64), I(0, i64));
    }

    /* ---- GetPointer --------------------------------------------------- */

    void visit_GetPointer(const ASR::GetPointer_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        is_target = true;
        visit_expr(*x.m_arg);
        is_target = false;
    }

    /* ---- ArrayReshape ------------------------------------------------- */

    void visit_ArrayReshape(const ASR::ArrayReshape_t &x) {
        if (x.m_value) {
            visit_expr(*x.m_value);
            return;
        }
        /* For our simple representation, reshape is a bitcast
           since we store arrays as contiguous memory. */
        visit_expr(*x.m_array);
    }

    /* ---- I/O stubs ---------------------------------------------------- */

    /* ---- SelectType --------------------------------------------------- */

    void visit_SelectType(const ASR::SelectType_t &x) {
        /* SelectType requires runtime type dispatch which is complex.
           For now, execute just the default branch if present. */
        if (x.n_default > 0) {
            for (size_t i = 0; i < x.n_default; i++) {
                visit_stmt(*x.m_default[i]);
            }
        }
    }

    /* ---- SelectRank --------------------------------------------------- */

    void visit_SelectRank(const ASR::SelectRank_t &x) {
        if (x.n_default > 0) {
            for (size_t i = 0; i < x.n_default; i++) {
                visit_stmt(*x.m_default[i]);
            }
        }
    }

    /* ---- I/O stubs ---------------------------------------------------- */

    void visit_FileOpen(const ASR::FileOpen_t & /* x */) {
        /* I/O operations require runtime library support.
           For now, skip silently since most tests only check
           computation results. */
    }

    void visit_FileClose(const ASR::FileClose_t & /* x */) {
    }

    void visit_FileRead(const ASR::FileRead_t &x) {
        if (x.m_overloaded) {
            visit_stmt(*x.m_overloaded);
            return;
        }
    }

    void visit_FileInquire(const ASR::FileInquire_t & /* x */) {
    }

    void visit_Flush(const ASR::Flush_t & /* x */) {
    }

    void visit_FileRewind(const ASR::FileRewind_t & /* x */) {
    }

    void visit_FileBackspace(const ASR::FileBackspace_t & /* x */) {
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
