/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_c_host.cc
 */
#include "codegen_c_host.h"

#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/str_escape.h"
#include "../build_common.h"
#include "../func_registry_generator.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

CodeGenCHost::CodeGenCHost() { module_name_ = name_supply_->FreshName("__tvm_module_ctx"); }

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
                        std::string target_str, const std::unordered_set<std::string>& devices) {
  output_ssa_ = output_ssa;
  target_str_ = target_str;
  devices_ = devices;
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  declared_globals_.clear();
  static_init_stream_.str("");
  static_init_stream_.clear();
  static_init_stream_sink_ = &static_init_stream_;
  static_init_declarations_.clear();
  static_init_declarations_sink_ = &static_init_declarations_;
  static_init_declaration_stream_sink_ = &decl_stream;
  static_init_name_supply_ = NameSupply();
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#define TVM_EXPORTS\n";
  decl_stream << "#include \"tvm/runtime/c_runtime_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "#include <math.h>\n";
  decl_stream << "#include <stdbool.h>\n";
  if (devices.find("ethos-u") != devices.end()) {
    decl_stream << "#include <tvm_ethosu_runtime.h>\n";
  }
  if (devices.find("cmsis-nn") != devices.end()) {
    decl_stream << "#include <stdio.h>\n";
    decl_stream << "#include <stdlib.h>\n";
    decl_stream << "#include <dlpack/dlpack.h>\n";
    decl_stream << "#include <arm_nnfunctions.h>\n";
    decl_stream << "#include <arm_nn_types.h>\n";
    decl_stream << "#include <arm_nn_math_types.h>\n";
  }
  CodeGenC::Init(output_ssa);
}

std::string CodeGenCHost::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << fwd_decl_stream.str();
  code << static_init_stream_.str();
  code << stream.str();
  return code.str();
}

void CodeGenCHost::InitGlobalContext(bool weak) {
  decl_stream << "void* ";
  if (weak) {
    decl_stream << "TVM_WEAK ";
  }
  decl_stream << tvm::runtime::symbol::tvm_module_ctx << " = NULL;\n";
}

void CodeGenCHost::DefineModuleName() { decl_stream << "void* " << module_name_ << " = NULL;\n"; }

void CodeGenCHost::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  return AddFunction(gvar, func, /*emit_fwd_func_decl=*/false);
}

void CodeGenCHost::AddFunction(const GlobalVar& gvar, const PrimFunc& func,
                               bool emit_fwd_func_decl) {
  if (HasStaticInitScope(func->body) && !ReturnsInt32(func->ret_type)) {
    LOG(FATAL) << "CodeGenCHost: coproc_uop_scope requires an int32-returning function "
               << "so initializer failures can be propagated; function " << gvar
               << " returns " << func->ret_type;
  }
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  if (global_symbol) {
    function_names_.push_back(global_symbol.value());
  }

  emit_fwd_func_decl_ = emit_fwd_func_decl;
  // Keep the base generator's complete per-function lifecycle, while adding
  // an explicit success return for static-uop functions.  A static initializer
  // can fail and return -1, so an int32 entry must also have a defined success
  // value instead of relying on C's undefined fall-through return.
  DeclareFunction(gvar, func);
  auto function_name = GetFunctionName(gvar);
  InitFuncState(func);
  PrintFunctionSignature(function_name, func, stream);
  stream << " {\n";
  PreFunctionBody(func);
  int func_scope = BeginScope();
  PrintStmt(func->body);
  EndScope(func_scope);
  if (HasStaticInitScope(func->body) && ReturnsInt32(func->ret_type)) {
    PrintIndent();
    stream << "return 0;\n";
  }
  PrintIndent();
  stream << "}\n\n";
  if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
    ICHECK(global_symbol.defined())
        << "CodeGenCHost: The entry func must have the global_symbol attribute, "
        << "but function " << gvar << " only has attributes " << func->attrs;

    function_names_.push_back(runtime::symbol::tvm_module_main);
    stream << "// CodegenC: NOTE: Auto-generated entry function\n";
    PrintFuncPrefix(stream);
    PrintType(func->ret_type, stream);
    stream << " " << tvm::runtime::symbol::tvm_module_main
           << "(void* args, int* arg_type_ids, int num_args, void* out_ret_value, "
           << "int* out_ret_tcode, void* resource_handle) {\n";
    stream << "  return " << global_symbol.value()
           << "(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);\n";
    stream << "}\n";
  }
}

bool CodeGenCHost::HasStaticInitScope(const Stmt& stmt) const {
  bool found = false;
  tir::PostOrderVisit(stmt, [&found](const ObjectRef& node) {
    if (node.as<AttrStmtNode>() != nullptr &&
        node.as<AttrStmtNode>()->attr_key == tir::attr::coproc_uop_scope) {
      found = true;
    }
  });
  return found;
}

bool CodeGenCHost::ReturnsInt32(const Type& ret_type) const {
  const auto* prim_type = ret_type.as<PrimTypeNode>();
  return prim_type != nullptr && prim_type->dtype == DataType::Int(32);
}

void CodeGenCHost::GenerateForwardFunctionDeclarations(String global_symbol,

                                                       const Array<Type>& arg_types,
                                                       const Type& ret_type) {
  if (!emit_fwd_func_decl_) {
    return;
  }
  // These runtime entry points are declared by c_backend_api.h.  Repeating a
  // declaration inferred from lowered TIR can give a different pointer type
  // (for example, float* instead of void*) and is rejected by C++ compilers.
  if (global_symbol == "TVMBackendAllocWorkspace" ||
      global_symbol == "TVMBackendFreeWorkspace") {
    return;
  }
  for (auto& func_already_defined : GetFunctionNames()) {
    if (global_symbol == func_already_defined) {
      return;
    }
  }
  this->PrintFuncPrefix(fwd_decl_stream);
  this->PrintType(ret_type, fwd_decl_stream);
  fwd_decl_stream << " " << global_symbol << "(";
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (i > 0) {
      fwd_decl_stream << ", ";
    }
    this->PrintType(arg_types[i], fwd_decl_stream);
  }
  fwd_decl_stream << ");\n";
}

void CodeGenCHost::PrintFuncPrefix(std::ostream& os) {  // NOLINT(*)
  os << "#ifdef __cplusplus\n"
     << "extern \"C\"\n"
     << "#endif\n"
     << "TVM_DLL ";
}

void CodeGenCHost::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenCHost::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->dtype.lanes();
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenCHost::PrintGetFuncFromBackend(const std::string& func_name,
                                           const std::string& packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_ << ", \"" << func_name << "\""
               << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::PrintFuncCall(const std::string& packed_func_name, int num_args) {
  this->PrintIndent();
  std::string ret_val = name_supply_->FreshName("ret_val");
  std::string ret_type_code = name_supply_->FreshName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();
  this->stream << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMValue*) stack_value"
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::PrintFuncCallC(const std::string& packed_func_name, int num_args,
                                  const std::string& resource_handle_name) {
  this->PrintIndent();
  std::string ret_val = name_supply_->FreshName("ret_val");
  std::string ret_type_code = name_supply_->FreshName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();

  this->stream << "if (" << packed_func_name << "( "
               << "(TVMValue*) stack_value "
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ", " << resource_handle_name << ") != 0){\n";

  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

std::string CodeGenCHost::GetPackedName(const CallNode* op) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
  std::string func_name = s->value;
  std::string packed_func_name = func_name + "_packed";
  std::string unique_name;
  auto it = declared_globals_.find(packed_func_name);
  if (it != declared_globals_.end()) {
    unique_name = it->second;
  } else {
    unique_name = name_supply_->FreshName(packed_func_name);
    declared_globals_[packed_func_name] = unique_name;
    decl_stream << "static void* " << unique_name << " = NULL;\n";
  }
  return unique_name;
}

CodeGenCHost::FunctionInfo CodeGenCHost::GetFunctionInfo(const CallNode* op,
                                                         bool has_resource_handle) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_[c]packed_lowered expects first argument as function name";
  int64_t begin = op->args[3].as<IntImmNode>()->value;
  int64_t end = op->args[4].as<IntImmNode>()->value;
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0);
  std::string func_name = s->value;

  if (has_resource_handle) {
    const StringImmNode* resource_handle_var = op->args[5].as<StringImmNode>();
    if (resource_handle_var != nullptr) {
      std::string resource_handle_name = resource_handle_var->value;
      return {func_name, num_args - 1, resource_handle_name};
    } else {
      // The final arg should be "(void*) NULL" to indicate the empty resource_handle.
      num_args--;

      const CallNode* reinterpret_call = op->args[5].as<CallNode>();
      ICHECK_NE(reinterpret_call, (void*)nullptr)
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK_EQ(reinterpret_call->op, builtin::reinterpret())
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK(is_zero(reinterpret_call->args[0])) << "At CallNode to " << s
                                                 << " arg 5: Expect either StringImm naming the "
                                                    "resource_handle var from interface API, or "
                                                 << "zero; got " << op->args[5];
    }
  }
  return {func_name, num_args, "NULL"};
}

void CodeGenCHost::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMValue " << stack_name << "[" << size << "];\n";
    os << stack_name;
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    auto function_info = GetFunctionInfo(op, false /* has_resource_handle */);
    std::string func_name_packed = GetPackedName(op);
    this->PrintGetFuncFromBackend(function_info.func_name, func_name_packed);
    this->PrintFuncCall(func_name_packed, function_info.num_args);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    auto function_info = GetFunctionInfo(op, true /* has_resource_handle */);
    this->PrintFuncCallC(function_info.func_name, function_info.num_args,
                         function_info.resource_handle_name);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else if (auto call_op = op->op.as<Op>();
             call_op.defined() && op_attr_global_symbol_.count(call_op.value())) {
    CodeGenC::VisitExpr_(op, os);
    Array<Type> arg_types;
    for (const PrimExpr& arg : op->args) {
      arg_types.push_back(GetType(arg));
    }
    this->GenerateForwardFunctionDeclarations(
        op_attr_global_symbol_[call_op.value()], arg_types, GetTypeFromRuntimeDataType(op->dtype));
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCHost::VisitStmt_(const AssertStmtNode* op) {  // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
    stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenCHost::VisitStmt_(const AllocateConstNode* op) {  // NOLINT(*)
  std::string symbol_name = AllocVarID(op->buffer_var.get());

  int64_t num_elements = 1;
  const auto& data = op->data.value();
  for (int64_t dim : data.Shape()) {
    num_elements *= dim;
  }

  // CodeGenC places constants in an ELF-specific section.  The host C
  // backend is also compiled as C++ on macOS, where that section spelling is
  // rejected by Mach-O.  Keep the same module-local static constant and
  // alignment contract, but use only the portable compiler alignment
  // attribute.
  decl_stream << "\nstatic const ";
  PrintType(data.DataType(), decl_stream);
  decl_stream << " __attribute__((aligned(" << constants_byte_alignment_->value << "))) "
              << symbol_name << "[" << num_elements << "] = {\n";
  NDArrayDataToC(data, 4, decl_stream);
  decl_stream << "};\n";

  var_idmap_[op->buffer_var.operator->()] = symbol_name;
  this->PrintStmt(op->body);
}

void CodeGenCHost::VisitStmt_(const AttrStmtNode* op) {  // NOLINT(*)
  if (op->attr_key == tir::attr::coproc_uop_scope) {
    PrintStaticInit(op);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCHost::PrintStaticInit(const AttrStmtNode* op) {
  const auto* initializer = op->value.as<StringImmNode>();
  ICHECK(initializer != nullptr)
      << "CodeGenCHost: coproc_uop_scope requires a StringImm initializer";

  Array<Var> captured = tir::UndefinedVars(op->body, {});
  ICHECK(captured.empty())
      << "CodeGenCHost: coproc_uop_scope only supports capture-free bodies; "
      << "captured variables are not supported (coproc_uop_scope initializer '"
      << initializer->value << "')";

  std::string handle_name = static_init_name_supply_->FreshName("__tvm_static_handle");
  std::string callback_name = static_init_name_supply_->FreshName("__tvm_static_init_lambda");

  if (static_init_declarations_sink_->insert(initializer->value).second) {
    (*static_init_declaration_stream_sink_)
        << "#ifdef __cplusplus\n"
        << "extern \"C\"\n"
        << "#endif\n"
        << "int32_t " << initializer->value
        << "(void**, int32_t (*)(void*), void*, int32_t);\n";
  }
  (*static_init_declaration_stream_sink_) << "static void* " << handle_name << " = NULL;\n";

  // A callback is a separate C function.  Generate it with a separate codegen
  // object so all per-function state (SSA bindings, variable ids, scopes,
  // allocation types, and the name supply) is isolated from the enclosing
  // entry function.  Swapping only `stream` is insufficient for LetStmt and
  // Allocate bodies because those visitors mutate the other state as well.
  CodeGenCHost callback_codegen;
  callback_codegen.Init(output_ssa_, emit_asserts_, /*emit_fwd_func_decl=*/true, target_str_,
                        devices_);
  callback_codegen.static_init_name_supply_ = static_init_name_supply_;
  callback_codegen.static_init_stream_sink_ = static_init_stream_sink_;
  callback_codegen.static_init_declarations_sink_ = static_init_declarations_sink_;
  callback_codegen.static_init_declaration_stream_sink_ = static_init_declaration_stream_sink_;
  callback_codegen.InitFuncState(tir::PrimFunc({}, op->body));
  int callback_scope = callback_codegen.BeginScope();
  callback_codegen.PrintStmt(op->body);
  callback_codegen.EndScope(callback_scope);

  std::ostringstream callback_body;
  callback_body << callback_codegen.stream.str();
  // Calls emitted in the isolated context may require declarations (for
  // example VTAUopLoopBegin).  They belong before the static definitions.
  fwd_decl_stream << callback_codegen.fwd_decl_stream.str();

  (*static_init_stream_sink_) << "static int32_t " << callback_name
                              << "(void* closure_data) {\n"
                              << callback_body.str();
  (*static_init_stream_sink_) << "  (void)closure_data;\n"
      << "  return 0;\n"
      << "}\n";

  PrintIndent();
  stream << "if (" << initializer->value << "(&" << handle_name << ", " << callback_name
         << ", NULL, 0) != 0) {\n";
  int init_scope = this->BeginScope();
  PrintIndent();
  stream << "return -1;\n";
  this->EndScope(init_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenCHost::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T* op, const char* compare,
                                               std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

runtime::Module BuildCHost(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts") != nullptr) {
    Map<GlobalVar, String> device_contexts =
        mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
    for (auto const& context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.SetConstantsByteAlignment(target->GetAttr<Integer>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](const PrimFunc& func) -> bool {
    return func->GetAttr<Bool>("runner_function", Bool(false)).value();
  };

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodegenCHost: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    funcs.push_back({gvar, prim_func});
  }

  // Sort functions
  auto sort_key = [&is_aot_executor_fn](const auto& kv) {
    return std::tuple{is_aot_executor_fn(kv.second), kv.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(), [&sort_key](const auto& kv_a, const auto& kv_b) {
    return sort_key(kv_a) < sort_key(kv_b);
  });

  // Declare all functions first.  This ensures that all functions,
  // including the __tvm_main__ used in AOT, have access to forward
  // declarations of other functions in the IRModule.
  for (const auto& [gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }

  // Codegen all functions.  Passing emit_fwd_func_decl=true adds a
  // forward declaration for any `builtin::call_extern`, based on the
  // arguments provided to it.
  for (const auto& [gvar, prim_func] : funcs) {
    cg.AddFunction(gvar, prim_func, emit_fwd_func_decl);
  }

  // NOTE: it's possible that kRuntime attr is not attached when the mod was built with tvm.build().
  // See issue #10373.
  auto opt_runtime = mod->GetAttr<relay::Runtime>(tvm::attr::kRuntime);
  relay::Runtime runtime;
  if (opt_runtime.get() != nullptr) {
    runtime = opt_runtime.value();
  } else {
    runtime = relay::Runtime::Create("cpp", {});
  }

  bool has_aot_executor_fn = std::any_of(
      funcs.begin(), funcs.end(), [&](const auto& kv) { return is_aot_executor_fn(kv.second); });
  if (runtime->name == relay::kTvmRuntimeCpp) {
    cg.InitGlobalContext(/*weak=*/!has_aot_executor_fn);
  }

  if (target->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_REGISTER_GLOBAL("target.build.c").set_body_typed(BuildCHost);
}  // namespace codegen
}  // namespace tvm
