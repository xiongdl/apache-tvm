# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import tvm.testing
import pytest
import ctypes

from tvm import te
from tvm.contrib import utils
from tvm.script import tir as T, ir as I

import numpy as np


def test_add():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)

    def check_c():
        mhost = tvm.build(s, [A, B, C], "c", name="test_fadd")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd"]
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_add_pipeline():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    AA = te.compute((n,), lambda *i: A(*i), name="A")
    BB = te.compute((n,), lambda *i: B(*i), name="B")
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = te.compute(A.shape, lambda *i: T(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    # FIXME(tvm-team): vector operators are not supported for codegen to C yet
    # s[C].vectorize(xi)

    def check_c():
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.tir.decl_buffer(
            A.shape, A.dtype, elem_offset=te.size_var("Aoffset"), offset_factor=8, name="A"
        )
        binds = {A: Ab}
        # BUILD and invoke the kernel.
        f1 = tvm.lower(s, [A, B, C], name="test_fadd_pipeline")
        mhost = tvm.build(f1, target="c")

        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd_pipeline"]
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_reinterpret():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="int32")
    B = te.compute(
        A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.reinterpret", 2 + A(*i)), name="B"
    )
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_reinterpret")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_reinterpret"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.randint(-(2**30), 2**30, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fadd(a, b)
        tvm.testing.assert_allclose(b.numpy(), (2 + a.numpy()).view("float32"))

    check_c()


def test_ceil():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.ceil", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_ceil")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fceil = m["test_ceil"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fceil(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.ceil(a.numpy()).view("float32")))

    check_c()


def test_floor():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.floor", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_floor")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        ffloor = m["test_floor"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        ffloor(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.floor(a.numpy()).view("float32")))

    check_c()


def test_round():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.round", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_round")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fround = m["test_round"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fround(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_call_packed():
    def fake_func(fname="fake.func"):
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        fake_func1 = tvm.tir.call_packed(fname, A[0])

        ib.emit(fake_func1)
        body = ib.get()
        return A, body

    def check_global_packed_func():
        fname = "fake.func"
        A, body = fake_func(fname)
        func1 = tvm.tir.PrimFunc([A], body).with_attr("global_symbol", "func1")
        B, body = fake_func()
        func2 = tvm.tir.PrimFunc([B], body).with_attr("global_symbol", "func2")
        mod = tvm.IRModule({"fake_func1": func1, "fake_func2": func2})
        fcode = tvm.build(mod, None, "c")
        src = fcode.get_source()

        # there are two locations calling the packed func
        assert src.count(fname) == 2

        suffix = "_packed"
        packed_func_name = fname + suffix
        # func name will be standardized by GetUniqueName and not exists anymore
        assert src.find(packed_func_name) == -1

        packed_func_real_name = "_".join(fname.split(".")) + suffix
        func_declaration = "static void* %s = NULL;" % packed_func_real_name
        # src only has 1 valid declaration
        assert src.count(func_declaration) == 1

    check_global_packed_func()


def test_subroutine_call():
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer(1, dtype="float32")):
            mod.subroutine(A.data)

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 42.0

    built = tvm.build(mod, target="c")

    func_names = list(built["get_func_names"]())
    assert (
        "main" in func_names
    ), "Externally exposed functions should be listed in available functions."
    assert (
        "subroutine" not in func_names
    ), "Internal function should not be listed in available functions."

    source = built.get_source()
    assert (
        source.count("main(void*") == 2
    ), "Expected two occurrences, for forward-declaration and definition"
    assert (
        source.count("subroutine(float*") == 2
    ), "Expected two occurrences, for forward-declaration and definition"
    assert (
        source.count("subroutine(") == 3
    ), "Expected three occurrences, for forward-declaration, definition, and call from main."


def _make_c_module(symbol):
    n = tvm.te.size_var("n")
    A = tvm.tir.decl_buffer((n,), dtype="float32", name="A")
    i = tvm.te.var("i")
    body = tvm.tir.For(
        i,
        0,
        n,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.BufferStore(A, tvm.tir.BufferLoad(A, [i]), [i]),
    )
    func = tvm.tir.PrimFunc([A], body).with_attr("global_symbol", symbol)
    return tvm.build(tvm.IRModule({symbol: func}), target="c")


def test_graph_c_module_context_is_weak_and_reloadable():
    module = _make_c_module("graph_context")
    source = module.get_source()

    assert "TVM_WEAK" in source
    assert "__tvm_module_ctx" in source

    temp = utils.tempdir()
    path_dso = temp.relpath("graph_context.so")
    module.export_library(path_dso)
    loaded = tvm.runtime.load_module(path_dso)
    assert loaded.get_function("graph_context") is not None


def test_aot_c_module_context_remains_strong():
    n = tvm.te.size_var("n")
    A = tvm.tir.decl_buffer((n,), dtype="float32", name="A")
    func = tvm.tir.PrimFunc([A], tvm.tir.Evaluate(0)).with_attr(
        "global_symbol", "aot_context"
    )
    func = func.with_attr("runner_function", True)
    source = tvm.build(tvm.IRModule({"aot_context": func}), target="c").get_source()

    assert "void* __tvm_module_ctx = NULL;" in source
    assert "void* TVM_WEAK __tvm_module_ctx = NULL;" not in source


def _make_allocate_const_module(symbol, dtype, values):
    const_var = tvm.tir.Var(
        "constant", tvm.ir.PointerType(tvm.ir.PrimType(dtype), "global")
    )
    const_buffer = tvm.tir.decl_buffer(
        (len(values),), dtype=dtype, data=const_var, name="constant"
    )
    output = tvm.tir.decl_buffer((len(values),), dtype=dtype, name="output")
    index = tvm.tir.Var("i", "int32")
    copy = tvm.tir.For(
        index,
        0,
        len(values),
        tvm.tir.ForKind.SERIAL,
        tvm.tir.BufferStore(
            output, tvm.tir.BufferLoad(const_buffer, [index]), [index]
        ),
    )
    body = tvm.tir.AllocateConst(
        const_var,
        dtype,
        [len(values)],
        tvm.nd.array(np.asarray(values, dtype=dtype)),
        copy,
        annotations={},
    )
    func = tvm.tir.PrimFunc([output], body).with_attr("global_symbol", symbol)
    return tvm.IRModule({symbol: func})


def test_c_host_allocate_const_is_static_and_portable():
    source = tvm.build(
        _make_allocate_const_module("portable_constant", "int32", [-3, 0, 7, 42]),
        target="c",
    ).get_source()

    assert "static const int32_t" in source
    assert "aligned(16)" in source
    assert ".rodata.tvm" not in source
    assert "TVMBackendAllocWorkspace" not in source

    custom_alignment_source = tvm.build(
        _make_allocate_const_module("portable_constant_aligned", "int32", [1, 2]),
        target="c -constants-byte-alignment=32",
    ).get_source()
    assert "aligned(32)" in custom_alignment_source


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("int8", [-128, -1, 0, 127]),
        ("uint8", [0, 1, 127, 255]),
        ("float32", [-1.5, 0.0, 2.25, 42.5]),
    ],
)
def test_c_host_allocate_const_exports_and_reloads(dtype, values):
    symbol = "portable_constant_" + dtype
    module = tvm.build(_make_allocate_const_module(symbol, dtype, values), target="c")
    temp = utils.tempdir()
    path_dso = temp.relpath(symbol + ".so")
    module.export_library(path_dso)
    loaded = tvm.runtime.load_module(path_dso)
    output = tvm.nd.array(np.zeros(len(values), dtype=dtype), tvm.cpu(0))
    loaded[symbol](output)
    np.testing.assert_equal(output.numpy(), np.asarray(values, dtype=dtype))


def test_c_host_allocate_const_names_are_unique_and_deterministic():
    def make_module():
        output = tvm.tir.decl_buffer((8,), dtype="int32", name="output")
        const_vars = [
            tvm.tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType("int32"), "global"))
            for name in ("constant_a", "constant_b")
        ]
        const_buffers = [
            tvm.tir.decl_buffer((4,), dtype="int32", data=var, name=var.name)
            for var in const_vars
        ]
        copies = []
        for offset, const_buffer in ((0, const_buffers[0]), (4, const_buffers[1])):
            index = tvm.tir.Var("i", "int32")
            copies.append(
                tvm.tir.For(
                    index,
                    0,
                    4,
                    tvm.tir.ForKind.SERIAL,
                    tvm.tir.BufferStore(
                        output,
                        tvm.tir.BufferLoad(const_buffer, [index]),
                        [index + offset],
                    ),
                )
            )
        body = tvm.tir.AllocateConst(
            const_vars[0],
            "int32",
            [4],
            tvm.nd.array(np.asarray([1, -2, 3, -4], dtype="int32")),
            tvm.tir.AllocateConst(
                const_vars[1],
                "int32",
                [4],
                tvm.nd.array(np.asarray([5, -6, 7, -8], dtype="int32")),
                tvm.tir.SeqStmt(copies),
                annotations={},
            ),
            annotations={},
        )
        return tvm.IRModule(
            {
                "portable_constant_two": tvm.tir.PrimFunc([output], body).with_attr(
                    "global_symbol", "portable_constant_two"
                )
            }
        )

    first = tvm.build(
        make_module(),
        target="c",
    ).get_source()
    second = tvm.build(
        make_module(),
        target="c",
    ).get_source()
    assert first == second
    assert first.count("static const int32_t") == 2
    assert "constant_a[4]" in first
    assert "constant_b[4]" in first
    assert "constant_a)[i]" in first
    assert "constant_b)[i_1]" in first


def _make_source_c_module(symbol):
    source = """
#include "tvm/runtime/c_runtime_api.h"
void* TVM_WEAK __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C" {{
#endif
TVM_DLL int32_t {symbol}(void* args, int* arg_type_ids, int num_args,
                         void* out_ret_value, int* out_ret_tcode,
                         void* resource_handle) {{
  return 0;
}}
#ifdef __cplusplus
}}
#endif
""".format(symbol=symbol)
    return tvm.runtime._ffi_api.CSourceModuleCreate(source, "c", [symbol], None)


def test_multiple_graph_c_modules_link_without_context_collision():
    module = _make_source_c_module("graph_context_a")
    imported = _make_source_c_module("graph_context_b")
    module.import_module(imported)

    temp = utils.tempdir()
    path_dso = temp.relpath("graph_context_multi.so")
    module.export_library(path_dso)
    loaded = tvm.runtime.load_module(path_dso)
    assert loaded.get_function("graph_context_a") is not None
    assert loaded.get_function("graph_context_b", query_imports=True) is not None


def test_tglobal_symbol_call_has_c_host_forward_declaration():
    tvm.ir.register_op_attr("tir.test_tglobal_symbol", "TGlobalSymbol", "test_tglobal_symbol")
    tvm.ir.register_op_attr(
        "tir.test_tglobal_symbol", "TCallEffectKind", tvm.tir.CallEffectKind.Opaque
    )
    value = tvm.tir.call_intrin("int32", "tir.test_tglobal_symbol", tvm.tir.const(1, "int32"))
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(value)).with_attr("global_symbol", "caller")
    source = tvm.build(tvm.IRModule({"caller": func}), target="c").get_source()

    declaration = "TVM_DLL int32_t test_tglobal_symbol(int32_t);"
    assert declaration in source
    assert source.index(declaration) < source.index("test_tglobal_symbol(1)")


def _make_c_uop_scope_module(initializers):
    ib = tvm.tir.ir_builder.create()
    cp = te.thread_axis((0, 1), "cop")
    for index, initializer in enumerate(initializers):
        ib.scope_attr(cp, "coproc_uop_scope", tvm.tir.StringImm(initializer))
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "VTAUopLoopBegin",
                tvm.tir.const(1, "uint32"),
                tvm.tir.const(0, "uint32"),
                tvm.tir.const(0, "uint32"),
                tvm.tir.const(0, "uint32"),
            )
        )
        ib.emit(tvm.tir.call_extern("int32", "VTAUopLoopEnd"))
    stmt = ib.get()
    func = tvm.tir.PrimFunc([], stmt, tvm.ir.PrimType("int32")).with_attr(
        "global_symbol", "uop_scope"
    )
    return tvm.IRModule({"uop_scope": func})


def _build_c_host_module(mod):
    return tvm.get_global_func("target.build.c")(mod, tvm.target.Target("c"))


def test_c_host_static_uop_scope_generates_callback_and_initializer_call():
    source = _build_c_host_module(_make_c_uop_scope_module(["VTAPushGEMMOp"])).get_source()

    assert "static void* __tvm_static_handle" in source
    assert "static int32_t __tvm_static_init_lambda" in source
    assert "int32_t VTAPushGEMMOp(void**, int32_t (*)(void*), void*, int32_t);" in source
    assert "VTAPushGEMMOp(&__tvm_static_handle" in source
    assert "VTAUopLoopBegin" in source[source.index("__tvm_static_init_lambda") :]
    entry_start = source.index("TVM_DLL int32_t uop_scope() {")
    entry = source[entry_start:]
    assert entry.count("VTAUopLoopBegin") == 0
    assert "return -1" in entry


def test_c_host_static_uop_scopes_have_unique_callbacks_and_deduplicated_declaration():
    source = _build_c_host_module(
        _make_c_uop_scope_module(["VTAPushGEMMOp", "VTAPushGEMMOp", "VTAPushALUOp"])
    ).get_source()

    assert source.count("int32_t VTAPushGEMMOp(void**, int32_t (*)(void*), void*, int32_t);") == 1
    assert source.count("int32_t VTAPushALUOp(void**, int32_t (*)(void*), void*, int32_t);") == 1
    assert source.count("static void* __tvm_static_handle") == 3
    assert source.count("static int32_t __tvm_static_init_lambda") == 3
    assert source.count("VTAPushGEMMOp(&__tvm_static_handle") == 2
    assert source.count("VTAPushALUOp(&__tvm_static_handle") == 1


def test_c_host_static_uop_scope_rejects_captured_variables():
    n = tvm.tir.Var("n", "int32")
    ib = tvm.tir.ir_builder.create()
    cp = te.thread_axis((0, 1), "cop")
    ib.scope_attr(cp, "coproc_uop_scope", tvm.tir.StringImm("VTAPushGEMMOp"))
    ib.emit(tvm.tir.Evaluate(n))
    func = tvm.tir.PrimFunc([n], ib.get()).with_attr("global_symbol", "captured_scope")

    with pytest.raises(tvm.error.TVMError, match="coproc_uop_scope"):
        _build_c_host_module(tvm.IRModule({"captured_scope": func}))


def test_c_host_static_uop_scope_rejects_non_int32_entry():
    ib = tvm.tir.ir_builder.create()
    cp = te.thread_axis((0, 1), "cop")
    ib.scope_attr(cp, "coproc_uop_scope", tvm.tir.StringImm("VTAPushGEMMOp"))
    ib.emit(tvm.tir.Evaluate(tvm.tir.const(0, "int32")))
    stmt = ib.get()
    for name, func in [
        ("non_int32", tvm.tir.PrimFunc([], stmt, tvm.ir.PrimType("int64"))),
        ("void", tvm.tir.PrimFunc([], stmt)),
    ]:
        func = func.with_attr("global_symbol", name)
        with pytest.raises(tvm.error.TVMError, match="int32-returning function"):
            _build_c_host_module(tvm.IRModule({name: func}))


def _make_c_uop_scope_state_module():
    cp = te.thread_axis((0, 1), "cop")
    local = tvm.tir.Var("local", "int32")
    tmp = tvm.tir.Var("tmp", tvm.ir.PointerType(tvm.ir.PrimType("int32"), "global"))
    callback_body = tvm.tir.LetStmt(
        local,
        tvm.tir.const(7, "int32"),
        tvm.tir.Allocate(
            tmp,
            tvm.DataType("int32"),
            [tvm.tir.const(4, "int32")],
            tvm.tir.const(True, "bool"),
            tvm.tir.Evaluate(
                tvm.tir.call_extern("int32", "VTAUopLoopBegin", local, tvm.tir.const(0, "uint32"),
                                    tvm.tir.const(0, "uint32"), tvm.tir.const(0, "uint32"))
            ),
        ),
    )
    scoped_body = tvm.tir.AttrStmt(
        cp,
        "coproc_uop_scope",
        tvm.tir.StringImm("VTAPushGEMMOp"),
        callback_body,
    )
    # This statement is deliberately outside the static scope.  It must stay
    # in the entry function and must not inherit the callback's local state.
    body = tvm.tir.SeqStmt(
        [scoped_body, tvm.tir.Evaluate(tvm.tir.call_extern("int32", "VTAUopLoopEnd"))]
    )
    func = tvm.tir.PrimFunc([], body, tvm.ir.PrimType("int32")).with_attr(
        "global_symbol", "state_scope"
    )
    return tvm.IRModule({"state_scope": func})


def test_c_host_static_uop_scope_isolated_state_is_deterministic_and_compilable():
    source = _build_c_host_module(_make_c_uop_scope_state_module()).get_source()
    source_again = _build_c_host_module(_make_c_uop_scope_state_module()).get_source()
    assert source == source_again
    callback_start = source.index("static int32_t __tvm_static_init_lambda")
    entry_start = source.index("TVM_DLL int32_t state_scope() {")
    callback = source[callback_start:entry_start]
    entry = source[entry_start:]
    assert "int32_t local" in callback
    assert "int32_t tmp" in callback
    assert "VTAUopLoopEnd" in entry
    assert entry.count("VTAUopLoopBegin") == 0


def _make_c_uop_scope_support_module():
    source = r'''
#include "tvm/runtime/c_runtime_api.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
static int32_t init_calls = 0;
static int32_t body_calls = 0;
int32_t uop_scope(void);
int32_t VTAUopLoopBegin(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  (void)a; (void)b; (void)c; (void)d; body_calls++;
  return body_calls == 1 ? 0 : -9;
}
int32_t VTAUopLoopEnd(void) { return 0; }
int32_t VTAPushGEMMOp(void** handle, int32_t (*callback)(void*), void* data, int32_t x) {
  (void)handle; (void)data; (void)x; init_calls++;
  if (init_calls > 1) return -7;
  return callback(NULL);
}
TVM_DLL int32_t vta_test_invoke(TVMValue* args, int* arg_type_ids, int num_args,
                                 TVMValue* out_ret_value, int* out_ret_tcode,
                                 void* resource_handle) {
  (void)args; (void)arg_type_ids; (void)num_args; (void)out_ret_value;
  (void)out_ret_tcode; (void)resource_handle;
  return uop_scope();
}
#ifdef __cplusplus
}
#endif
'''
    return tvm.runtime._ffi_api.CSourceModuleCreate(
        source,
        "c",
        ["vta_test_invoke"],
        None,
    )


def test_c_host_static_uop_scope_compiles_reloads_and_propagates_initializer_failure():
    generated = _build_c_host_module(_make_c_uop_scope_module(["VTAPushGEMMOp"]))
    support = _make_c_uop_scope_support_module()
    module = tvm.runtime._ffi_api.CSourceModuleCreate(
        generated.get_source() + support.get_source(),
        "c",
        ["vta_test_invoke"],
        None,
    )
    temp = utils.tempdir()
    path_dso = temp.relpath("static_uop_behavior.so")
    module.export_library(path_dso)
    # Call the exported packed ABI wrapper through the reloaded shared
    # library.  This avoids relying on CSourceModule's single-symbol packed
    # lookup while still exercising the exact exported ABI.
    library = ctypes.CDLL(path_dso)
    entry = library.vta_test_invoke
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]

    # The support initializer invokes the generated callback on the first
    # call, and deliberately returns a nonzero error before invoking it again.
    # Thus a successful first call proves the callback path is linked, while
    # the second call proves the packed entry propagates the initializer error.
    assert entry(None, None, 0, None, None, None) == 0
    assert entry(None, None, 0, None, None, None) == -1


if __name__ == "__main__":
    tvm.testing.main()
