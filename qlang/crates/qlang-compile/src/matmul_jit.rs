//! LLVM JIT code generation for matrix multiplication.
//!
//! Generates a triple-loop matmul:
//!   for i in 0..M:
//!     for j in 0..N:
//!       sum = 0
//!       for k in 0..K:
//!         sum += A[i*K+k] * B[k*N+j]
//!       C[i*N+j] = sum

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::OptimizationLevel;

use crate::codegen::CodegenError;

/// JIT-compiled matrix multiplication.
pub struct CompiledMatMul<'ctx> {
    _context: &'ctx Context,
    _module: inkwell::module::Module<'ctx>,
    execution_engine: inkwell::execution_engine::ExecutionEngine<'ctx>,
    pub llvm_ir: String,
}

type MatMulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64, u64, u64);

/// Helper to build a for-loop in LLVM IR
fn build_for<'ctx, F>(
    context: &'ctx Context,
    builder: &inkwell::builder::Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    name: &str,
    start: inkwell::values::IntValue<'ctx>,
    end: inkwell::values::IntValue<'ctx>,
    step: inkwell::values::IntValue<'ctx>,
    mut body: F,
) where
    F: FnMut(inkwell::values::IntValue<'ctx>),
{
    let pre_header = builder.get_insert_block().unwrap();
    let loop_header = context.append_basic_block(function, &format!("{}_header", name));
    let loop_body = context.append_basic_block(function, &format!("{}_body", name));
    let loop_exit = context.append_basic_block(function, &format!("{}_exit", name));

    builder.build_unconditional_branch(loop_header).unwrap();

    // Header
    builder.position_at_end(loop_header);
    let phi = builder.build_phi(start.get_type(), name).unwrap();
    phi.add_incoming(&[(&start, pre_header)]);
    let i = phi.as_basic_value().into_int_value();
    
    let cond = builder.build_int_compare(inkwell::IntPredicate::ULT, i, end, &format!("{}_cond", name)).unwrap();
    builder.build_conditional_branch(cond, loop_body, loop_exit).unwrap();

    // Body
    builder.position_at_end(loop_body);
    body(i);

    // Increment
    let current_body_end = builder.get_insert_block().unwrap();
    let next_i = builder.build_int_add(i, step, &format!("{}_next", name)).unwrap();
    phi.add_incoming(&[(&next_i, current_body_end)]);
    builder.build_unconditional_branch(loop_header).unwrap();

    // Exit
    builder.position_at_end(loop_exit);
}

/// Compile a matrix multiplication kernel via LLVM.
pub fn compile_matmul(context: &Context) -> Result<CompiledMatMul<'_>, CodegenError> {
    let module = context.create_module("qlang_matmul");
    let builder = context.create_builder();

    let f32_type = context.f32_type();
    let f32_ptr = context.ptr_type(inkwell::AddressSpace::default());
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    let fn_type = void_type.fn_type(
        &[f32_ptr.into(), f32_ptr.into(), f32_ptr.into(),
          i64_type.into(), i64_type.into(), i64_type.into()],
        false,
    );

    let function = module.add_function("qlang_matmul", fn_type, None);

    let is_aarch64 = cfg!(target_arch = "aarch64");
    let vec_len = if is_aarch64 { 4 } else { 8 };
    let vec_len_u64 = vec_len as u64;

    if is_aarch64 {
        function.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            context.create_string_attribute("target-features", "+neon"),
        );
    } else {
        function.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            context.create_string_attribute("target-features", "+avx2,+fma"),
        );
    }

    let vec_f32_type = f32_type.vec_type(vec_len);
    let zero_i64 = i64_type.const_int(0, false);
    let one_i64 = i64_type.const_int(1, false);
    let vec_step = i64_type.const_int(vec_len_u64, false);
    let tile_size = i64_type.const_int(32, false);

    // Blocks
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let m_param = function.get_nth_param(3).unwrap().into_int_value();
    let k_param = function.get_nth_param(4).unwrap().into_int_value();
    let n_param = function.get_nth_param(5).unwrap().into_int_value();

    // i0 loop
    build_for(context, &builder, function, "i0", zero_i64, m_param, tile_size, |i0| {
        // k0 loop
        build_for(context, &builder, function, "k0", zero_i64, k_param, tile_size, |k0| {
            // j0 loop
            build_for(context, &builder, function, "j0", zero_i64, n_param, tile_size, |j0| {
                let i0_plus = builder.build_int_add(i0, tile_size, "").unwrap();
                let c_i = builder.build_int_compare(inkwell::IntPredicate::ULT, i0_plus, m_param, "").unwrap();
                let i_end = builder.build_select(c_i, i0_plus, m_param, "").unwrap().into_int_value();

                build_for(context, &builder, function, "i", i0, i_end, one_i64, |i| {
                    let k0_plus = builder.build_int_add(k0, tile_size, "").unwrap();
                    let c_k = builder.build_int_compare(inkwell::IntPredicate::ULT, k0_plus, k_param, "").unwrap();
                    let k_end = builder.build_select(c_k, k0_plus, k_param, "").unwrap().into_int_value();

                    build_for(context, &builder, function, "k", k0, k_end, one_i64, |k| {
                        let ik = builder.build_int_mul(i, k_param, "").unwrap();
                        let a_idx = builder.build_int_add(ik, k, "").unwrap();
                        let a_gep = unsafe { builder.build_gep(f32_type, a_ptr, &[a_idx], "").unwrap() };
                        let a_val = builder.build_load(f32_type, a_gep, "av").unwrap().into_float_value();

                        // Splat a_val
                        let mut a_vec = vec_f32_type.get_undef();
                        for v in 0..vec_len_u64 {
                            let idx = context.i32_type().const_int(v, false);
                            a_vec = builder.build_insert_element(a_vec, a_val, idx, "").unwrap();
                        }

                        let j0_plus = builder.build_int_add(j0, tile_size, "").unwrap();
                        let c_j = builder.build_int_compare(inkwell::IntPredicate::ULT, j0_plus, n_param, "").unwrap();
                        let j_end = builder.build_select(c_j, j0_plus, n_param, "").unwrap().into_int_value();

                        let j_len = builder.build_int_sub(j_end, j0, "").unwrap();
                        let j_vecs = builder.build_int_unsigned_div(j_len, vec_step, "").unwrap();
                        let j_vec_len = builder.build_int_mul(j_vecs, vec_step, "").unwrap();
                        let j_vec_end = builder.build_int_add(j0, j_vec_len, "").unwrap();

                        // Vector J loop
                        build_for(context, &builder, function, "j_vec", j0, j_vec_end, vec_step, |j| {
                            let kn = builder.build_int_mul(k, n_param, "").unwrap();
                            let b_idx = builder.build_int_add(kn, j, "").unwrap();
                            let b_gep = unsafe { builder.build_gep(f32_type, b_ptr, &[b_idx], "").unwrap() };
                            let b_vec = builder.build_load(vec_f32_type, b_gep, "bv").unwrap().into_vector_value();

                            let in_val = builder.build_int_mul(i, n_param, "").unwrap();
                            let c_idx = builder.build_int_add(in_val, j, "").unwrap();
                            let c_gep = unsafe { builder.build_gep(f32_type, c_ptr, &[c_idx], "").unwrap() };
                            let c_vec = builder.build_load(vec_f32_type, c_gep, "cv").unwrap().into_vector_value();

                            let prod = builder.build_float_mul(a_vec, b_vec, "").unwrap();
                            // In LLVM, we can use llvm.fma.vNf32 but build_float_add is fine, LLVM optimizes it if +fma is enabled.
                            let new_c = builder.build_float_add(c_vec, prod, "").unwrap();
                            builder.build_store(c_gep, new_c).unwrap();
                        });

                        // Scalar remainder J loop
                        build_for(context, &builder, function, "j_sca", j_vec_end, j_end, one_i64, |j| {
                            let kn = builder.build_int_mul(k, n_param, "").unwrap();
                            let b_idx = builder.build_int_add(kn, j, "").unwrap();
                            let b_gep = unsafe { builder.build_gep(f32_type, b_ptr, &[b_idx], "").unwrap() };
                            let b_val = builder.build_load(f32_type, b_gep, "bvs").unwrap().into_float_value();

                            let in_val = builder.build_int_mul(i, n_param, "").unwrap();
                            let c_idx = builder.build_int_add(in_val, j, "").unwrap();
                            let c_gep = unsafe { builder.build_gep(f32_type, c_ptr, &[c_idx], "").unwrap() };
                            let c_val = builder.build_load(f32_type, c_gep, "cvs").unwrap().into_float_value();

                            let prod = builder.build_float_mul(a_val, b_val, "").unwrap();
                            let new_c = builder.build_float_add(c_val, prod, "").unwrap();
                            builder.build_store(c_gep, new_c).unwrap();
                        });
                    });
                });
            });
        });
    });

    let exit = context.append_basic_block(function, "exit");
    builder.build_unconditional_branch(exit).unwrap();
    builder.position_at_end(exit);
    builder.build_return(None).unwrap();

    let llvm_ir = module.print_to_string().to_string();
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    Ok(CompiledMatMul {
        _context: context,
        _module: module,
        execution_engine,
        llvm_ir,
    })
}

/// Execute JIT-compiled matmul: C[m×n] = A[m×k] × B[k×n]
pub fn execute_matmul(
    compiled: &CompiledMatMul,
    a: &[f32], b: &[f32],
    m: usize, k: usize, n: usize,
) -> Result<Vec<f32>, CodegenError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];

    unsafe {
        let func: JitFunction<MatMulFn> = compiled.execution_engine
            .get_function("qlang_matmul")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        func.call(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as u64, k as u64, n as u64);
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_matmul_2x3_times_3x2() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = execute_matmul(&compiled, &a, &b, 2, 3, 2).unwrap();
        assert_eq!(c, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn jit_matmul_1x1() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();
        let c = execute_matmul(&compiled, &[3.0], &[5.0], 1, 1, 1).unwrap();
        assert_eq!(c, vec![15.0]);
    }

    #[test]
    fn jit_matmul_identity() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = execute_matmul(&compiled, &a, &id, 2, 3, 3).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn jit_matmul_large_correct() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let m = 32;
        let k = 64;
        let n = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect();

        let c = execute_matmul(&compiled, &a, &b, m, k, n).unwrap();

        // Verify against naive
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for p in 0..k {
                    expected += a[i * k + p] * b[p * n + j];
                }
                assert!((c[i * n + j] - expected).abs() < 1e-3,
                    "Mismatch at [{i},{j}]");
            }
        }
    }

    #[test]
    fn jit_matmul_ir_contains_fmul() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();
        assert!(compiled.llvm_ir.contains("qlang_matmul"));
        assert!(compiled.llvm_ir.contains("fmul"));
    }
}
