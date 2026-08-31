// Copyright (c) 2026 by Alibaba.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use log::debug;
#[cfg(feature = "regorus-regovm")]
use regorus::languages::rego::compiler::Compiler;
// The legacy interpreter backend exposes host functions through regorus's
// sync `Extension` trait; the RVM backend drives them through the suspendable
// host-call loop. Only the one compiled for the active backend is needed.
#[cfg(feature = "regorus-regovm")]
use regorus::rvm::vm::{ExecutionMode, ExecutionState, SuspendReason};
#[cfg(feature = "regorus-interpreter")]
use regorus::Extension;
#[cfg(feature = "regorus-regovm")]
use regorus::Rc;
use sha2::Digest;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(all(
    feature = "policy-rvps",
    not(all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    ))
))]
use std::time::Duration;

use crate::rvps::ReferenceValueResolver;

use super::{EvaluationResult, PolicyError};

// Exactly one policy execution backend must be enabled. The two are mutually
// exclusive and together exhaustive: `regorus-interpreter` is the stable legacy
// path (sync `Engine` + `Extension`s via `spawn_blocking`); `regorus-regovm`
// is the unstable Regorus VM suspendable host-call path. The public interface
// (`ExtensionFunction` / `with_extra_extension_functions`) is identical under
// either, so downstream code is unaffected by the choice.
#[cfg(all(feature = "regorus-regovm", feature = "regorus-interpreter"))]
compile_error!(
    "features `regorus-regovm` and `regorus-interpreter` are mutually exclusive; enable exactly one"
);
#[cfg(not(any(feature = "regorus-regovm", feature = "regorus-interpreter")))]
compile_error!("exactly one of `regorus-regovm` / `regorus-interpreter` must be enabled");
// The legacy interpreter backend relies on a multi-threaded tokio runtime
// (`Handle::current` + `spawn_blocking` + `block_on`) and cannot run on the
// single-threaded wasm32 target; there only `regorus-regovm` is available.
#[cfg(all(
    feature = "regorus-interpreter",
    target_arch = "wasm32",
    target_vendor = "unknown",
    target_os = "unknown"
))]
compile_error!("`regorus-interpreter` backend is unavailable on wasm32; use `regorus-regovm`");

#[cfg(feature = "fs")]
mod fs;
mod in_memory;

#[cfg(feature = "fs")]
pub use fs::OPA;
pub use in_memory::OPAInMemory;

#[cfg(all(
    feature = "policy-rvps",
    not(test),
    not(all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    ))
))]
const REFERENCE_QUERY_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(all(
    feature = "policy-rvps",
    test,
    not(all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    ))
))]
const REFERENCE_QUERY_TIMEOUT: Duration = Duration::from_millis(100);

fn is_valid_policy_id(policy_id: &str) -> bool {
    policy_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

fn policy_uses_legacy_reference(policy: &str) -> Result<bool, PolicyError> {
    use regorus::unstable::{Lexer, Source, TokenKind};

    let source = Source::from_contents("policy.rego".to_string(), policy.to_string())
        .map_err(PolicyError::LoadPolicyFailed)?;
    let mut lexer = Lexer::new(&source);
    let mut tokens = Vec::new();

    loop {
        let token = lexer.next_token().map_err(PolicyError::LoadPolicyFailed)?;
        if token.0 == TokenKind::Eof {
            break;
        }
        tokens.push((token.0, token.1.text().to_string()));
    }

    let dotted = tokens.windows(3).any(|tokens| {
        tokens[0].0 == TokenKind::Ident
            && tokens[0].1 == "data"
            && tokens[1].0 == TokenKind::Symbol
            && tokens[1].1 == "."
            && tokens[2].0 == TokenKind::Ident
            && tokens[2].1 == "reference"
    });
    let indexed = tokens.windows(4).any(|tokens| {
        tokens[0].0 == TokenKind::Ident
            && tokens[0].1 == "data"
            && tokens[1].0 == TokenKind::Symbol
            && tokens[1].1 == "["
            && matches!(tokens[2].0, TokenKind::String | TokenKind::RawString)
            && tokens[2].1 == "reference"
            && tokens[3].0 == TokenKind::Symbol
            && tokens[3].1 == "]"
    });

    Ok(dotted || indexed)
}

/// A host-supplied function callable from rego policy under either execution
/// backend. The caller provides `Vec<(String, ExtensionFunction)>` via
/// [`OPAInMemory::with_extra_extension_functions`](in_memory::OPAInMemory::with_extra_extension_functions);
/// each pair registers a rego function named after the key that policy can call
/// to perform work regorus does not ship built-in (e.g. an RVPS lookup, an
/// artifact-server resolve, or any downstream host function). The same async
/// closure type serves both backends:
/// - `regorus-regovm`: the VM suspends on `__builtin_host_await` and the host
///   resumes it with the closure's result;
/// - `regorus-interpreter`: the closure is wrapped in a sync regorus `Extension`
///   that `block_on`s it on a blocking thread.
// On wasm32 the RVPS resolver returns `?Send` futures (single-threaded async,
// matching `RvpsApi`'s `async_trait(?Send)` cfg), so the closure and its
// future drop the `Send` bound. Everywhere else `Send` is required because
// the multi-threaded tokio runtime (interpreter) or the VM (regovm) needs it.
#[cfg(all(
    target_arch = "wasm32",
    target_vendor = "unknown",
    target_os = "unknown"
))]
pub type ExtensionFunction = Arc<
    dyn Fn(
            regorus::Value,
        )
            -> Pin<Box<dyn std::future::Future<Output = Result<regorus::Value, PolicyError>>>>
        + Send
        + Sync,
>;

#[cfg(not(all(
    target_arch = "wasm32",
    target_vendor = "unknown",
    target_os = "unknown"
)))]
pub type ExtensionFunction = Arc<
    dyn Fn(
            /* argument */ regorus::Value,
        )
            -> Pin<Box<dyn std::future::Future<Output = Result<regorus::Value, PolicyError>> + Send>>
        + Send
        + Sync,
>;

/// Compiled RVM programs for one policy, keyed by rule name. Only rules the
/// policy actually defines are present (undefined rules are skipped at compile
/// time, preserving origin/main's `not a valid rule path` -> skip contract).
type RulePrograms = HashMap<String, Arc<regorus::rvm::Program>>;

/// Cross-evaluation program cache: `policy_hash` -> per-rule `Arc<Program>`s.
/// Keyed by content hash so a changed policy (different hash) naturally misses
/// and recompiles; `set_policy`/`delete_policy` clear it to bound memory under
/// policy churn. The cached `Program` excludes per-eval `data`/`input` (those
/// are set on the VM at run time) and the host-await wrapper is constant per
/// engine instance, so a `policy_hash` key is sufficient.
pub type ProgramCache = tokio::sync::RwLock<HashMap<String, RulePrograms>>;

#[cfg(feature = "policy-rvps")]
fn query_reference_value_extension(
    reference_value_resolver: Arc<ReferenceValueResolver>,
) -> ExtensionFunction {
    Arc::new(move |argument| {
        let reference_value_resolver = reference_value_resolver.clone();
        Box::pin(async move {
            let key = argument
                .as_string()
                .map_err(|e| {
                    PolicyError::EvalPolicyFailed(anyhow!(
                        "query_reference_value arg not a string: {e}"
                    ))
                })?
                .to_string();
            let fut = reference_value_resolver.query_reference_value(&key);
            let value = {
                #[cfg(not(all(
                    target_arch = "wasm32",
                    target_vendor = "unknown",
                    target_os = "unknown"
                )))]
                {
                    let timeout = REFERENCE_QUERY_TIMEOUT;
                    tokio::time::timeout(timeout, fut).await.map_err(|_| {
                        PolicyError::EvalPolicyFailed(anyhow!(
                            "query_reference_value({key:?}) timed out after {timeout:?}"
                        ))
                    })?
                }
                #[cfg(all(
                    target_arch = "wasm32",
                    target_vendor = "unknown",
                    target_os = "unknown"
                ))]
                {
                    // tokio::time::timeout in WASM is not supported
                    fut.await
                }
            }
            .map_err(|e| {
                PolicyError::EvalPolicyFailed(anyhow!("query_reference_value({key:?}) failed: {e}"))
            })?;
            Ok(match value {
                Some(v) => regorus::Value::from(v),
                None => regorus::Value::Null,
            })
        })
    })
}

#[cfg(feature = "policy-artifact-server")]
fn query_artifact_server_extension(
    artifact_server_client: Arc<artifact_resolve_sdk::Client>,
) -> ExtensionFunction {
    Arc::new(move |argument| {
        let artifact_server_client = artifact_server_client.clone();
        Box::pin(async move {
            use artifact_resolve_sdk::{Measurement, ReleaseManifest};
            let slices = argument.as_object().map_err(|e| {
                PolicyError::EvalPolicyFailed(anyhow!(
                    "query_artifact_server argument must be an object: {e}"
                ))
            })?;

            debug!("query artifact value from artifact server: {slices:?}");

            let measurements = slices
                .iter()
                .map(|(key, value)| -> Result<Measurement> {
                    use anyhow::Context as _;

                    let key = key.as_string().context("key is not a string")?.to_string();
                    let value = value
                        .as_string()
                        .context("value is not a string")?
                        .to_string();
                    Ok(Measurement::text(key, value))
                })
                .collect::<Result<Vec<Measurement>>>()
                .map_err(PolicyError::EvalPolicyFailed)?;
            let resolve_request =
                artifact_resolve_sdk::ResolveRequest::new(ReleaseManifest::new(measurements));
            match artifact_server_client.resolve(&resolve_request).await {
                Ok(resp) => {
                    if resp.status != "resolved" {
                        Err(PolicyError::EvalPolicyFailed(anyhow!(
                            "query_artifact_server returned unexpected status {:?}",
                            resp.status
                        )))
                    } else {
                        Ok(regorus::Value::Bool(true))
                    }
                }
                Err(err) if err.is_measurement_not_found() || err.is_measurement_revoked() => {
                    debug!("query_artifact_server denied: {err}");
                    Ok(regorus::Value::Bool(false))
                }
                Err(err) => Err(PolicyError::EvalPolicyFailed(anyhow!(
                    "query_artifact_server failed: {err}"
                ))),
            }
        })
    })
}

// Eight parameters, each a distinct concern (policy source, input, id, rule
// list, RVPS resolver, optional artifact client, caller-injected extension
// functions, cross-evaluation program cache); bundling would obscure the
// call sites rather than clarify them.
#[allow(clippy::too_many_arguments)]
async fn common_evaluate(
    policy: String,
    input: String,
    policy_id: String,
    evaluation_rules: Vec<String>,
    reference_value_resolver: Arc<ReferenceValueResolver>,
    #[cfg(feature = "policy-artifact-server")] artifact_server_client: Arc<
        artifact_resolve_sdk::Client,
    >,
    extra_extension_functions: Option<Vec<(String, ExtensionFunction)>>,
    program_cache: &ProgramCache,
) -> Result<EvaluationResult, PolicyError> {
    // Legacy policies read reference values from data.reference; fetch them via
    // the resolver. All other policies get an empty data document.
    let data = if policy_uses_legacy_reference(&policy)? {
        let reference_values = reference_value_resolver
            .get_reference_values()
            .await
            .map_err(|e| PolicyError::LoadReferenceDataFailed(e.into()))?;
        serde_json::json!({ "reference": reference_values }).to_string()
    } else {
        "{}".to_string()
    };

    #[allow(unused_mut)]
    let mut extension_functions = HashMap::<String, ExtensionFunction>::default();

    #[cfg(feature = "policy-rvps")]
    {
        extension_functions.insert(
            "query_reference_value".to_string(),
            query_reference_value_extension(reference_value_resolver),
        );
    }

    #[cfg(feature = "policy-artifact-server")]
    {
        extension_functions.insert(
            "query_artifact_server".to_string(),
            query_artifact_server_extension(artifact_server_client),
        );
    }

    // Merge caller-injected functions after the built-ins. This is the generic
    // extension point that lets a downstream crate supply arbitrary host
    // functions callable from rego that regorus does not ship built-in. User
    // functions are inserted last so a key colliding with a built-in is
    // overridden by the caller's explicit choice.
    if let Some(extras) = extra_extension_functions {
        for (key, function) in extras {
            extension_functions.insert(key, function);
        }
    }

    // Dispatch to the selected backend. Exactly one of the two features is on
    // (enforced by the compile_error guards at the top of this file), so only
    // one branch is compiled. The `program_cache` is only read by the RVM
    // backend (it caches compiled `regorus::rvm::Program`s); under the
    // interpreter backend the param is inert, so reference it once to keep the
    // shared signature warning-free without a cfg on the param itself.
    #[cfg(feature = "regorus-interpreter")]
    let _ = program_cache;
    #[cfg(feature = "regorus-regovm")]
    return evaluate_with_regovm(
        policy,
        input,
        policy_id,
        evaluation_rules,
        data,
        extension_functions,
        program_cache,
    )
    .await;
    #[cfg(feature = "regorus-interpreter")]
    return evaluate_with_interpreter(
        policy,
        input,
        policy_id,
        evaluation_rules,
        data,
        extension_functions,
    )
    .await;
}

#[cfg(feature = "regorus-regovm")]
async fn evaluate_with_regovm(
    policy: String,
    input: String,
    policy_id: String,
    evaluation_rules: Vec<String>,
    data: String,
    extension_functions: HashMap<String, ExtensionFunction>,
    program_cache: &ProgramCache,
) -> Result<EvaluationResult, PolicyError> {
    let policy_hash = {
        let mut hasher = sha2::Sha384::new();
        hasher.update(&policy);
        hex::encode(hasher.finalize())
    };

    // Emit the extension wrappers as a *separate* rego.v1 module (see
    // [`build_extensions_module`]) rather than concatenating them onto the
    // user policy. This keeps legacy rego.v0 policies parseable: the user
    // module (no `import rego.v1`) parses in v0 mode while the wrapper module
    // parses in v1 mode. Skip it when no extension functions are registered.
    let wrapper_module = if extension_functions.is_empty() {
        None
    } else {
        Some(build_extensions_module(&extension_functions))
    };

    let data_value =
        regorus::Value::from_json_str(&data).map_err(PolicyError::JsonSerializationFailed)?;
    let input_value =
        regorus::Value::from_json_str(&input).map_err(PolicyError::SetInputDataFailed)?;

    // C (program cache): a hit skips Engine construction, policy parsing and
    // compilation entirely — only the per-rule VM runs. A miss builds the
    // engine once (A: hoisted out of the rule loop) and compiles each
    // requested rule, skipping rules the policy does not define (origin/main
    // `not a valid rule path` -> skip contract, preserved here and in the
    // cached map's absence for that rule).
    let programs: RulePrograms = {
        let cached = program_cache.read().await.get(&policy_hash).cloned();
        if let Some(m) = cached {
            m
        } else {
            let mut engine = regorus::Engine::new();
            engine.set_rego_v0(true);
            engine
                .add_data(data_value.clone())
                .map_err(PolicyError::LoadPolicyFailed)?;
            engine
                .add_policy(policy_id.clone(), policy.clone())
                .map_err(PolicyError::LoadPolicyFailed)?;
            if let Some(wrapper) = &wrapper_module {
                engine
                    .add_policy(EXTENSIONS_WRAPPER_MODULE_ID.to_string(), wrapper.clone())
                    .map_err(PolicyError::LoadPolicyFailed)?;
            }

            let mut compiled: RulePrograms = HashMap::new();
            for rule in &evaluation_rules {
                // regorus rejects a bare rule name with "not a valid rule
                // path"; use the full data.policy path. See [`common_evaluate`].
                let entry_point = format!("data.policy.{rule}");
                let cp = match engine.compile_with_entrypoint(&Rc::from(entry_point.clone())) {
                    Ok(cp) => cp,
                    Err(e) if e.to_string().contains("not a valid rule path") => {
                        debug!("Policy `{policy_id}` does not check {rule}");
                        continue;
                    }
                    Err(e) => return Err(PolicyError::LoadPolicyFailed(e)),
                };
                let program = Compiler::compile_from_policy(&cp, &[entry_point.as_str()])
                    .map_err(|e| PolicyError::LoadPolicyFailed(e.into()))?;
                compiled.insert(rule.clone(), program);
            }
            program_cache
                .write()
                .await
                .insert(policy_hash.clone(), compiled.clone());
            compiled
        }
    };

    let mut rules_result = std::collections::HashMap::new();
    for rule in &evaluation_rules {
        // Rules absent from `programs` were skipped (policy does not define
        // them) — preserve origin/main's skip behavior on cache hits too.
        let Some(program) = programs.get(rule) else {
            continue;
        };

        // Lower the CompiledPolicy to a Program, then load it onto a fresh VM.
        // RegoVM::new_with_policy stores the policy but never loads a program,
        // so execute() returns Undefined — do not use it.
        let mut vm = regorus::rvm::RegoVM::new();
        vm.load_program(program.clone());
        vm.set_data(data_value.clone())
            .map_err(|e| PolicyError::LoadReferenceDataFailed(e.into()))?;
        vm.set_input(input_value.clone());
        vm.set_execution_mode(ExecutionMode::Suspendable);

        let _ = vm
            .execute()
            .map_err(|e| PolicyError::EvalPolicyFailed(e.into()))?;
        let result_value = loop {
            match vm.execution_state().clone() {
                ExecutionState::Suspended {
                    reason:
                        SuspendReason::HostAwait {
                            argument,
                            identifier,
                            ..
                        },
                    ..
                } => {
                    let v = dispatch(identifier, argument, &extension_functions).await?;
                    vm.resume(Some(v))
                        .map_err(|e| PolicyError::EvalPolicyFailed(e.into()))?;
                }
                ExecutionState::Completed { result } => break result,
                ExecutionState::Error { error } => {
                    return Err(PolicyError::EvalPolicyFailed(anyhow!(
                        "RegoVM error: {error}"
                    )))
                }
                other => {
                    return Err(PolicyError::EvalPolicyFailed(anyhow!(
                        "unexpected VM state: {other:?}"
                    )))
                }
            }
        };

        let claim_value = serde_json::from_str(
            &result_value
                .to_json_str()
                .map_err(PolicyError::JsonSerializationFailed)?,
        )
        .map_err(PolicyError::SerdeJsonError)?;
        rules_result.insert(rule.clone(), claim_value);
    }

    Ok(EvaluationResult {
        rules_result,
        policy_hash,
    })
}

// === Legacy interpreter backend ==============================================
// The stable path: the sync regorus `Engine` + `Extension`s, driven via
// `tokio::task::spawn_blocking` so each extension's `block_on` lands on a
// blocking-pool thread (never nesting the tokio runtime). The same async
// `ExtensionFunction` closures the regovm backend drives through its suspend
// loop are reused here unchanged -- [`async_to_sync_extension`] adapts each
// into a sync regorus `Extension` that `block_on`s the closure's future.
// ==========================================================================

#[cfg(feature = "regorus-interpreter")]
async fn evaluate_with_interpreter(
    policy: String,
    input: String,
    policy_id: String,
    evaluation_rules: Vec<String>,
    data: String,
    extension_functions: HashMap<String, ExtensionFunction>,
) -> Result<EvaluationResult, PolicyError> {
    // Captured on the async thread, then moved onto a blocking-pool thread so
    // the extensions' `block_on` never runs inside a runtime context guard.
    let runtime_handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        evaluate_sync(
            policy,
            input,
            policy_id,
            evaluation_rules,
            data,
            extension_functions,
            runtime_handle,
        )
    })
    .await
    .map_err(|e| {
        PolicyError::EvalPolicyFailed(anyhow!("Regorus blocking evaluation task failed: {e}"))
    })?
}

#[cfg(feature = "regorus-interpreter")]
fn evaluate_sync(
    policy: String,
    input: String,
    policy_id: String,
    evaluation_rules: Vec<String>,
    data: String,
    extension_functions: HashMap<String, ExtensionFunction>,
    runtime_handle: tokio::runtime::Handle,
) -> Result<EvaluationResult, PolicyError> {
    let policy_hash = {
        let mut hasher = sha2::Sha384::new();
        hasher.update(&policy);
        hex::encode(hasher.finalize())
    };

    let mut engine = regorus::Engine::new();
    // regorus 0.11 defaults to rego.v1; keep accepting legacy `allow { ... }`
    // (rego.v0) policies saved before the rego.v1 migration. `import rego.v1`
    // policies still work.
    engine.set_rego_v0(true);
    engine
        .add_policy(policy_id.clone(), policy)
        .map_err(PolicyError::LoadPolicyFailed)?;
    let data_value =
        regorus::Value::from_json_str(&data).map_err(PolicyError::JsonSerializationFailed)?;
    engine
        .add_data(data_value)
        .map_err(PolicyError::LoadReferenceDataFailed)?;
    engine
        .set_input_json(&input)
        .map_err(PolicyError::SetInputDataFailed)?;

    for (name, function) in &extension_functions {
        engine
            .add_extension(
                name.clone(),
                1,
                async_to_sync_extension(function.clone(), runtime_handle.clone()),
            )
            .map_err(PolicyError::EvalPolicyFailed)?;
    }

    let mut rules_result = std::collections::HashMap::new();
    for rule in evaluation_rules {
        // regorus rejects a bare rule name with "not a valid rule path"; use
        // the full data.policy path.
        let whole_rule = format!("data.policy.{rule}");
        let claim_value = match engine.eval_rule(whole_rule) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("not a valid rule path") => {
                debug!("Policy `{policy_id}` does not check {rule}");
                continue;
            }
            Err(error) => return Err(PolicyError::EvalPolicyFailed(error)),
        };
        let claim_value = claim_value
            .to_json_str()
            .map_err(PolicyError::JsonSerializationFailed)?;
        let claim_value =
            serde_json::from_str(&claim_value).map_err(PolicyError::SerdeJsonError)?;
        rules_result.insert(rule, claim_value);
    }

    Ok(EvaluationResult {
        rules_result,
        policy_hash,
    })
}

/// Bridge an async [`ExtensionFunction`] into the sync `regorus::Extension`
/// the interpreter backend expects: drive the closure's future to completion
/// on the captured tokio runtime handle. Called from a `spawn_blocking`
/// thread, so `block_on` does not nest the runtime.
#[cfg(feature = "regorus-interpreter")]
fn async_to_sync_extension(
    function: ExtensionFunction,
    runtime_handle: tokio::runtime::Handle,
) -> Box<dyn Extension> {
    Box::new(move |params: Vec<regorus::Value>| {
        if params.len() != 1 {
            return Err(anyhow!(
                "extension expects exactly 1 argument, got {}",
                params.len()
            ));
        }
        let argument = params[0].clone();
        let future = function(argument);
        runtime_handle.block_on(future).map_err(anyhow::Error::new)
    })
}
/// by the identifier the VM passed to `__builtin_host_await`.
#[cfg(feature = "regorus-regovm")]
async fn dispatch(
    identifier: regorus::Value,
    argument: regorus::Value,
    extension_functions: &HashMap<String, ExtensionFunction>,
) -> Result<regorus::Value, PolicyError> {
    let id = identifier.as_string().map_err(|e| {
        PolicyError::EvalPolicyFailed(anyhow!("extension identifier not a string: {e}"))
    })?;

    match extension_functions.get(id.as_ref()) {
        Some(function) => function(argument).await,
        None => Err(PolicyError::EvalPolicyFailed(anyhow!(
            "unknown extension function: {id}"
        ))),
    }
}

/// Per-key rego wrappers appended to the policy source on the regovm backend.
/// Dynamically generates a rego function for each registered extension,
/// forwarding the friendly name onto regorus's native `__builtin_host_await`
/// primitive, which suspends the VM so the host can run async I/O and resume it.
/// Uses `if` + `:=` (rego.v1 syntax) and no `import rego.v1` -- see
/// [`build_extensions_module`], which wraps these definitions in their own
/// rego.v1 module.
#[cfg(feature = "regorus-regovm")]
fn build_extensions(extension_functions: &HashMap<String, ExtensionFunction>) -> String {
    let mut ext = String::from(r#"# === trustee EXTENSIONS (generated) ==="#);
    for key in extension_functions.keys() {
        ext.push_str(&format!(
            r#"
{key}(arg) := v if {{ v := __builtin_host_await(arg, "{key}") }}
"#,
        ));
    }
    ext
}

/// Module id under which the generated extension wrappers are loaded, so they
/// form a distinct module from the user-supplied policy.
#[cfg(feature = "regorus-regovm")]
const EXTENSIONS_WRAPPER_MODULE_ID: &str = "__trustee_extensions__.rego";

/// Wrap the generated extension function definitions in their own rego.v1
/// module (own `package` + `import rego.v1`). Keeping them separate from the
/// user policy lets a legacy rego.v0 policy (no `import rego.v1`) parse in v0
/// mode while the wrappers still parse in v1 mode: the regorus parser picks
/// the dialect per module (auto-enabling v1 when it sees `import rego.v1`),
/// but a single shared module cannot mix dialects, so concatenating v1-syntax
/// wrappers onto a v0 policy would force the whole module into v1 and reject
/// legacy `allow { ... }` bodies.
#[cfg(feature = "regorus-regovm")]
fn build_extensions_module(extension_functions: &HashMap<String, ExtensionFunction>) -> String {
    format!(
        "package policy\nimport rego.v1\n\n{}",
        build_extensions(extension_functions)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "policy-artifact-server")]
    use crate::policy_engine::PolicyEngine;
    #[cfg(feature = "policy-artifact-server")]
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread::JoinHandle,
        time::Duration,
    };

    #[test]
    fn detects_legacy_reference_without_comment_or_string_false_positives() {
        assert!(policy_uses_legacy_reference(
            "package policy\nallow { input.svn in data.reference.svn }"
        )
        .unwrap());
        assert!(policy_uses_legacy_reference(
            "package policy\nallow { input.svn in data[\"reference\"].svn }"
        )
        .unwrap());
        assert!(!policy_uses_legacy_reference(
            r#"package policy
               # data.reference.comment_only
               message := "data.reference.string_only"
               allow := true"#
        )
        .unwrap());
    }

    #[cfg(feature = "policy-artifact-server")]
    fn mock_artifact_server(
        status: u16,
        response_body: serde_json::Value,
    ) -> (String, JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();

            let mut request = Vec::new();
            loop {
                let mut buffer = [0; 4096];
                let bytes_read = stream.read(&mut buffer).unwrap();
                if bytes_read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..bytes_read]);

                if let Some(header_end) =
                    request.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&request[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or_default();
                    if request.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
            }

            let reason = match status {
                200 => "OK",
                404 => "Not Found",
                409 => "Conflict",
                500 => "Internal Server Error",
                _ => "Unknown",
            };
            let response_body = response_body.to_string();
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            request
        });

        (format!("http://{address}"), server)
    }

    #[cfg(feature = "policy-artifact-server")]
    async fn call_artifact_server_extension(base_url: String) -> Result<bool> {
        use anyhow::Context as _;

        let http_client = reqwest::Client::builder().no_proxy().build()?;
        let client = Arc::new(
            artifact_resolve_sdk::Client::builder()
                .base_url(base_url)
                .http_client(http_client)
                .build()?,
        );

        let extension = query_artifact_server_extension(client);
        let argument = regorus::Value::from_json_str(r#"{"tdx.td-shim":"582f8ed2"}"#)?;
        let result = extension(argument).await?;
        Ok(*result
            .as_bool()
            .context("query_artifact_server result must be a boolean")?)
    }

    #[cfg(feature = "policy-artifact-server")]
    fn request_json(request: &[u8]) -> serde_json::Value {
        let body_start = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4;
        serde_json::from_slice(&request[body_start..]).unwrap()
    }

    #[cfg(feature = "policy-artifact-server")]
    #[tokio::test(flavor = "multi_thread")]
    async fn artifact_server_policy_sends_text_measurement_and_returns_true() {
        let (base_url, server) = mock_artifact_server(
            200,
            serde_json::json!({
                "status": "resolved",
                "release_manifest": {
                    "schemaVersion": "1.0.0",
                    "measurements": [{
                        "type": "tdx.td-shim",
                        "value": "582f8ed2"
                    }]
                },
                "log_entries": []
            }),
        );

        let policy = r#"package policy
default allow = false
allow = query_artifact_server({"tdx.td-shim": "582f8ed2"})
"#;
        let engine =
            OPAInMemory::with_raw_default_policy(policy, "artifact.rego", &base_url).unwrap();
        let result = engine
            .evaluate(
                "{}",
                "artifact",
                vec!["allow".to_string()],
                crate::rvps::test_resolver(HashMap::new()),
            )
            .await
            .unwrap();
        assert!(result.rules_result.get("allow").unwrap().as_bool().unwrap());

        let request = server.join().unwrap();
        assert_eq!(
            request_json(&request),
            serde_json::json!({
                "release_manifest": {
                    "schemaVersion": "1.0.0",
                    "measurements": [{
                        "type": "tdx.td-shim",
                        "value": "582f8ed2"
                    }]
                }
            })
        );
    }

    #[cfg(feature = "policy-artifact-server")]
    #[tokio::test(flavor = "multi_thread")]
    async fn artifact_server_missing_or_revoked_measurement_returns_false() {
        for (status, error_code) in [(404, "measurement_not_found"), (409, "measurement_revoked")] {
            let (base_url, server) = mock_artifact_server(
                status,
                serde_json::json!({
                    "error_code": error_code,
                    "error_message": "measurement denied"
                }),
            );

            assert!(!call_artifact_server_extension(base_url).await.unwrap());
            server.join().unwrap();
        }
    }

    #[cfg(feature = "policy-artifact-server")]
    #[tokio::test(flavor = "multi_thread")]
    async fn artifact_server_infrastructure_error_is_propagated() {
        let (base_url, server) = mock_artifact_server(
            500,
            serde_json::json!({
                "error_code": "internal_error",
                "error_message": "server unavailable"
            }),
        );

        let error = call_artifact_server_extension(base_url)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("query_artifact_server failed"));
        assert!(error.contains("internal_error"));
        server.join().unwrap();
    }

    // regorus 0.11 defaults to rego.v1 (`Engine::new()` -> rego_v1 = true),
    // which rejects legacy `allow { ... }` (rego.v0) policies. The production
    // engine creation sites call `engine.set_rego_v0(true)` to keep those
    // pre-migration policies working. The four tests below pin the
    // compatibility behaviour of that mode for each policy shape.

    // Pins the full parse/eval matrix across both engine modes so that a
    // default-mode or v0-mode behavioural shift is caught. The v0-mode
    // columns are also asserted individually by the `rego_v0_mode_*` tests.
    #[test]
    fn rego_v0_v1_parse_compat_matrix() {
        let new_no_import = "package policy\nallow if { true }";
        let new_with_import = "package policy\nimport rego.v1\nallow if { true }";
        let old_no_import = "package policy\nallow { true }";
        let old_with_import = "package policy\nimport rego.v1\nallow { true }";

        let eval_allow = |policy: &str, v0_mode: bool| {
            let mut engine = regorus::Engine::new();
            if v0_mode {
                engine.set_rego_v0(true);
            }
            engine.add_policy("p.rego".to_string(), policy.to_string())?;
            let results = engine.eval_query("data.policy.allow".to_string(), false)?;
            Ok::<_, anyhow::Error>(
                results
                    .result
                    .first()
                    .and_then(|r| r.expressions.first())
                    .and_then(|e| e.value.as_bool().ok().copied())
                    == Some(true),
            )
        };

        // Default (rego.v1) engine: new dialect works with or without the import.
        assert!(eval_allow(new_no_import, false).unwrap());
        assert!(eval_allow(new_with_import, false).unwrap());
        // Legacy `allow { ... }` policies break under the default engine.
        assert!(eval_allow(old_no_import, false).is_err());
        assert!(eval_allow(old_with_import, false).is_err());

        // `set_rego_v0(true)` restores backward compatibility: legacy policies
        // parse again, and the new dialect still works (with or without import).
        assert!(eval_allow(old_no_import, true).unwrap());
        assert!(eval_allow(new_no_import, true).unwrap());
        assert!(eval_allow(new_with_import, true).unwrap());
        // Legacy body shape + `import rego.v1` stays invalid in either mode.
        assert!(eval_allow(old_with_import, true).is_err());
    }

    fn eval_allow_v0_mode(policy: &str) -> Result<bool, anyhow::Error> {
        let mut engine = regorus::Engine::new();
        engine.set_rego_v0(true);
        engine.add_policy("p.rego".to_string(), policy.to_string())?;
        let results = engine.eval_query("data.policy.allow".to_string(), false)?;
        Ok(results
            .result
            .first()
            .and_then(|r| r.expressions.first())
            .and_then(|e| e.value.as_bool().ok().copied())
            == Some(true))
    }

    #[test]
    fn rego_v0_mode_accepts_new_format_without_import() {
        // rego.v1 syntax (`allow if { ... }`) but without `import rego.v1`.
        // Accepted: the `if` keyword is recognised even in rego.v0 mode.
        let policy = "package policy\nallow if { true }";
        assert!(eval_allow_v0_mode(policy).unwrap());
    }

    #[test]
    fn rego_v0_mode_accepts_new_format_with_import() {
        // rego.v1 syntax with an explicit `import rego.v1`.
        let policy = "package policy\nimport rego.v1\nallow if { true }";
        assert!(eval_allow_v0_mode(policy).unwrap());
    }

    #[test]
    fn rego_v0_mode_accepts_legacy_format_without_import() {
        // Legacy rego.v0 `allow { ... }` body (no `if`). This is the
        // backward-compatibility case the v1 default would have broken.
        let policy = "package policy\nallow { true }";
        assert!(eval_allow_v0_mode(policy).unwrap());
    }

    #[test]
    fn rego_v0_mode_rejects_legacy_body_with_v1_import() {
        // Legacy `allow { ... }` body shape combined with `import rego.v1`
        // is self-contradictory: the import turns on rego.v1 for the module,
        // which then requires `if`. This stays a parse error in either mode.
        let policy = "package policy\nimport rego.v1\nallow { true }";
        assert!(eval_allow_v0_mode(policy).is_err());
    }

    #[cfg(all(feature = "policy-rvps", feature = "regorus-regovm"))]
    #[tokio::test]
    async fn dispatch_routes_reference_value_lookup_and_returns_null_when_missing() {
        use crate::rvps::test_resolver;
        let rvps = test_resolver(std::collections::HashMap::from([]));
        let mut functions = HashMap::<String, ExtensionFunction>::new();
        functions.insert(
            "query_reference_value".to_string(),
            query_reference_value_extension(rvps),
        );
        let id = regorus::Value::String(regorus::Rc::from("query_reference_value"));
        let arg = regorus::Value::String(regorus::Rc::from("missing-key"));
        let v = dispatch(id, arg, &functions).await.unwrap();
        assert!(matches!(v, regorus::Value::Null));
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn dispatch_unknown_identifier_errors() {
        let functions = HashMap::<String, ExtensionFunction>::new();
        let id = regorus::Value::String(regorus::Rc::from("nope"));
        let arg = regorus::Value::Null;
        assert!(dispatch(id, arg, &functions).await.is_err());
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn dispatch_passes_argument_to_registered_function() {
        let mut functions = HashMap::<String, ExtensionFunction>::new();
        functions.insert(
            "echo".to_string(),
            Arc::new(|argument| Box::pin(async move { Ok::<_, PolicyError>(argument) })),
        );
        let id = regorus::Value::String(regorus::Rc::from("echo"));
        let arg = regorus::Value::String(regorus::Rc::from("payload"));
        let v = dispatch(id, arg, &functions).await.unwrap();
        match v {
            regorus::Value::String(s) => assert_eq!(s.as_ref(), "payload"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[cfg(feature = "regorus-regovm")]
    #[test]
    fn build_extensions_empty_returns_only_header() {
        let functions = HashMap::<String, ExtensionFunction>::new();
        assert_eq!(
            build_extensions(&functions),
            "# === trustee EXTENSIONS (generated) ==="
        );
    }

    #[cfg(feature = "regorus-regovm")]
    #[test]
    fn build_extensions_generates_wrapper_per_registered_key() {
        let make = || -> ExtensionFunction {
            Arc::new(|_a: regorus::Value| {
                Box::pin(async { Ok::<_, PolicyError>(regorus::Value::Null) })
            })
        };
        let mut functions = HashMap::<String, ExtensionFunction>::new();
        functions.insert("alpha".to_string(), make());
        functions.insert("beta".to_string(), make());
        let ext = build_extensions(&functions);
        assert!(
            ext.contains(r#"alpha(arg) := v if { v := __builtin_host_await(arg, "alpha") }"#),
            "{ext}"
        );
        assert!(
            ext.contains(r#"beta(arg) := v if { v := __builtin_host_await(arg, "beta") }"#),
            "{ext}"
        );
    }
    // A fresh, empty program cache for `evaluate_with_regovm` tests. Each test
    // gets its own so cached programs never leak across cases.
    #[cfg(feature = "regorus-regovm")]
    fn fresh_program_cache() -> ProgramCache {
        ProgramCache::default()
    }

    // Extension closure that always returns a fixed value, ignoring its argument.
    #[cfg(feature = "regorus-regovm")]
    fn fixed_value_extension(value: regorus::Value) -> ExtensionFunction {
        Arc::new(move |_argument| {
            let value = value.clone();
            Box::pin(async move { Ok(value) })
        })
    }

    // Extension closure mapping a string argument to a number ("a"->1, "b"->2),
    // else null. Used to verify the VM forwards the policy argument to the host.
    #[cfg(feature = "regorus-regovm")]
    fn lookup_extension() -> ExtensionFunction {
        Arc::new(move |argument| {
            Box::pin(async move {
                let key = argument.as_string().map_err(|e| {
                    PolicyError::EvalPolicyFailed(anyhow!("lookup arg not a string: {e}"))
                })?;
                Ok(match key.as_ref() {
                    "a" => regorus::Value::from(serde_json::json!(1)),
                    "b" => regorus::Value::from(serde_json::json!(2)),
                    _ => regorus::Value::Null,
                })
            })
        })
    }

    // Extension closure that always fails, to exercise error propagation from a
    // suspended host call back up through evaluate_with_regovm.
    #[cfg(feature = "regorus-regovm")]
    fn failing_extension() -> ExtensionFunction {
        Arc::new(move |_argument| {
            Box::pin(async move {
                Err(PolicyError::EvalPolicyFailed(anyhow!(
                    "async function failed"
                )))
            })
        })
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn evaluate_with_regovm_async_builtin_drives_rule_true() {
        // A policy calls an extension whose returned value satisfies the rule
        // body, so the rule evaluates to true. Exercises the full
        // compile -> suspend -> host resume -> complete loop.
        let mut functions = HashMap::new();
        functions.insert(
            "my_async".to_string(),
            fixed_value_extension(regorus::Value::String(regorus::Rc::from("ok"))),
        );
        let policy = r#"package policy
import rego.v1
allow if {
    my_async("anything") == "ok"
}
"#;
        let result = evaluate_with_regovm(
            policy.to_string(),
            "{}".to_string(),
            "test".to_string(),
            vec!["allow".to_string()],
            "{}".to_string(),
            functions,
            &fresh_program_cache(),
        )
        .await
        .unwrap();
        assert_eq!(
            result.rules_result.get("allow"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn evaluate_with_regovm_async_builtin_receives_policy_argument() {
        // The policy calls the same builtin twice with different arguments and
        // requires both to match. If the VM did not forward the policy argument to
        // the host, the second lookup would not return 2 and allow would be false.
        let mut functions = HashMap::new();
        functions.insert("lookup".to_string(), lookup_extension());
        let policy = r#"package policy
import rego.v1
allow if {
    lookup("a") == 1
    lookup("b") == 2
}
"#;
        let result = evaluate_with_regovm(
            policy.to_string(),
            "{}".to_string(),
            "test".to_string(),
            vec!["allow".to_string()],
            "{}".to_string(),
            functions,
            &fresh_program_cache(),
        )
        .await
        .unwrap();
        assert_eq!(
            result.rules_result.get("allow"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn evaluate_with_regovm_async_builtin_null_satisfies_rule() {
        // A builtin returning null (unknown key) is compared against null in the
        // rule body and satisfies it. Exercises the Null return path end-to-end.
        let mut functions = HashMap::new();
        functions.insert("maybe".to_string(), lookup_extension());
        let policy = r#"package policy
import rego.v1
default allow = false
allow if {
    maybe("unknown") == null
}
"#;
        let result = evaluate_with_regovm(
            policy.to_string(),
            "{}".to_string(),
            "test".to_string(),
            vec!["allow".to_string()],
            "{}".to_string(),
            functions,
            &fresh_program_cache(),
        )
        .await
        .unwrap();
        assert_eq!(
            result.rules_result.get("allow"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[cfg(feature = "regorus-regovm")]
    #[tokio::test]
    async fn evaluate_with_regovm_propagates_async_builtin_error() {
        // A failing extension call must surface as a PolicyError from
        // evaluate_with_regovm, not panic or be silently swallowed.
        let mut functions = HashMap::new();
        functions.insert("failing".to_string(), failing_extension());
        let policy = r#"package policy
import rego.v1
allow if {
    failing("x") == 1
}
"#;
        let err = evaluate_with_regovm(
            policy.to_string(),
            "{}".to_string(),
            "test".to_string(),
            vec!["allow".to_string()],
            "{}".to_string(),
            functions,
            &fresh_program_cache(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("async function failed"), "{err}");
    }

    // Pins origin/main's "skip missing rule" contract: a policy that defines
    // only some of the requested rules must evaluate successfully, with the
    // undefined rules simply ABSENT from the result — not a hard error. Both
    // the old interpreter (`eval_rule` + catch `not a valid rule path`) and the
    // current `evaluate_with_regovm` preserve this; any optimization of
    // `evaluate_with_regovm` (hoist, program cache, etc.) MUST preserve it too.
    // See the `eval_bench` module for why multi-entry compile was rejected: it
    // would turn this skip into a whole-compile failure.
    #[cfg(all(feature = "regorus-regovm", feature = "policy-rvps"))]
    #[tokio::test]
    async fn evaluate_skips_rules_not_defined_in_policy() {
        let policy = r#"package policy
import rego.v1
default executables := 3
default hardware := 2
"#;
        let mut functions = HashMap::<String, ExtensionFunction>::default();
        functions.insert(
            "query_reference_value".to_string(),
            query_reference_value_extension(crate::rvps::test_resolver(
                std::collections::HashMap::new(),
            )),
        );
        let result = evaluate_with_regovm(
            policy.to_string(),
            "{}".to_string(),
            "partial".to_string(),
            vec![
                "executables".to_string(),
                "hardware".to_string(),
                "configuration".to_string(), // not defined -> must be skipped
                "file_system".to_string(),   // not defined -> must be skipped
            ],
            "{}".to_string(),
            functions,
            &fresh_program_cache(),
        )
        .await
        .expect("partial policy must evaluate, skipping undefined rules");

        let present: std::collections::HashSet<&str> =
            result.rules_result.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            present,
            ["executables", "hardware"]
                .into_iter()
                .collect::<std::collections::HashSet<&str>>(),
            "only defined rules should be present; undefined ones skipped"
        );
        assert_eq!(
            result.rules_result.get("executables").unwrap(),
            &serde_json::json!(3)
        );
        assert_eq!(
            result.rules_result.get("hardware").unwrap(),
            &serde_json::json!(2)
        );
    }
}

// ============================================================================
// Evaluation micro-benchmark.
//
// Reproduces the code-review finding that the RegoVM path re-creates an
// `Engine`, re-loads the same policy/data, and re-compiles an RVM program once
// per trust-vector rule inside the `evaluation_rules` loop. A default EAR
// appraisal evaluates 4 rules (`executables`, `hardware`, `configuration`,
// `file_system`), so the per-rule compile cost is amplified 4× per appraisal.
//
// Strategies timed back-to-back in one process on identical inputs:
//
//   * `interpreter_baseline` — a faithful in-bench reconstruction of the
//     origin/main `evaluate_sync` path (build `Engine` once, load policy/data/
//     input once, loop rules via `engine.eval_rule`). This is the traditional
//     regorus interpreter the reviewer used as the baseline (~0.67s / 20 runs).
//
//   * `regovm_current` — the production `evaluate_with_regovm` (per-rule
//     `Engine` + policy/data load + `compile_with_entrypoint` + RVM program
//     compile + fresh `RegoVM`). This is the path the reviewer measured at
//     ~4.92s / 20 runs (~7.4× slower).
//
//   * `regovm_a` — reviewer's first suggestion, applied minimally: hoist the
//     `Engine` + `add_policy`/`add_data`/wrapper load OUT of the rule loop, but
//     keep per-rule `compile_with_entrypoint` + `compile_from_policy` (single
//     entry) unchanged. This preserves the "skip rules not defined in the
//     policy" contract natively (the per-rule `compile_with_entrypoint` still
//     throws `not a valid rule path` -> caught -> skip), so it carries NO
//     behavioral risk. Isolates how much pure load-hoisting recovers.
//
//   * `regovm_a_c` — `regovm_a` plus a per-(policy_hash, rule) `Arc<Program>`
//     cache persisted across evaluations. Cache misses pay the full compile
//     (and build the `Engine` lazily, only on the first miss); cache hits skip
//     `Engine`/parse/compile entirely and just `load_program` + run. This
//     measures the EAR broker's real load (same `default.rego` evaluated
//     repeatedly): 1 cold eval + 19 cached.
//
// `regovm_a` / `regovm_a_c` deliberately do NOT use multi-entry-point compile
// (`compile_from_policy(&cp, &[all entries])`): that would change the
// "missing rule -> skip" behavior into "missing rule -> whole compile fails",
// which is a behavioral regression we are not willing to ship silently.
//
// The empty-input scenario (`}`) means no platform block matches; the
// `query_reference_value` host-await wrapper is still *compiled* (that is the
// cost being amplified) but host-await I/O is incidental, not the bottleneck.
//
// Run: `cargo test -p attestation-service eval_bench_default_policy -- --ignored --nocapture`
// Benchmarks the regovm optimization (cached vs uncached program runs against
// the sync-`Engine` interpreter baseline), so it only compiles under
// `regorus-regovm`. The `interpreter_eval` baseline uses `regorus::Engine`
// directly, which regorus exposes regardless of our backend feature.
#[cfg(all(test, feature = "policy-rvps", feature = "regorus-regovm"))]
mod eval_bench {
    use super::*;
    use crate::rvps::test_resolver;
    use sha2::Digest;
    use std::time::{Duration, Instant};

    /// The production default EAR policy, the same source `EarAttestationTokenBroker`
    /// loads as `default.rego`.
    const DEFAULT_POLICY: &str = include_str!("../../token/ear_default_policy_cpu.rego");

    /// The four AR4SI trustworthiness-claim rules `EarAttestationTokenBroker`
    /// derives from `TrustVector::new()` (hyphens -> underscores).
    const TRUST_VECTOR_RULES: &[&str] =
        &["executables", "hardware", "configuration", "file_system"];

    /// Faithful reconstruction of origin/main's `evaluate_sync`: one `Engine`,
    /// policy/data/input loaded once, rules evaluated via `engine.eval_rule`.
    /// This is the "traditional regorus interpreter" baseline.
    #[allow(dead_code)]
    fn interpreter_eval(
        policy: &str,
        input: &str,
        policy_id: &str,
        evaluation_rules: &[String],
        data: &str,
    ) -> Result<std::collections::HashMap<String, serde_json::Value>, PolicyError> {
        let mut engine = regorus::Engine::new();
        // regorus 0.11 defaults to rego.v1; the production engine sets rego.v0
        // so legacy `allow { ... }` policies still parse. Match that here.
        engine.set_rego_v0(true);

        engine
            .add_policy(policy_id.to_string(), policy.to_string())
            .map_err(PolicyError::LoadPolicyFailed)?;
        let data_value =
            regorus::Value::from_json_str(data).map_err(PolicyError::JsonSerializationFailed)?;
        engine
            .add_data(data_value)
            .map_err(PolicyError::LoadReferenceDataFailed)?;
        engine
            .set_input_json(input)
            .map_err(PolicyError::SetInputDataFailed)?;

        // The default policy references `query_reference_value(...)`. origin/main
        // registered it as a sync `Engine` extension; the RegoVM path defines it
        // via the generated host-await wrapper module. The interpreter baseline
        // must register it too so regorus can resolve the call at compile time.
        // With the empty test_resolver every key maps to None -> Null, so the
        // extension returns Null, matching the regovm host-await path exactly.
        engine
            .add_extension(
                "query_reference_value".to_string(),
                1,
                Box::new(|_params: Vec<regorus::Value>| Ok(regorus::Value::Null)),
            )
            .map_err(PolicyError::EvalPolicyFailed)?;

        let mut rules_result = std::collections::HashMap::new();
        for rule in evaluation_rules {
            let whole_rule = format!("data.policy.{rule}");
            let claim_value = match engine.eval_rule(whole_rule) {
                Ok(value) => value,
                Err(error) if error.to_string().contains("not a valid rule path") => {
                    debug!("Policy `{policy_id}` does not check {rule}");
                    continue;
                }
                Err(error) => return Err(PolicyError::EvalPolicyFailed(error)),
            };
            let claim_value = claim_value
                .to_json_str()
                .map_err(PolicyError::JsonSerializationFailed)?;
            let claim_value =
                serde_json::from_str(&claim_value).map_err(PolicyError::SerdeJsonError)?;
            rules_result.insert(rule.clone(), claim_value);
        }
        Ok(rules_result)
    }

    /// Build the same `query_reference_value` host-await function the production
    /// `common_evaluate` registers, so the generated wrapper module is compiled
    /// on every call — matching real per-appraisal cost.
    ///
    /// This is a faithful IN-BENCH reconstruction of the *pre-optimization*
    /// `evaluate_with_regovm` (per-rule `Engine` + policy/data load +
    /// `compile_with_entrypoint` + `compile_from_policy` + fresh `RegoVM`). It
    /// is deliberately independent of the production fn so that production
    /// refactors (hoist/cache) do not silently change what this strategy
    /// measures — same approach as `interpreter_eval` reconstructing
    /// origin/main.
    async fn regovm_current_eval(
        policy: &str,
        input: &str,
        policy_id: &str,
        evaluation_rules: Vec<String>,
        data: String,
        resolver: Arc<ReferenceValueResolver>,
    ) -> Result<EvaluationResult, PolicyError> {
        let (functions, wrapper_module, data_value, input_value, policy_hash) =
            build_setup(policy, input, &data, resolver)?;

        let mut rules_result = std::collections::HashMap::new();
        for rule in &evaluation_rules {
            let entry_point = format!("data.policy.{rule}");
            // Build the engine PER RULE (the pre-optimization behaviour).
            let cp = {
                let mut engine = regorus::Engine::new();
                engine.set_rego_v0(true);
                engine
                    .add_data(data_value.clone())
                    .map_err(PolicyError::LoadPolicyFailed)?;
                engine
                    .add_policy(policy_id.to_string(), policy.to_string())
                    .map_err(PolicyError::LoadPolicyFailed)?;
                if let Some(wrapper) = &wrapper_module {
                    engine
                        .add_policy(EXTENSIONS_WRAPPER_MODULE_ID.to_string(), wrapper.clone())
                        .map_err(PolicyError::LoadPolicyFailed)?;
                }
                match engine.compile_with_entrypoint(&regorus::Rc::from(entry_point.clone())) {
                    Ok(cp) => cp,
                    Err(e) if e.to_string().contains("not a valid rule path") => {
                        debug!("Policy `{policy_id}` does not check {rule}");
                        continue;
                    }
                    Err(e) => return Err(PolicyError::LoadPolicyFailed(e)),
                }
            };
            let program = Compiler::compile_from_policy(&cp, &[entry_point.as_str()])
                .map_err(|e| PolicyError::LoadPolicyFailed(e.into()))?;
            let result_value = run_vm_rule(program, &data_value, &input_value, &functions).await?;
            let claim = serde_json::from_str(
                &result_value
                    .to_json_str()
                    .map_err(PolicyError::JsonSerializationFailed)?,
            )
            .map_err(PolicyError::SerdeJsonError)?;
            rules_result.insert(rule.clone(), claim);
        }
        Ok(EvaluationResult {
            rules_result,
            policy_hash,
        })
    }

    /// SHA-384 hex digest of the policy source, matching the `policy_hash`
    /// `evaluate_with_regovm` returns. Used as the cache key for `regovm_a_c`.
    fn policy_hash(policy: &str) -> String {
        let mut hasher = sha2::Sha384::new();
        sha2::Digest::update(&mut hasher, policy);
        hex::encode(sha2::Digest::finalize(hasher))
    }

    /// Run a single-entry `Program` on a fresh `RegoVM` with the given data/
    /// input, driving the suspendable host-await resume loop exactly like the
    /// production `evaluate_with_regovm` inner loop. Returns the rule's result.
    async fn run_vm_rule(
        program: Arc<regorus::rvm::Program>,
        data_value: &regorus::Value,
        input_value: &regorus::Value,
        functions: &HashMap<String, ExtensionFunction>,
    ) -> Result<regorus::Value, PolicyError> {
        let mut vm = regorus::rvm::RegoVM::new();
        vm.load_program(program);
        vm.set_data(data_value.clone())
            .map_err(|e| PolicyError::LoadReferenceDataFailed(e.into()))?;
        vm.set_input(input_value.clone());
        vm.set_execution_mode(ExecutionMode::Suspendable);

        let _ = vm
            .execute()
            .map_err(|e| PolicyError::EvalPolicyFailed(e.into()))?;
        loop {
            match vm.execution_state().clone() {
                ExecutionState::Suspended {
                    reason:
                        SuspendReason::HostAwait {
                            argument,
                            identifier,
                            ..
                        },
                    ..
                } => {
                    let v = dispatch(identifier, argument, functions).await?;
                    vm.resume(Some(v))
                        .map_err(|e| PolicyError::EvalPolicyFailed(e.into()))?;
                }
                ExecutionState::Completed { result } => break Ok(result),
                ExecutionState::Error { error } => {
                    break Err(PolicyError::EvalPolicyFailed(anyhow!(
                        "RegoVM error: {error}"
                    )))
                }
                other => {
                    break Err(PolicyError::EvalPolicyFailed(anyhow!(
                        "unexpected VM state: {other:?}"
                    )))
                }
            }
        }
    }

    /// Build the shared per-call setup (host-await functions, wrapper module,
    /// parsed data/input values, policy hash) used by both `regovm_a` and
    /// `regovm_a_c`.
    fn build_setup(
        policy: &str,
        input: &str,
        data: &str,
        resolver: Arc<ReferenceValueResolver>,
    ) -> Result<
        (
            HashMap<String, ExtensionFunction>,
            Option<String>,
            regorus::Value,
            regorus::Value,
            String,
        ),
        PolicyError,
    > {
        let mut functions = HashMap::<String, ExtensionFunction>::default();
        functions.insert(
            "query_reference_value".to_string(),
            query_reference_value_extension(resolver),
        );
        let wrapper_module = if functions.is_empty() {
            None
        } else {
            Some(build_extensions_module(&functions))
        };
        let data_value =
            regorus::Value::from_json_str(data).map_err(PolicyError::JsonSerializationFailed)?;
        let input_value =
            regorus::Value::from_json_str(input).map_err(PolicyError::SetInputDataFailed)?;
        Ok((
            functions,
            wrapper_module,
            data_value,
            input_value,
            policy_hash(policy),
        ))
    }

    /// `regovm_a`: hoist `Engine` + policy/data/wrapper load OUT of the rule
    /// loop; per rule, keep `compile_with_entrypoint` + single-entry
    /// `compile_from_policy` unchanged (so `not a valid rule path` is still
    /// caught per-rule -> skip, no behavioral change). No cross-eval caching.
    async fn regovm_a_eval(
        policy: &str,
        input: &str,
        policy_id: &str,
        evaluation_rules: &[String],
        data: &str,
        resolver: Arc<ReferenceValueResolver>,
    ) -> Result<EvaluationResult, PolicyError> {
        let (functions, wrapper_module, data_value, input_value, policy_hash) =
            build_setup(policy, input, data, resolver)?;

        // Hoisted out of the loop: build the engine and load policy/data once.
        let mut engine = regorus::Engine::new();
        engine.set_rego_v0(true);
        engine
            .add_data(data_value.clone())
            .map_err(PolicyError::LoadPolicyFailed)?;
        engine
            .add_policy(policy_id.to_string(), policy.to_string())
            .map_err(PolicyError::LoadPolicyFailed)?;
        if let Some(wrapper) = &wrapper_module {
            engine
                .add_policy(EXTENSIONS_WRAPPER_MODULE_ID.to_string(), wrapper.clone())
                .map_err(PolicyError::LoadPolicyFailed)?;
        }

        let mut rules_result = std::collections::HashMap::new();
        for rule in evaluation_rules {
            let entry_point = format!("data.policy.{rule}");
            let cp = match engine.compile_with_entrypoint(&regorus::Rc::from(entry_point.clone())) {
                Ok(cp) => cp,
                Err(e) if e.to_string().contains("not a valid rule path") => {
                    debug!("Policy `{policy_id}` does not check {rule}");
                    continue;
                }
                Err(e) => return Err(PolicyError::LoadPolicyFailed(e)),
            };
            let program = Compiler::compile_from_policy(&cp, &[entry_point.as_str()])
                .map_err(|e| PolicyError::LoadPolicyFailed(e.into()))?;
            let result_value = run_vm_rule(program, &data_value, &input_value, &functions).await?;
            let claim = serde_json::from_str(
                &result_value
                    .to_json_str()
                    .map_err(PolicyError::JsonSerializationFailed)?,
            )
            .map_err(PolicyError::SerdeJsonError)?;
            rules_result.insert(rule.clone(), claim);
        }
        Ok(EvaluationResult {
            rules_result,
            policy_hash,
        })
    }

    /// `regovm_a_c`: `regovm_a` plus a per-(policy_hash, rule) `Arc<Program>`
    /// cache. On a cache miss the `Engine` is built lazily (parse + loads
    /// happen once, for the first missed rule only); on a hit, skip
    /// `Engine`/parse/compile entirely and just `load_program` + run.
    async fn regovm_a_c_eval(
        policy: &str,
        input: &str,
        policy_id: &str,
        evaluation_rules: &[String],
        data: &str,
        resolver: Arc<ReferenceValueResolver>,
        cache: &mut HashMap<(String, String), Arc<regorus::rvm::Program>>,
    ) -> Result<EvaluationResult, PolicyError> {
        let (functions, wrapper_module, data_value, input_value, policy_hash) =
            build_setup(policy, input, data, resolver)?;

        let mut engine: Option<regorus::Engine> = None;
        let mut rules_result = std::collections::HashMap::new();
        for rule in evaluation_rules {
            let entry_point = format!("data.policy.{rule}");
            let program = match cache.get(&(policy_hash.clone(), rule.clone())) {
                Some(p) => p.clone(),
                None => {
                    // Build the engine lazily, only on the first cache miss.
                    if engine.is_none() {
                        let mut e = regorus::Engine::new();
                        e.set_rego_v0(true);
                        e.add_data(data_value.clone())
                            .map_err(PolicyError::LoadPolicyFailed)?;
                        e.add_policy(policy_id.to_string(), policy.to_string())
                            .map_err(PolicyError::LoadPolicyFailed)?;
                        if let Some(wrapper) = &wrapper_module {
                            e.add_policy(EXTENSIONS_WRAPPER_MODULE_ID.to_string(), wrapper.clone())
                                .map_err(PolicyError::LoadPolicyFailed)?;
                        }
                        engine = Some(e);
                    }
                    let eng = engine.as_mut().unwrap();
                    let cp = match eng
                        .compile_with_entrypoint(&regorus::Rc::from(entry_point.clone()))
                    {
                        Ok(cp) => cp,
                        Err(e) if e.to_string().contains("not a valid rule path") => {
                            debug!("Policy `{policy_id}` does not check {rule}");
                            continue;
                        }
                        Err(e) => return Err(PolicyError::LoadPolicyFailed(e)),
                    };
                    let p = Compiler::compile_from_policy(&cp, &[entry_point.as_str()])
                        .map_err(|e| PolicyError::LoadPolicyFailed(e.into()))?;
                    cache.insert((policy_hash.clone(), rule.clone()), p.clone());
                    p
                }
            };
            let result_value = run_vm_rule(program, &data_value, &input_value, &functions).await?;
            let claim = serde_json::from_str(
                &result_value
                    .to_json_str()
                    .map_err(PolicyError::JsonSerializationFailed)?,
            )
            .map_err(PolicyError::SerdeJsonError)?;
            rules_result.insert(rule.clone(), claim);
        }
        Ok(EvaluationResult {
            rules_result,
            policy_hash,
        })
    }

    /// Returns the wall-clock duration of the measured batch and the last result.
    async fn time_async<F, Fut>(
        warmup: usize,
        iters: usize,
        thunk: F,
    ) -> (Duration, EvaluationResult)
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<EvaluationResult, PolicyError>>,
    {
        for _ in 0..warmup {
            thunk().await.unwrap();
        }
        let start = Instant::now();
        let mut last = EvaluationResult {
            rules_result: std::collections::HashMap::new(),
            policy_hash: String::new(),
        };
        for _ in 0..iters {
            last = thunk().await.unwrap();
        }
        (start.elapsed(), last)
    }

    fn time_sync<F>(
        warmup: usize,
        iters: usize,
        thunk: F,
    ) -> (
        Duration,
        std::collections::HashMap<String, serde_json::Value>,
    )
    where
        F: Fn() -> Result<std::collections::HashMap<String, serde_json::Value>, PolicyError>,
    {
        for _ in 0..warmup {
            thunk().unwrap();
        }
        let start = Instant::now();
        let mut last = std::collections::HashMap::new();
        for _ in 0..iters {
            last = thunk().unwrap();
        }
        (start.elapsed(), last)
    }

    #[tokio::test]
    #[ignore = "perf benchmark: run with --ignored --nocapture"]
    async fn eval_bench_default_policy() {
        let policy = DEFAULT_POLICY.to_string();
        let input = "{}".to_string();
        let policy_id = "default".to_string();
        let rules: Vec<String> = TRUST_VECTOR_RULES.iter().map(|s| s.to_string()).collect();
        let resolver = test_resolver(std::collections::HashMap::new());

        // The default policy uses `query_reference_value(...)` (a host-await
        // builtin), never the legacy `data.reference` path, so `common_evaluate`
        // would hand `evaluate_with_regovm` an empty data document. Pin that so
        // both strategies see the same `data`.
        assert!(
            !policy_uses_legacy_reference(&policy).unwrap(),
            "default policy unexpectedly uses legacy `data.reference`; update the bench"
        );
        let data = "{}".to_string();

        const WARMUP: usize = 3;
        const ITERS: usize = 20;
        const BATCHES: usize = 3;

        // --- interpreter baseline ---
        let mut baseline_best = Duration::MAX;
        let mut baseline_result = std::collections::HashMap::new();
        for _ in 0..BATCHES {
            let (t, r) = time_sync(WARMUP, ITERS, || {
                interpreter_eval(&policy, &input, &policy_id, &rules, &data)
            });
            if t < baseline_best {
                baseline_best = t;
                baseline_result = r;
            }
        }

        // --- regovm current (per-rule Engine + compile) ---
        let mut regovm_best = Duration::MAX;
        let mut regovm_result = EvaluationResult {
            rules_result: std::collections::HashMap::new(),
            policy_hash: String::new(),
        };
        for _ in 0..BATCHES {
            let (t, r) = time_async(WARMUP, ITERS, || {
                regovm_current_eval(
                    &policy,
                    &input,
                    &policy_id,
                    rules.clone(),
                    data.clone(),
                    resolver.clone(),
                )
            })
            .await;
            if t < regovm_best {
                regovm_best = t;
                regovm_result = r;
            }
        }

        // --- regovm_a (hoist Engine+loads out of loop; no cross-eval cache) ---
        let mut a_best = Duration::MAX;
        let mut a_result = EvaluationResult {
            rules_result: std::collections::HashMap::new(),
            policy_hash: String::new(),
        };
        for _ in 0..BATCHES {
            let (t, r) = time_async(WARMUP, ITERS, || {
                regovm_a_eval(&policy, &input, &policy_id, &rules, &data, resolver.clone())
            })
            .await;
            if t < a_best {
                a_best = t;
                a_result = r;
            }
        }

        // --- regovm_a_c (regovm_a + per-(policy_hash,rule) program cache) ---
        // Each batch starts cold (empty cache): iter 1 pays the compile, iters
        // 2..=20 hit the cache. No warmup, so the measured batch captures the
        // realistic "1 cold + 19 warm" amortized cost.
        let mut ac_best = Duration::MAX;
        let mut ac_result = EvaluationResult {
            rules_result: std::collections::HashMap::new(),
            policy_hash: String::new(),
        };
        for _ in 0..BATCHES {
            let mut cache: HashMap<(String, String), Arc<regorus::rvm::Program>> = HashMap::new();
            let start = Instant::now();
            let mut last = EvaluationResult {
                rules_result: std::collections::HashMap::new(),
                policy_hash: String::new(),
            };
            for _ in 0..ITERS {
                last = regovm_a_c_eval(
                    &policy,
                    &input,
                    &policy_id,
                    &rules,
                    &data,
                    resolver.clone(),
                    &mut cache,
                )
                .await
                .unwrap();
            }
            let t = start.elapsed();
            if t < ac_best {
                ac_best = t;
                ac_result = last;
            }
        }

        // Correctness: all four RegoVM strategies must agree with the
        // interpreter baseline on every rule's claim value. (With empty input
        // every platform block is absent, so all four rules fall to their
        // `default` AR4SI values: 33/97/36/35.)
        for (label, result) in [
            ("regovm_current", &regovm_result),
            ("regovm_a", &a_result),
            ("regovm_a_c", &ac_result),
        ] {
            assert_eq!(
                baseline_result
                    .keys()
                    .collect::<std::collections::HashSet<_>>(),
                result
                    .rules_result
                    .keys()
                    .collect::<std::collections::HashSet<_>>(),
                "{label}: rule set differs from baseline"
            );
            for (rule, b) in &baseline_result {
                let v = result.rules_result.get(rule).unwrap();
                assert_eq!(
                    b, v,
                    "{label}: rule `{rule}` value differs: baseline={b} {label}={v}"
                );
            }
        }

        let baseline_per = baseline_best / ITERS as u32;
        let regovm_per = regovm_best / ITERS as u32;
        let a_per = a_best / ITERS as u32;
        let ac_per = ac_best / ITERS as u32;
        let ratio = |d: Duration| d.as_secs_f64() / baseline_best.as_secs_f64();

        eprintln!();
        eprintln!("================ eval_bench_default_policy ================");
        eprintln!(
            "policy: default EAR (ear_default_policy_cpu.rego), {} rules",
            rules.len()
        );
        eprintln!("input : {{}} (no platform match; host-await wrapper still compiled)");
        eprintln!("iters : {ITERS} (best of {BATCHES} batches)");
        eprintln!("          baseline/regovm_current/regovm_a: {WARMUP} warmup discarded");
        eprintln!(
            "          regovm_a_c: 1 cold + {warm} warm per batch (cache reset)",
            warm = ITERS - 1
        );
        eprintln!("----------------------------------------------------------");
        eprintln!(
            "{:<22} {:>12} {:>12} {:>10}",
            "strategy", "total(20)", "mean/eval", "ratio"
        );
        eprintln!(
            "{:<22} {:>10.2}s {:>9.1}ms {:>9.2}x",
            "interpreter_baseline",
            baseline_best.as_secs_f64(),
            baseline_per.as_secs_f64() * 1000.0,
            1.0
        );
        eprintln!(
            "{:<22} {:>10.2}s {:>9.1}ms {:>9.2}x",
            "regovm_current",
            regovm_best.as_secs_f64(),
            regovm_per.as_secs_f64() * 1000.0,
            ratio(regovm_best)
        );
        eprintln!(
            "{:<22} {:>10.2}s {:>9.1}ms {:>9.2}x",
            "regovm_a (hoist)",
            a_best.as_secs_f64(),
            a_per.as_secs_f64() * 1000.0,
            ratio(a_best)
        );
        eprintln!(
            "{:<22} {:>10.2}s {:>9.1}ms {:>9.2}x",
            "regovm_a_c (hoist+cache)",
            ac_best.as_secs_f64(),
            ac_per.as_secs_f64() * 1000.0,
            ratio(ac_best)
        );
        eprintln!("==========================================================");
        eprintln!("reviewer target: baseline ~0.67s, regovm_current ~4.92s, ratio ~7.4x");
        eprintln!("regovm_a isolates load-hoisting; regovm_a_c adds cross-eval program cache");
    }
}
