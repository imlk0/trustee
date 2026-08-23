//! fs-free `PolicyEngine`: holds rego policy sources in memory and evaluates
//! them with Regorus, mirroring `opa::OPA`'s engine usage but without any
//! filesystem access. Selected by `PolicyEngineType::InMemory`.

use std::collections::HashMap;
use std::sync::Arc;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha384};
use tokio::sync::RwLock;

use crate::{
    policy_engine::{EvaluationResult, PolicyDigest, PolicyEngine, PolicyError},
    rvps::ReferenceValueResolver,
};

/// In-memory policy store: `policy_id` -> raw rego source (decoded from the
/// base64url wire format `set_policy` receives, matching `opa::OPA`).
pub struct OPAInMemory {
    policies: RwLock<HashMap<String, Vec<u8>>>,
    #[cfg(feature = "policy-artifact-server")]
    artifact_server_client: Arc<artifact_resolve_sdk::Client>,
    /// Caller-injected host-await functions exposed to rego policy via the
    /// `__builtin_host_await` wrapper. Generic injection point so a downstream
    /// crate can supply functions regorus 0.11 lacks by design (e.g.
    /// `crypto.sha256`). Each entry's key becomes a rego-callable function name
    /// (dotted keys like `crypto.sha256` are supported by regorus's function
    /// rule syntax). `None` keeps the legacy behavior (built-ins only).
    extra_host_await_functions: Option<Vec<(String, super::RegoVmHostAwaitFunction)>>,
}

impl OPAInMemory {
    /// Build an engine with a default policy preloaded, mirroring `opa::OPA::new`
    /// which writes the default policy to `{dir}/{default_policy_id}` on disk.
    /// The policy is stored under the stem of `default_policy_id` (`.rego`
    /// stripped), matching how `opa::OPA::evaluate` looks up `{policy_id}.rego`.
    /// This lets a broker's default policy flow (`evaluate(..., "default", ...)`)
    /// succeed without any filesystem access.
    pub fn with_raw_default_policy(
        raw_default_policy: &str,
        default_policy_id: &str,
        #[cfg_attr(not(feature = "policy-artifact-server"), allow(unused_variables))]
        artifact_server_address: &str,
    ) -> Result<Self, PolicyError> {
        #[cfg(not(feature = "policy-artifact-server"))]
        let _ = artifact_server_address;

        let stem = default_policy_id.trim_end_matches(".rego");
        let mut policies = HashMap::new();
        policies.insert(stem.to_string(), raw_default_policy.as_bytes().to_vec());
        Ok(Self {
            policies: RwLock::new(policies),
            #[cfg(feature = "policy-artifact-server")]
            artifact_server_client: Arc::new(
                artifact_resolve_sdk::Client::new(artifact_server_address)
                    .map_err(PolicyError::ArtifactServerClientCreationFailed)?,
            ),
            extra_host_await_functions: None,
        })
    }

    /// Inject additional host-await functions callable from rego policy. Each
    /// `(key, function)` pair registers a rego function named `key` (dotted keys
    /// such as `crypto.sha256` are accepted) that suspends the VM and resumes it
    /// with the function's result. User functions are merged after the built-in
    /// `query_reference_value` / `query_artifact_server` extensions, so a
    /// colliding key is overridden by the caller's explicit choice.
    ///
    /// This is the generic extension point that lets a downstream crate supply
    /// host functions regorus 0.11 omits by design (e.g. `crypto.sha256`).
    pub fn with_extra_host_await_functions(
        mut self,
        functions: Vec<(String, super::RegoVmHostAwaitFunction)>,
    ) -> Self {
        self.extra_host_await_functions = Some(functions);
        self
    }
}

#[cfg_attr(all(target_arch = "wasm32", target_vendor = "unknown", target_os = "unknown"), async_trait::async_trait(?Send))]
#[cfg_attr(
    not(all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    )),
    async_trait::async_trait
)]
impl PolicyEngine for OPAInMemory {
    async fn evaluate(
        &self,
        input: &str,
        policy_id: &str,
        evaluation_rules: Vec<String>,
        reference_value_resolver: Arc<ReferenceValueResolver>,
    ) -> Result<EvaluationResult, PolicyError> {
        let policies = self.policies.read().await;
        let policy = policies
            .get(policy_id)
            .map(|b| b.as_slice())
            .ok_or_else(|| {
                PolicyError::ReadPolicyFileFailed(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("policy {policy_id} not found"),
                ))
            })?;
        let policy =
            std::str::from_utf8(policy).map_err(|e| PolicyError::InvalidPolicy(e.into()))?;

        super::common_evaluate(
            policy.to_string(),
            input.to_string(),
            policy_id.to_string(),
            evaluation_rules,
            reference_value_resolver,
            #[cfg(feature = "policy-artifact-server")]
            self.artifact_server_client.clone(),
            // The functions are `Arc` handles, so cloning the `Vec` is cheap and
            // lets `evaluate(&self)` hand the injected functions to
            // `common_evaluate` without moving out of `&self`.
            self.extra_host_await_functions.clone(),
        )
        .await
    }

    async fn set_policy(&self, policy_id: String, policy: String) -> Result<(), PolicyError> {
        if !super::is_valid_policy_id(&policy_id) {
            return Err(PolicyError::InvalidPolicyId);
        }
        let bytes = URL_SAFE_NO_PAD.decode(policy)?;
        // validate it compiles as rego
        {
            let src =
                std::str::from_utf8(&bytes).map_err(|e| PolicyError::InvalidPolicy(e.into()))?;
            let mut engine = regorus::Engine::new();
            // regorus 0.11 defaults to rego.v1; keep accepting legacy
            // `allow { ... }` (rego.v0) policies that were saved before the
            // rego.v1 migration. `import rego.v1` policies still work.
            engine.set_rego_v0(true);
            engine
                .add_policy(policy_id.clone(), src.to_string())
                .map_err(PolicyError::InvalidPolicy)?;
        }
        let mut policies = self.policies.write().await;
        policies.insert(policy_id, bytes);
        Ok(())
    }

    async fn list_policies(&self) -> Result<HashMap<String, PolicyDigest>, PolicyError> {
        let policies = self.policies.read().await;
        let mut out = HashMap::new();
        for (id, bytes) in policies.iter() {
            let mut h = Sha384::new();
            h.update(bytes);
            out.insert(id.clone(), URL_SAFE_NO_PAD.encode(h.finalize()));
        }
        Ok(out)
    }

    async fn get_policy(&self, policy_id: String) -> Result<String, PolicyError> {
        let policies = self.policies.read().await;
        let bytes = policies.get(&policy_id).ok_or_else(|| {
            PolicyError::ReadPolicyFileFailed(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("policy {policy_id} not found"),
            ))
        })?;
        Ok(URL_SAFE_NO_PAD.encode(bytes))
    }

    async fn delete_policy(&self, policy_id: String) -> Result<(), PolicyError> {
        if !super::is_valid_policy_id(&policy_id) {
            return Err(PolicyError::InvalidPolicyId);
        }
        if policy_id == "default" {
            return Err(PolicyError::CannotDeleteDefaultPolicy);
        }
        let mut policies = self.policies.write().await;
        policies.remove(&policy_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::DEFAULT_ARTIFACT_SERVER_ADDRESS;

    use super::*;

    const RAW_ALLOW_POLICY: &str = "package policy\ndefault allow = true";
    fn allow_policy() -> String {
        URL_SAFE_NO_PAD.encode(RAW_ALLOW_POLICY)
    }

    #[tokio::test]
    async fn set_get_list_delete_roundtrip() {
        let eng = OPAInMemory::with_raw_default_policy(
            RAW_ALLOW_POLICY,
            "test",
            DEFAULT_ARTIFACT_SERVER_ADDRESS,
        )
        .unwrap();
        assert_eq!(eng.list_policies().await.unwrap().len(), 1);
        assert_eq!(eng.get_policy("test".into()).await.unwrap(), allow_policy());
        eng.delete_policy("test".into()).await.unwrap();
        assert!(eng.list_policies().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_policy_then_get_roundtrip() {
        // Covers the set_policy -> get_policy path (the roundtrip above only
        // exercises with_default_policy). Verifies raw rego in == raw rego out.
        let eng = OPAInMemory::with_raw_default_policy(
            RAW_ALLOW_POLICY,
            "default",
            DEFAULT_ARTIFACT_SERVER_ADDRESS,
        )
        .unwrap();
        eng.set_policy("test".into(), allow_policy()).await.unwrap();
        let got = eng.get_policy("test".into()).await.unwrap();
        assert_eq!(got, allow_policy());
        // setting again overwrites cleanly
        eng.set_policy("test".into(), allow_policy()).await.unwrap();
        assert_eq!(eng.get_policy("test".into()).await.unwrap(), allow_policy());
    }

    #[tokio::test]
    async fn evaluate_uses_in_memory_policy() {
        let eng = OPAInMemory::with_raw_default_policy(
            RAW_ALLOW_POLICY,
            "test",
            DEFAULT_ARTIFACT_SERVER_ADDRESS,
        )
        .unwrap();
        eng.set_policy("p".into(), allow_policy()).await.unwrap();
        let res = eng
            .evaluate(
                "{}",
                "test",
                vec!["allow".into()],
                crate::rvps::test_resolver(HashMap::from([])),
            )
            .await
            .unwrap();
        assert!(res.rules_result.contains_key("allow"));
    }

    #[cfg(feature = "policy-rvps")]
    #[tokio::test]
    async fn evaluate_with_host_await_reference_value() {
        use crate::rvps::test_resolver;
        let eng = OPAInMemory::with_raw_default_policy(
            RAW_ALLOW_POLICY,
            "test",
            DEFAULT_ARTIFACT_SERVER_ADDRESS,
        )
        .unwrap();
        let policy = r#"package policy
import rego.v1
allow if {
  input.x == query_reference_value("k")
}
"#;
        eng.set_policy("p".into(), URL_SAFE_NO_PAD.encode(policy))
            .await
            .unwrap();
        let rvps = test_resolver(std::collections::HashMap::from([(
            "k".to_string(),
            serde_json::json!(1),
        )]));
        let res = eng
            .evaluate(r#"{"x":1}"#, "p", vec!["allow".into()], rvps)
            .await
            .unwrap();
        assert_eq!(
            res.rules_result.get("allow"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    // Injects a host-await function under the dotted key `crypto.sha256` (the
    // name regorus 0.11 lacks by design) and verifies a rego policy can call
    // `crypto.sha256("abc")` through the existing build_extensions wrapper and
    // receive the real sha256 hex. This exercises the generic injection point
    // end-to-end via OPAInMemory::evaluate, including the dotted function-rule
    // definition that build_extensions emits.
    #[cfg(feature = "policy-rvps")]
    #[tokio::test]
    async fn evaluate_with_injected_crypto_sha256_dotted_host_await() {
        use crate::policy_engine::opa::RegoVmHostAwaitFunction;
        use crate::rvps::test_resolver;
        use sha2::Digest;

        let sha256_fn: RegoVmHostAwaitFunction = Arc::new(|argument: regorus::Value| {
            Box::pin(async move {
                let s = argument.as_string().map_err(|e| {
                    PolicyError::EvalPolicyFailed(anyhow::anyhow!(
                        "crypto.sha256 arg not a string: {e}"
                    ))
                })?;
                let mut hasher = sha2::Sha256::new();
                hasher.update(s.as_bytes());
                Ok(regorus::Value::String(
                    hex::encode(hasher.finalize()).into(),
                ))
            })
        });
        let policy = r#"package policy
import rego.v1
test_hash := crypto.sha256("abc")
"#;
        let eng = OPAInMemory::with_raw_default_policy(
            policy,
            "default",
            DEFAULT_ARTIFACT_SERVER_ADDRESS,
        )
        .expect("build engine")
        .with_extra_host_await_functions(vec![("crypto.sha256".to_string(), sha256_fn)]);

        let res = eng
            .evaluate(
                "{}",
                "default",
                vec!["test_hash".to_string()],
                test_resolver(HashMap::new()),
            )
            .await
            .expect("evaluate");
        let got = res.rules_result.get("test_hash").expect("test_hash result");
        assert_eq!(
            got.as_str(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }
}
