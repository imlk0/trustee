//! fs-free `PolicyEngine`: holds rego policy sources in memory and evaluates
//! them with Regorus, mirroring `opa::OPA`'s engine usage but without any
//! filesystem access. Selected by `PolicyEngineType::InMemory`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha384};
use tokio::sync::RwLock;

use super::{EvaluationResult, PolicyDigest, PolicyEngine, PolicyError};

/// In-memory policy store: `policy_id` -> raw rego source (decoded from the
/// base64url wire format `set_policy` receives, matching `opa::OPA`).
pub struct InMemoryPolicyEngine {
    policies: RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryPolicyEngine {
    pub fn new() -> Self {
        Self { policies: RwLock::new(HashMap::new()) }
    }

    /// Build an engine with a default policy preloaded, mirroring `opa::OPA::new`
    /// which writes the default policy to `{dir}/{default_policy_id}` on disk.
    /// The policy is stored under the stem of `default_policy_id` (`.rego`
    /// stripped), matching how `opa::OPA::evaluate` looks up `{policy_id}.rego`.
    /// This lets a broker's default policy flow (`evaluate(..., "default", ...)`)
    /// succeed without any filesystem access.
    pub fn with_default_policy(default_policy: &str, default_policy_id: &str) -> Self {
        let stem = default_policy_id.trim_end_matches(".rego");
        let mut policies = HashMap::new();
        policies.insert(stem.to_string(), default_policy.as_bytes().to_vec());
        Self { policies: RwLock::new(policies) }
    }

    fn is_valid_policy_id(policy_id: &str) -> bool {
        policy_id.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    }
}

impl Default for InMemoryPolicyEngine {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl PolicyEngine for InMemoryPolicyEngine {
    async fn evaluate(
        &self,
        data: &str,
        input: &str,
        policy_id: &str,
        evaluation_rules: Vec<String>,
    ) -> Result<EvaluationResult, PolicyError> {
        let policies = self.policies.read().await;
        let policy = policies
            .get(policy_id)
            .map(|b| b.as_slice())
            .ok_or_else(|| PolicyError::ReadPolicyFileFailed(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("policy {policy_id} not found"),
            )))?;
        let policy = std::str::from_utf8(policy)
            .map_err(|e| PolicyError::InvalidPolicy(e.into()))?;

        let mut engine = regorus::Engine::new();
        engine
            .add_policy(policy_id.to_string(), policy.to_string())
            .map_err(PolicyError::LoadPolicyFailed)?;
        let data = regorus::Value::from_json_str(data)
            .map_err(PolicyError::JsonSerializationFailed)?;
        engine.add_data(data).map_err(PolicyError::LoadReferenceDataFailed)?;
        engine.set_input_json(input).context("set input").map_err(PolicyError::SetInputDataFailed)?;

        let policy_hash = {
            let mut h = Sha384::new();
            h.update(policy.as_bytes());
            hex::encode(h.finalize())
        };

        let mut rules_result = HashMap::new();
        for rule in evaluation_rules {
            let whole_rule = format!("data.policy.{rule}");
            if let Ok(v) = engine.eval_rule(whole_rule) {
                rules_result.insert(rule, v);
            }
        }
        Ok(EvaluationResult { rules_result, policy_hash })
    }

    async fn set_policy(&self, policy_id: String, policy: String) -> Result<(), PolicyError> {
        if !Self::is_valid_policy_id(&policy_id) {
            return Err(PolicyError::InvalidPolicyId);
        }
        let bytes = URL_SAFE_NO_PAD.decode(policy)?;
        // validate it compiles as rego
        {
            let src = std::str::from_utf8(&bytes).map_err(|e| PolicyError::InvalidPolicy(e.into()))?;
            let mut engine = regorus::Engine::new();
            engine.add_policy(policy_id.clone(), src.to_string()).map_err(PolicyError::InvalidPolicy)?;
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
        let bytes = policies.get(&policy_id).ok_or_else(|| PolicyError::ReadPolicyFileFailed(
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("policy {policy_id} not found")),
        ))?;
        Ok(URL_SAFE_NO_PAD.encode(bytes))
    }

    async fn delete_policy(&self, policy_id: String) -> Result<(), PolicyError> {
        if !Self::is_valid_policy_id(&policy_id) {
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
    use super::*;
    use serde_json::json;

    fn allow_policy() -> String {
        URL_SAFE_NO_PAD.encode(b"package policy\ndefault allow = true")
    }

    #[tokio::test]
    async fn set_get_list_delete_roundtrip() {
        let eng = InMemoryPolicyEngine::new();
        eng.set_policy("test".into(), allow_policy()).await.unwrap();
        assert_eq!(eng.list_policies().await.unwrap().len(), 1);
        assert_eq!(eng.get_policy("test".into()).await.unwrap(), allow_policy());
        eng.delete_policy("test".into()).await.unwrap();
        assert!(eng.list_policies().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn evaluate_uses_in_memory_policy() {
        let eng = InMemoryPolicyEngine::new();
        eng.set_policy("p".into(), allow_policy()).await.unwrap();
        let res = eng.evaluate("{}", "{}", "p", vec!["allow".into()]).await.unwrap();
        assert!(res.rules_result.contains_key("allow"));
    }
}
