use anyhow::*;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use rand::rngs::OsRng;
use rand::RngCore;
use rsa::pkcs1v15::{Signature, SigningKey, VerifyingKey};
use rsa::signature::{Signer, Verifier};
use rsa::{RsaPrivateKey, RsaPublicKey};
use serde_json::{json, Value};
use sha2::Sha384;

mod ephemeral;
#[cfg(feature = "fs")]
mod fs;

pub use ephemeral::EphemeralChallengeKey;
#[cfg(feature = "fs")]
pub use fs::FsChallengeKey;

pub const RSA_KEY_BITS: u32 = 2048;
const TOKEN_ALG: &str = "RS384";

/// Abstraction over how the Attestation Service signs and verifies
/// attestation-challenge (nonce) tokens. Implementations differ only in how
/// they obtain the RSA private key; the JWT protocol itself is shared.
#[cfg_attr(
    all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    ),
    async_trait::async_trait(?Send)
)]
#[cfg_attr(
    not(all(
        target_arch = "wasm32",
        target_vendor = "unknown",
        target_os = "unknown"
    )),
    async_trait::async_trait
)]
pub trait Challenger {
    /// Issue a fresh challenge (nonce) token. Returns the outer JSON
    /// `{"nonce": <b64>, "extra-params": {"jwt": <jwt>}}` — same shape as the
    /// historical free function, so the public AS API is unchanged.
    async fn generate_challenge(&self) -> Result<String>;

    /// Verify a challenge_token JWT, enforce its `exp` claim, and return the
    /// nonce base64url-no-pad encoded. Rejects bad signatures / expired
    /// tokens.
    async fn verify_challenge_and_extract_nonce_b64url(&self, token: &str) -> Result<String>;
}

/// Build the challenge JSON `{"nonce", "extra-params":{"jwt"}}` signed with
/// `key` (RS384, 5-minute `exp`). Shared by every [`Challenger`] impl.
fn build_challenge_json(key: &RsaPrivateKey) -> Result<String> {
    // nonce
    let mut nonce = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut nonce)
        .context("generate nonce failed")?;
    let nonce_b64 = STANDARD.encode(nonce);

    // header
    let header_value = json!({
        "typ": "JWT",
        "alg": TOKEN_ALG,
    });
    let header_string = serde_json::to_string(&header_value)?;
    let header_b64 = URL_SAFE_NO_PAD.encode(header_string.as_bytes());

    // claims with 5-minute expiry
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("time error")?
        .as_secs();
    let exp = now + 5 * 60;
    let claims_value = json!({
        "nonce": nonce_b64,
        "iat": now,
        "exp": exp,
    });
    let claims_string = serde_json::to_string(&claims_value)?;
    let claims_b64 = URL_SAFE_NO_PAD.encode(claims_string.as_bytes());

    // sign
    let signing_input = format!("{}.{}", header_b64, claims_b64);
    let signature = rs384_sign(key, signing_input.as_bytes())?;
    let signature_b64 = URL_SAFE_NO_PAD.encode(signature);
    let jwt = format!("{}.{}", signing_input, signature_b64);

    // output json
    let output = json!({
        "nonce": claims_value["nonce"].as_str().unwrap_or_default(),
        "extra-params": { "jwt": jwt },
    });
    Ok(serde_json::to_string(&output)?)
}

fn rs384_sign(rsa: &RsaPrivateKey, payload: &[u8]) -> Result<Vec<u8>> {
    let signing_key = SigningKey::<Sha384>::new(rsa.clone());
    let sig: Signature = signing_key.sign(payload);
    Ok(Box::<[u8]>::from(sig).to_vec())
}

/// Verify a challenge_token JWT signed by the public half of `key`, enforce
/// `exp`, and return the nonce base64url-no-pad encoded. Shared by every
/// [`Challenger`] impl.
fn verify_jwt(token: &str, key: &RsaPrivateKey) -> Result<String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        bail!("invalid JWT format in challenge_token");
    }

    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig = URL_SAFE_NO_PAD
        .decode(parts[2])
        .context("invalid JWT signature encoding")?;

    let public_key: RsaPublicKey = key.to_public_key();
    let verifying_key = VerifyingKey::<Sha384>::new(public_key);
    let sig_obj = Signature::try_from(sig.as_slice()).context("invalid signature bytes")?;
    verifying_key
        .verify(signing_input.as_bytes(), &sig_obj)
        .context("verify signature failed")?;

    let payload = URL_SAFE_NO_PAD
        .decode(parts[1])
        .context("invalid JWT payload encoding")?;
    let v: Value = serde_json::from_slice(&payload).context("invalid JWT payload json")?;

    // exp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("time error")?
        .as_secs() as i64;
    let exp = v
        .get("exp")
        .and_then(|x| x.as_i64())
        .ok_or_else(|| anyhow!("missing exp claim in challenge_token"))?;
    if now > exp {
        bail!("challenge_token expired");
    }

    let nonce_b64 = v
        .get("nonce")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("missing nonce claim in challenge_token"))?;
    let nonce_bytes = STANDARD
        .decode(nonce_b64)
        .or_else(|_| URL_SAFE_NO_PAD.decode(nonce_b64))
        .context("invalid nonce base64")?;
    Ok(URL_SAFE_NO_PAD.encode(nonce_bytes))
}
