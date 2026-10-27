//! Bearer token specs, as in the Python relay (`auth.py`): one token, `new,old` (any is
//! accepted, for rotation), or `@/path/tokens` — a file re-read when it changes, so tokens
//! rotate without a restart.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

const RELOAD_INTERVAL: Duration = Duration::from_secs(1);

pub fn parse_tokens(text: &str) -> Vec<Vec<u8>> {
    text.lines()
        .flat_map(|line| line.split('#').next().unwrap_or("").split(','))
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.as_bytes().to_vec())
        .collect()
}

struct Loaded {
    tokens: Vec<Vec<u8>>,
    mtime: Option<SystemTime>,
    checked: Option<Instant>,
}

pub struct TokenSet {
    path: Option<PathBuf>,
    loaded: Mutex<Loaded>,
}

impl TokenSet {
    pub fn new(spec: &str) -> Self {
        let path = spec.strip_prefix('@').map(PathBuf::from);
        let tokens = if path.is_some() {
            Vec::new()
        } else {
            parse_tokens(spec)
        };
        let set = TokenSet {
            path,
            loaded: Mutex::new(Loaded {
                tokens,
                mtime: None,
                checked: None,
            }),
        };
        set.accepts(b""); // loads a token file now, so errors show at startup
        set
    }

    pub fn accepts(&self, given: &[u8]) -> bool {
        let mut loaded = self.loaded.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(path) = &self.path {
            let due = loaded
                .checked
                .is_none_or(|at| at.elapsed() >= RELOAD_INTERVAL);
            if due {
                loaded.checked = Some(Instant::now());
                match std::fs::metadata(path).and_then(|m| m.modified()) {
                    Ok(mtime) if Some(mtime) != loaded.mtime => match std::fs::read_to_string(path)
                    {
                        Ok(text) => {
                            loaded.tokens = parse_tokens(&text);
                            loaded.mtime = Some(mtime);
                        }
                        Err(e) => eprintln!("[!] Cannot read token file {}: {e}", path.display()),
                    },
                    Ok(_) => {}
                    // Keep the last good tokens; with none loaded yet everything is refused.
                    Err(e) => eprintln!("[!] Cannot read token file {}: {e}", path.display()),
                }
            }
        }
        // Compare with every token (no early exit), each in constant time.
        loaded
            .tokens
            .iter()
            .fold(false, |ok, token| constant_time_eq(given, token) | ok)
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn several_tokens_and_comments() {
        let set = TokenSet::new("new, old");
        assert!(set.accepts(b"new") && set.accepts(b"old") && !set.accepts(b"ne"));
        assert_eq!(
            parse_tokens("a # note\n\nb,c"),
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
        );
    }

    #[test]
    fn missing_file_refuses_everything() {
        assert!(!TokenSet::new("@/nonexistent/distllm-tokens").accepts(b""));
    }
}
