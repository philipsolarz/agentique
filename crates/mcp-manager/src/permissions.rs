use std::collections::HashSet;

/// Approval policy for MCP tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalPolicy {
    /// Tool is auto-approved (read-only from trusted server).
    AutoApprove,
    /// Tool was approved once for this session; subsequent calls auto-approved.
    SessionApprove,
    /// Tool requires explicit approval for each call.
    PerCallApprove,
}

/// Manages tool approval decisions for MCP tools.
#[derive(Debug, Default)]
pub struct PermissionManager {
    /// Tools that have been approved for the current session.
    session_approved: HashSet<String>,
    /// Trusted server names (auto-approve read-only tools).
    trusted_servers: HashSet<String>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a server as trusted.
    pub fn trust_server(&mut self, server_name: &str) {
        self.trusted_servers.insert(server_name.to_string());
    }

    /// Determine the approval policy for a tool call.
    pub fn policy_for(
        &self,
        namespaced_tool: &str,
        is_read_only: bool,
        is_destructive: bool,
    ) -> ApprovalPolicy {
        // Extract server name from namespaced tool name (e.g., "github:create_issue" → "github")
        let server_name = namespaced_tool
            .split(':')
            .next()
            .unwrap_or(namespaced_tool);

        // Destructive tools always require per-call approval
        if is_destructive {
            return ApprovalPolicy::PerCallApprove;
        }

        // Read-only tools from trusted servers are auto-approved
        if is_read_only && self.trusted_servers.contains(server_name) {
            return ApprovalPolicy::AutoApprove;
        }

        // Check if already session-approved
        if self.session_approved.contains(namespaced_tool) {
            return ApprovalPolicy::SessionApprove;
        }

        // Default: per-call approval
        ApprovalPolicy::PerCallApprove
    }

    /// Record a session-level approval for a tool.
    pub fn approve_for_session(&mut self, namespaced_tool: &str) {
        self.session_approved.insert(namespaced_tool.to_string());
    }

    /// Check if a tool call is approved (auto or session-approved).
    pub fn is_approved(&self, namespaced_tool: &str, is_read_only: bool, is_destructive: bool) -> bool {
        matches!(
            self.policy_for(namespaced_tool, is_read_only, is_destructive),
            ApprovalPolicy::AutoApprove | ApprovalPolicy::SessionApprove
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_server_auto_approves_read_only() {
        let mut pm = PermissionManager::new();
        pm.trust_server("filesystem");

        assert_eq!(
            pm.policy_for("filesystem:read_file", true, false),
            ApprovalPolicy::AutoApprove
        );
        // Write tool from trusted server still needs approval
        assert_eq!(
            pm.policy_for("filesystem:write_file", false, true),
            ApprovalPolicy::PerCallApprove
        );
    }

    #[test]
    fn session_approval() {
        let mut pm = PermissionManager::new();
        assert_eq!(
            pm.policy_for("github:list_repos", true, false),
            ApprovalPolicy::PerCallApprove
        );
        pm.approve_for_session("github:list_repos");
        assert_eq!(
            pm.policy_for("github:list_repos", true, false),
            ApprovalPolicy::SessionApprove
        );
    }

    #[test]
    fn destructive_always_per_call() {
        let mut pm = PermissionManager::new();
        pm.trust_server("database");
        pm.approve_for_session("database:drop_table");
        // Destructive overrides everything
        assert_eq!(
            pm.policy_for("database:drop_table", false, true),
            ApprovalPolicy::PerCallApprove
        );
    }
}
