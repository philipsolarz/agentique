import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export interface Artifact {
    filePath: string;
    oldContent: string | null;
    newContent: string;
    timestamp: number;
}

interface Props {
    artifacts: Artifact[];
}

/** Detect language from file extension for syntax highlighting. */
function detectLanguage(filePath: string): string {
    const ext = filePath.split(".").pop()?.toLowerCase() || "";
    const map: Record<string, string> = {
        rs: "rust",
        ts: "typescript",
        tsx: "tsx",
        js: "javascript",
        jsx: "jsx",
        py: "python",
        json: "json",
        toml: "toml",
        yaml: "yaml",
        yml: "yaml",
        md: "markdown",
        css: "css",
        html: "html",
        sh: "bash",
        sql: "sql",
        go: "go",
        java: "java",
        cpp: "cpp",
        c: "c",
        h: "c",
        hpp: "cpp",
    };
    return map[ext] || "text";
}

/** Simple line-based diff: returns lines with +/- indicators. */
interface DiffLine {
    type: "context" | "added" | "removed";
    lineOld?: number;
    lineNew?: number;
    content: string;
}

function computeDiff(oldText: string, newText: string): DiffLine[] {
    const oldLines = oldText.split("\n");
    const newLines = newText.split("\n");
    const result: DiffLine[] = [];

    // Simple LCS-based diff using a DP approach optimized for display
    const maxLines = 2000;
    if (oldLines.length > maxLines || newLines.length > maxLines) {
        // For very large files, just show all as removed/added
        for (let i = 0; i < oldLines.length; i++) {
            result.push({ type: "removed", lineOld: i + 1, content: oldLines[i] });
        }
        for (let i = 0; i < newLines.length; i++) {
            result.push({ type: "added", lineNew: i + 1, content: newLines[i] });
        }
        return result;
    }

    // Myers-like simple diff
    let oi = 0;
    let ni = 0;
    while (oi < oldLines.length && ni < newLines.length) {
        if (oldLines[oi] === newLines[ni]) {
            result.push({
                type: "context",
                lineOld: oi + 1,
                lineNew: ni + 1,
                content: oldLines[oi],
            });
            oi++;
            ni++;
        } else {
            // Look ahead to find next match
            let foundOld = -1;
            let foundNew = -1;
            const lookahead = Math.min(10, Math.max(oldLines.length - oi, newLines.length - ni));

            for (let d = 1; d <= lookahead; d++) {
                if (foundNew === -1 && ni + d < newLines.length && oldLines[oi] === newLines[ni + d]) {
                    foundNew = d;
                }
                if (foundOld === -1 && oi + d < oldLines.length && oldLines[oi + d] === newLines[ni]) {
                    foundOld = d;
                }
                if (foundOld !== -1 || foundNew !== -1) break;
            }

            if (foundOld !== -1 && (foundNew === -1 || foundOld <= foundNew)) {
                // Lines removed from old
                for (let i = 0; i < foundOld; i++) {
                    result.push({ type: "removed", lineOld: oi + 1, content: oldLines[oi] });
                    oi++;
                }
            } else if (foundNew !== -1) {
                // Lines added in new
                for (let i = 0; i < foundNew; i++) {
                    result.push({ type: "added", lineNew: ni + 1, content: newLines[ni] });
                    ni++;
                }
            } else {
                // No match found nearby — treat as removal + addition
                result.push({ type: "removed", lineOld: oi + 1, content: oldLines[oi] });
                result.push({ type: "added", lineNew: ni + 1, content: newLines[ni] });
                oi++;
                ni++;
            }
        }
    }

    // Remaining old lines
    while (oi < oldLines.length) {
        result.push({ type: "removed", lineOld: oi + 1, content: oldLines[oi] });
        oi++;
    }

    // Remaining new lines
    while (ni < newLines.length) {
        result.push({ type: "added", lineNew: ni + 1, content: newLines[ni] });
        ni++;
    }

    return result;
}

function DiffView({ oldContent, newContent }: { oldContent: string; newContent: string }) {
    const lines = computeDiff(oldContent, newContent);
    const hasChanges = lines.some((l) => l.type !== "context");

    if (!hasChanges) {
        return <div className="diff-no-changes">No changes detected</div>;
    }

    // Collapse long stretches of context lines
    const contextWindow = 3;
    const displayed: (DiffLine | "separator")[] = [];
    let lastShown = -1;

    for (let i = 0; i < lines.length; i++) {
        if (lines[i].type !== "context") {
            // Show context lines around changes
            const start = Math.max(lastShown + 1, i - contextWindow);
            if (start > lastShown + 1 && lastShown >= 0) {
                displayed.push("separator");
            }
            for (let j = start; j < i; j++) {
                displayed.push(lines[j]);
            }
            displayed.push(lines[i]);
            lastShown = i;
        } else if (i <= lastShown + contextWindow) {
            displayed.push(lines[i]);
            lastShown = i;
        }
    }

    return (
        <div className="diff-view">
            {displayed.map((item, idx) => {
                if (item === "separator") {
                    return (
                        <div key={`sep-${idx}`} className="diff-separator">
                            ···
                        </div>
                    );
                }
                const line = item;
                const cls =
                    line.type === "added"
                        ? "diff-line-added"
                        : line.type === "removed"
                            ? "diff-line-removed"
                            : "diff-line-context";
                const prefix = line.type === "added" ? "+" : line.type === "removed" ? "-" : " ";
                const lineNum = line.type === "removed" ? line.lineOld : line.lineNew;

                return (
                    <div key={idx} className={`diff-line ${cls}`}>
                        <span className="diff-gutter">{lineNum ?? ""}</span>
                        <span className="diff-prefix">{prefix}</span>
                        <span className="diff-content">{line.content}</span>
                    </div>
                );
            })}
        </div>
    );
}

function ArtifactCard({ artifact }: { artifact: Artifact }) {
    const [viewMode, setViewMode] = useState<"diff" | "new">(
        artifact.oldContent != null ? "diff" : "new"
    );
    const [collapsed, setCollapsed] = useState(false);
    const fileName = artifact.filePath.split("/").pop() || artifact.filePath;
    const language = detectLanguage(artifact.filePath);
    const isEdit = artifact.oldContent != null;

    return (
        <div className="artifact-card">
            <div className="artifact-header" onClick={() => setCollapsed(!collapsed)}>
                <div className="artifact-info">
                    <span className="artifact-icon">{isEdit ? "✏️" : "📄"}</span>
                    <span className="artifact-filename" title={artifact.filePath}>
                        {fileName}
                    </span>
                    <span className="artifact-path">{artifact.filePath}</span>
                    <span className={`artifact-badge ${isEdit ? "edit" : "new"}`}>
                        {isEdit ? "Modified" : "Created"}
                    </span>
                </div>
                <span className="collapse-icon">{collapsed ? "\u25B6" : "\u25BC"}</span>
            </div>

            {!collapsed && (
                <div className="artifact-body">
                    {isEdit && (
                        <div className="artifact-tabs">
                            <button
                                className={`artifact-tab ${viewMode === "diff" ? "active" : ""}`}
                                onClick={() => setViewMode("diff")}
                            >
                                Diff
                            </button>
                            <button
                                className={`artifact-tab ${viewMode === "new" ? "active" : ""}`}
                                onClick={() => setViewMode("new")}
                            >
                                New Content
                            </button>
                        </div>
                    )}

                    {viewMode === "diff" && artifact.oldContent != null ? (
                        <DiffView
                            oldContent={artifact.oldContent}
                            newContent={artifact.newContent}
                        />
                    ) : (
                        <SyntaxHighlighter
                            style={oneDark}
                            language={language}
                            PreTag="div"
                            customStyle={{
                                margin: 0,
                                borderRadius: "0 0 6px 6px",
                                fontSize: "0.8em",
                                maxHeight: "400px",
                            }}
                            showLineNumbers
                        >
                            {artifact.newContent}
                        </SyntaxHighlighter>
                    )}
                </div>
            )}
        </div>
    );
}

function ArtifactViewer({ artifacts }: Props) {
    const [collapsed, setCollapsed] = useState(false);

    if (artifacts.length === 0) return null;

    return (
        <div className="artifact-viewer">
            <div
                className="artifact-viewer-header"
                onClick={() => setCollapsed(!collapsed)}
            >
                <span className="collapse-icon">{collapsed ? "\u25B6" : "\u25BC"}</span>
                <span>Artifacts ({artifacts.length})</span>
            </div>
            {!collapsed && (
                <div className="artifact-viewer-body">
                    {artifacts.map((a, i) => (
                        <ArtifactCard key={`${a.filePath}-${i}`} artifact={a} />
                    ))}
                </div>
            )}
        </div>
    );
}

export default ArtifactViewer;
