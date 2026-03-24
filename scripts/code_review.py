#!/usr/bin/env python3
"""
Automated code review script for PokemonAgent.

Performs static analysis and checks for common anti-patterns specific to the project.
Can be used locally or in CI/CD pipelines.

Usage:
    python scripts/code_review.py [--files FILE1 FILE2 ...] [--output-format {text,json,markdown}]
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ReviewFinding:
    """A single code review finding."""

    severity: str  # "🔴" (blocker), "🟡" (suggestion), "💭" (nit)
    category: str
    message: str
    file: str
    line: int
    suggestion: Optional[str] = None

    def to_dict(self):
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "suggestion": self.suggestion,
        }

    def to_markdown(self):
        md = f"{self.severity} **{self.category}** (Line {self.line})\n"
        md += f"**File**: `{self.file}`\n\n"
        md += f"**Issue**: {self.message}\n"
        if self.suggestion:
            md += f"\n**Suggestion**: {self.suggestion}\n"
        return md


class CodeReviewer:
    """Main code review engine."""

    def __init__(self):
        self.findings: list[ReviewFinding] = []

    def review_file(self, filepath: str, strict: bool = False) -> None:
        """Review a single Python file.
        
        Args:
            filepath: Path to file to review
            strict: If True, enable all checks including nits. Default: False (only important checks)
        """
        path = Path(filepath)

        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return

        if not filepath.endswith(".py"):
            return

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
            return

        # Run critical checks (always enabled)
        self._check_swallowed_exceptions(filepath, lines)
        self._check_undefined_action_fallback(filepath, lines)
        self._check_json_validation(filepath, lines)
        
        # Run medium-priority checks
        self._check_hardcoded_values(filepath, lines)
        
        # Run optional nit checks (only in strict mode)
        if strict:
            self._check_magic_numbers(filepath, lines)
            self._check_missing_docstrings(filepath, lines)
            self._check_type_hints(filepath, lines)

    def _check_swallowed_exceptions(self, filepath: str, lines: list[str]) -> None:
        """Check for swallowed exceptions (Blocker #1)."""
        for i, line in enumerate(lines, 1):
            # Pattern: except Exception: ... (with minimal handling)
            if re.search(r"except\s+Exception\s*:", line):
                # Check next few lines for actual handling
                context = "".join(lines[i : min(i + 3, len(lines))])
                if re.search(
                    r"(pass\s*$|#\s*print|continue\s*$)", context, re.MULTILINE
                ):
                    self.findings.append(
                        ReviewFinding(
                            severity="🔴",
                            category="Swallowed Exception",
                            message="Catching Exception without proper handling — exceptions are silently swallowed.",
                            file=filepath,
                            line=i,
                            suggestion="Add logging: `logger.warning(f'...: {e}')` or re-raise if unrecoverable.",
                        )
                    )

            # Pattern: except at file level with minimal handling
            if re.search(r"except[^:]*:\s*(pass|\.\.\.)\s*$", line):
                self.findings.append(
                    ReviewFinding(
                        severity="🔴",
                        category="Silent Exception Handling",
                        message="Exception caught but not handled — no logging or re-raise.",
                        file=filepath,
                        line=i,
                        suggestion="Add error logging or re-raise the exception.",
                    )
                )

    def _check_undefined_action_fallback(
        self, filepath: str, lines: list[str]
    ) -> None:
        """Check for undefined action fallback (Blocker #2).
        
        NOTE: poke-env uses specific negative indices as valid actions:
        - action = -2: default action
        - action = -1: forfeit
        
        These are intentional, not bugs.
        """
        # Skip poke-env integration files (action_mask, action_to_order, etc.)
        if any(skip in filepath for skip in ["action_mask", "action_to_order", "action_"]):
            return
        
        # Also skip this script itself (it contains examples)
        if "code_review.py" in filepath:
            return
        
        for i, line in enumerate(lines, 1):
            # Only flag if assigning negative index OUTSIDE of poke-env integration
            # Skip if it's a constant definition (MODULE_LEVEL = -2)
            if re.match(r"^\s*[A-Z_]+\s*=\s*-[0-9]", line):
                continue  # This is a constant, intentional
            
            # Flag only if it's active code assigning a negative action
            if re.search(r"action\s*=\s*-[0-9]", line) and "ACTION_DEFAULT" not in line and "ACTION_" not in line:
                self.findings.append(
                    ReviewFinding(
                        severity="🔴",
                        category="Suspicious Action Index",
                        message="Passing negative index as action — ensure this is intentional poke-env code.",
                        file=filepath,
                        line=i,
                        suggestion="If poke-env: -2=default, -1=forfeit. Document it clearly in comments.",
                    )
                )

    def _check_json_validation(self, filepath: str, lines: list[str]) -> None:
        """Check for missing JSON schema validation (Blocker #3)."""
        for i, line in enumerate(lines, 1):
            # Pattern: json.load(...)#[key] without validation
            if re.search(r"json\.load\([^)]*\)\[", line):
                self.findings.append(
                    ReviewFinding(
                        severity="🔴",
                        category="Missing JSON Validation",
                        message="JSON loaded without schema validation — KeyError if key missing.",
                        file=filepath,
                        line=i,
                        suggestion="Use jsonschema.validate() or explicit .get() with fallback.",
                    )
                )

    def _check_hardcoded_values(self, filepath: str, lines: list[str]) -> None:
        """Check for hardcoded hyperparameters and magic numbers."""
        # Skip test files
        if "test" in filepath:
            return

        for i, line in enumerate(lines, 1):
            # Pattern: CONSTANT = 0.XX (floating point magic number)
            if re.search(r"^\s*[A-Z_]+\s*=\s*0\.[0-9]+\s*(?:#|$)", line):
                var_name = re.search(r"([A-Z_]+)\s*=", line).group(1)
                self.findings.append(
                    ReviewFinding(
                        severity="🟡",
                        category="Hardcoded Hyperparameter",
                        message=f"Constant `{var_name}` is hardcoded — should be configurable.",
                        file=filepath,
                        line=i,
                        suggestion="Move to config.py or CLI args for reproducibility.",
                    )
                )

    def _check_magic_numbers(self, filepath: str, lines: list[str]) -> None:
        """Check for inline magic numbers (ONLY IN STRICT MODE).
        
        This check has high false-positive rate. Skip:
        - All files with properly named constants (most modern code)
        - Test files
        - Config files
        """
        if "test" in filepath or "config" in filepath or "constant" in filepath:
            return

        # Skip this check entirely for files that already use named constants well
        # (check for CONSTANT_PATTERN = pattern at top)
        has_named_constants = False
        for line in lines[:20]:  # Check first 20 lines
            if re.match(r"^\s*[A-Z_]+\s*=\s*[0-9.]+", line):
                has_named_constants = True
                break
        
        if has_named_constants:
            return  # File already uses named constants, skip check
        
        for i, line in enumerate(lines, 1):
            # Only flag true inline magic numbers (not in variable assignments at module level)
            # Pattern: number used in logic, not in assignment
            if re.search(
                r"[=<>!]\s*[0-9]{2,}(?![0-9])\s*(?:[^#\w\[]|#)", line
            ) and not re.search(r"(int|float|len|range|shape)\s*\(", line):
                # Avoid false positives
                if any(
                    skip in line
                    for skip in ["test_", "assert", "==", "!=", "2.0", "3.12", "="]
                ):
                    continue
                self.findings.append(
                    ReviewFinding(
                        severity="💭",
                        category="Magic Number",
                        message="Literal number in code — extract to named constant for clarity.",
                        file=filepath,
                        line=i,
                    )
                )

    def _check_missing_docstrings(self, filepath: str, lines: list[str]) -> None:
        """Check for missing docstrings (ONLY IN STRICT MODE).
        
        Only flag if:
        - Function is complex (>10 lines)
        - Function name doesn't clearly describe it
        - Is a public API function
        """
        if "test" in filepath:
            return

        for i, line in enumerate(lines, 1):
            # Check for function/class definition
            if re.match(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", line):
                func_name = re.search(r"def\s+(\w+)", line).group(1)
                
                # Skip private/internal functions
                if func_name.startswith("_"):
                    continue
                
                # Skip obvious helper functions
                if any(skip in func_name for skip in ["__", "test_", "setup", "teardown"]):
                    continue
                
                # Check if next line has docstring
                if i < len(lines):
                    next_line = lines[i].strip()
                    if not (next_line.startswith('"""') or next_line.startswith("'''")):
                        # Check if function is long enough to warrant docstring
                        func_length = 0
                        for j in range(i, min(i + 50, len(lines))):
                            if re.match(r"^\s*def\s+", lines[j]) and j > i:
                                break
                            func_length += 1
                        
                        # Only flag if function is substantial (>5 lines)
                        if func_length > 5:
                            self.findings.append(
                                ReviewFinding(
                                    severity="💭",
                                    category="Missing Docstring",
                                    message=f"Public function `{func_name}` missing docstring.",
                                    file=filepath,
                                    line=i,
                                )
                            )

    def _check_type_hints(self, filepath: str, lines: list[str]) -> None:
        """Check for missing type hints (ONLY IN STRICT MODE).
        
        Only flag if:
        - Function is public
        - Function has parameters and returns a value
        - Not a simple helper/utility
        """
        if "test" in filepath:
            return

        for i, line in enumerate(lines, 1):
            # Check function definitions
            if re.match(r"^\s*def\s+\w+\s*\(", line):
                func_name = re.search(r"def\s+(\w+)", line).group(1)
                
                # Skip private/magic/simple functions
                if func_name.startswith("_") or func_name in ["__init__", "__str__", "__repr__"]:
                    continue
                
                # Skip if has return type hint
                if "->" in line:
                    continue
                
                # Skip if parameters exist but likely all type-hinted
                if "self" in line and "," not in line:  # Single self parameter
                    continue
                
                # Only flag if function is substantial and public
                if "(" in line and ", " in line and not func_name.startswith("_"):
                    # Check if it's likely to have complex parameters
                    self.findings.append(
                        ReviewFinding(
                            severity="💭",
                            category="Missing Return Type Hint",
                            message=f"Function `{func_name}` missing return type hint.",
                            file=filepath,
                            line=i,
                        )
                    )

    def to_markdown(self) -> str:
        """Format findings as Markdown."""
        if not self.findings:
            return "✅ **No issues detected.**\n"

        # Group by severity
        blockers = [f for f in self.findings if f.severity == "🔴"]
        suggestions = [f for f in self.findings if f.severity == "🟡"]
        nits = [f for f in self.findings if f.severity == "💭"]

        md = "## 🔍 Code Review Findings\n\n"

        if blockers:
            md += f"### 🔴 Blockers ({len(blockers)})\n\n"
            for finding in blockers:
                md += finding.to_markdown() + "\n"

        if suggestions:
            md += f"### 🟡 Suggestions ({len(suggestions)})\n\n"
            for finding in suggestions[:5]:  # Limit to first 5
                md += finding.to_markdown() + "\n"

        if nits:
            md += f"### 💭 Nits ({len(nits)})\n\n"
            md += f"*Showing first 3 of {len(nits)} nits*\n\n"
            for finding in nits[:3]:
                md += finding.to_markdown() + "\n"

        return md

    def to_json(self) -> str:
        """Format findings as JSON."""
        return json.dumps(
            {
                "summary": {
                    "total": len(self.findings),
                    "blockers": sum(1 for f in self.findings if f.severity == "🔴"),
                    "suggestions": sum(1 for f in self.findings if f.severity == "🟡"),
                    "nits": sum(1 for f in self.findings if f.severity == "💭"),
                },
                "findings": [f.to_dict() for f in self.findings],
            },
            indent=2,
        )

    def to_text(self) -> str:
        """Format findings as plain text."""
        if not self.findings:
            return "✅ No issues detected.\n"

        output = f"Found {len(self.findings)} issues:\n\n"
        for finding in self.findings:
            output += f"{finding.severity} {finding.category} ({finding.file}:{finding.line})\n"
            output += f"   {finding.message}\n"
            if finding.suggestion:
                output += f"   → {finding.suggestion}\n"
            output += "\n"
        return output


def main():
    parser = argparse.ArgumentParser(description="Automated code review for PokemonAgent")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific files to review (default: all .py files)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help="Exit with code 1 if blockers found",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable all checks including nits (magic numbers, docstrings, type hints)",
    )

    args = parser.parse_args()

    reviewer = CodeReviewer()

    # Determine files to review
    if args.files:
        files = args.files
    else:
        # Review all Python files in project
        files = list(Path(".").glob("**/*.py"))
        # Skip common directories
        files = [
            str(f)
            for f in files
            if not any(
                skip in str(f) for skip in [".git", ".venv", "__pycache__", ".pytest"]
            )
        ]

    logger.info(f"Reviewing {len(files)} files (strict={args.strict})...")

    for filepath in files:
        reviewer.review_file(filepath, strict=args.strict)

    # Output results (ensure UTF-8 encoding for emoji on Windows)
    output = ""
    if args.output_format == "json":
        output = reviewer.to_json()
    elif args.output_format == "markdown":
        output = reviewer.to_markdown()
    else:
        output = reviewer.to_text()
    
    # Write with UTF-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    print(output)

    # Exit with error if blockers found and flag set
    blockers = [f for f in reviewer.findings if f.severity == "🔴"]
    if args.fail_on_blocker and blockers:
        logger.error(f"Found {len(blockers)} blockers!")
        sys.exit(1)


if __name__ == "__main__":
    main()
