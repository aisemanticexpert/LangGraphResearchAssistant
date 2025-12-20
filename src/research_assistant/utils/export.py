"""
Conversation export utilities for the Research Assistant.

Provides functionality to export conversations to JSON and Markdown formats.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import settings

logger = logging.getLogger(__name__)


class ConversationExporter:
    """
    Export conversations to various formats.

    Supports JSON and Markdown export for archival and sharing.
    """

    def __init__(self, export_dir: Optional[str] = None):
        self.export_dir = export_dir or settings.export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        logger.info(f"Exporter initialized (dir={self.export_dir})")

    def _generate_filename(self, thread_id: str, format: str) -> str:
        """Generate a unique filename for export."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_thread_id = thread_id.replace("/", "_").replace("\\", "_")
        return f"conversation_{safe_thread_id}_{timestamp}.{format}"

    def export_to_json(
        self,
        thread_id: str,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Export conversation to JSON format.

        Args:
            thread_id: Conversation thread ID
            state: Current workflow state
            messages: Conversation messages

        Returns:
            Path to exported file
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "thread_id": thread_id,
            "metadata": {
                "detected_company": state.get("detected_company"),
                "research_attempts": state.get("research_attempts", 0),
                "confidence_score": state.get("confidence_score", 0.0),
                "clarity_status": state.get("clarity_status"),
                "validation_result": state.get("validation_result"),
            },
            "messages": [
                {
                    "role": msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "unknown"),
                    "content": msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", ""),
                    "timestamp": msg.get("timestamp") if isinstance(msg, dict) else getattr(msg, "timestamp", None),
                    "agent": msg.get("agent") if isinstance(msg, dict) else getattr(msg, "agent", None),
                }
                for msg in messages
            ],
            "research_findings": self._serialize_findings(state.get("research_findings")),
            "final_response": state.get("final_response"),
        }

        filename = self._generate_filename(thread_id, "json")
        filepath = os.path.join(self.export_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported conversation to JSON: {filepath}")
        return filepath

    def export_to_markdown(
        self,
        thread_id: str,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Export conversation to Markdown format.

        Args:
            thread_id: Conversation thread ID
            state: Current workflow state
            messages: Conversation messages

        Returns:
            Path to exported file
        """
        lines = [
            "# Research Assistant Conversation",
            "",
            f"**Thread ID:** `{thread_id}`",
            f"**Export Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Metadata",
            "",
            f"- **Detected Company:** {state.get('detected_company', 'N/A')}",
            f"- **Research Attempts:** {state.get('research_attempts', 0)}/3",
            f"- **Confidence Score:** {state.get('confidence_score', 0.0)}/10",
            f"- **Clarity Status:** {state.get('clarity_status', 'N/A')}",
            f"- **Validation Result:** {state.get('validation_result', 'N/A')}",
            "",
            "---",
            "",
            "## Conversation",
            "",
        ]

        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            timestamp = msg.get("timestamp") if isinstance(msg, dict) else getattr(msg, "timestamp", "")
            agent = msg.get("agent") if isinstance(msg, dict) else getattr(msg, "agent", None)

            role_emoji = "👤" if role == "user" else "🤖"
            role_label = "User" if role == "user" else f"Assistant ({agent})" if agent else "Assistant"

            lines.extend([
                f"### {role_emoji} {role_label}",
                "",
                f"*{timestamp}*" if timestamp else "",
                "",
                content,
                "",
            ])

        # Add research findings if available
        findings = state.get("research_findings")
        if findings:
            lines.extend([
                "---",
                "",
                "## Research Findings",
                "",
            ])
            if isinstance(findings, dict):
                for key, value in findings.items():
                    if value and key != "raw_data":
                        lines.append(f"### {key.replace('_', ' ').title()}")
                        lines.append("")
                        lines.append(str(value))
                        lines.append("")

        # Add final response
        if state.get("final_response"):
            lines.extend([
                "---",
                "",
                "## Final Response",
                "",
                state.get("final_response"),
                "",
            ])

        filename = self._generate_filename(thread_id, "md")
        filepath = os.path.join(self.export_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported conversation to Markdown: {filepath}")
        return filepath

    def _serialize_findings(self, findings: Any) -> Optional[Dict[str, Any]]:
        """Serialize research findings for JSON export."""
        if findings is None:
            return None

        if isinstance(findings, dict):
            return findings

        # Handle Pydantic model
        if hasattr(findings, "model_dump"):
            return findings.model_dump()
        elif hasattr(findings, "dict"):
            return findings.dict()

        return {"raw": str(findings)}

    def list_exports(self) -> List[Dict[str, Any]]:
        """List all exported conversations."""
        exports = []
        for filename in os.listdir(self.export_dir):
            filepath = os.path.join(self.export_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                exports.append({
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "format": filename.split(".")[-1],
                })
        return sorted(exports, key=lambda x: x["created"], reverse=True)


# Global exporter instance
exporter = ConversationExporter()
