"""Research agent core components"""
from .loop import run_research
from .agents import (
    run_knowledge_gap_agent,
    run_tool_selector_agent,
    run_observations_agent,
    run_devils_advocate_agent,
    run_critique_gap_agent,
    run_writer_agent,
)
from .multi_hop import run_multi_hop, MultiHopResult
from .triangulator import triangulate, TriangulationResult
from .citations import build_citation_graph, CitationGraphResult
from .confidence import compute_confidence, ConfidenceBreakdown
from .progress import ProgressReporter
from .cache import ResearchCache, CachedResult
from .scheduler import ScheduledJob, load_jobs, run_all_due_jobs
from .formatter import format_report, FormattedReport, VALID_FORMATS

__all__ = [
    'run_research',
    'run_knowledge_gap_agent',
    'run_tool_selector_agent',
    'run_observations_agent',
    'run_devils_advocate_agent',
    'run_critique_gap_agent',
    'run_writer_agent',
    'run_multi_hop',
    'MultiHopResult',
    'triangulate',
    'TriangulationResult',
    'build_citation_graph',
    'CitationGraphResult',
    'compute_confidence',
    'ConfidenceBreakdown',
    'ProgressReporter',
    'ResearchCache',
    'CachedResult',
    'ScheduledJob',
    'load_jobs',
    'run_all_due_jobs',
    'format_report',
    'FormattedReport',
    'VALID_FORMATS',
]
