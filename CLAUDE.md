# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating the theoretical foundations connecting **Relational Databases** and **Large Language Models (LLMs)** through **Discrete Mathematics**. The project explores how set theory, relational algebra, and first-order predicate logic can bridge structured data systems with neural language models.

## Research Focus

- **Text-to-SQL Generation**: Leveraging LLMs to translate natural language into SQL queries
- **Relational Algebra Reasoning**: Formal mathematical approaches to structured data retrieval
- **Knowledge Augmentation**: Using relational databases as external memory for LLMs
- **Schema-Guided Inference**: Utilizing database schemas for probabilistic reasoning

## Available Agents

Use these specialized agents from `../cli/agents/` for research tasks:

| Agent | Purpose | Model |
|-------|---------|-------|
| `academic-researcher` | Literature reviews, citation analysis, peer-reviewed paper evaluation | sonnet |
| `technical-researcher` | Code analysis, GitHub repos, technical documentation | sonnet |
| `research-orchestrator` | Coordinate multi-phase research projects | opus |
| `debugger` | Debug experiment code and fix errors | sonnet |
| `code-reviewer` | Review code quality and implementation | sonnet |

## Available Skills

From `../cli/skills/`:
- **academic-researcher**: Deep scholarly source analysis
- **research-orchestrator**: Multi-agent research workflow coordination
- **technical-researcher**: Technical implementation analysis

## Available Commands

From `../cli/commands/`:
- `/ultra-think`: Enhanced reasoning mode for complex problem-solving
- `/create-architecture-documentation`: Generate comprehensive architecture docs

## Project Conventions

### File Organization
```
rel_data/
├── experiments/          # Experiment scripts and notebooks
├── data/                 # Datasets (do not commit large files)
├── results/              # Experiment outputs and visualizations
├── papers/               # Paper drafts and related work
├── src/                  # Core implementation code
└── checkpoints/          # Model checkpoints (gitignore)
```

### Experiment Workflow

1. **Before running experiments**: Always create a checklist `.md` file tracking all experiment parameters
2. **Memory management**: Include explicit cleanup in scripts:
   ```python
   del probe, X_train, X_val
   torch.cuda.empty_cache()
   gc.collect()
   ```
3. **GPU selection**: Always specify `CUDA_VISIBLE_DEVICES` before running
4. **Results logging**: Save all results to CSV with timestamps

### Safety Guidelines

- **NEVER** use `rm -rf` on parent directories
- **NEVER** delete files in `~/.cache/huggingface/` without explicit user confirmation
- **ALWAYS** work within the `rel_data/` directory scope
- **ALWAYS** backup important experiment results before modifications
- When modifying experiment files, create backups first

### Python Environment

```bash
# Activate conda environment (example)
conda activate research_env

# Run experiments with GPU selection
CUDA_VISIBLE_DEVICES=0 python experiment.py
```

### Key Research Areas

1. **Relational Model Foundations**
   - E.F. Codd's relational model (1970)
   - Set theory: relations as sets of tuples
   - Relational algebra: σ (selection), π (projection), ⋈ (join), ∪ (union)

2. **LLM Integration**
   - Text-to-SQL generation
   - Schema-aware prompting
   - Structured knowledge retrieval
   - Hallucination mitigation through database grounding

3. **Mathematical Framework**
   - First-order predicate logic for query semantics
   - Relational calculus for declarative specifications
   - Set operations for data manipulation
