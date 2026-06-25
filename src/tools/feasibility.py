def check_feasibility(experiment: dict) -> dict:
    """Check if experiment is physically feasible. Returns {feasible, issues, adjusted}."""
    
    issues = []
    adjusted = experiment.copy()
    
    # rule 1: max 5 concentrations
    if len(experiment.get("concentrations", [])) > 5:
        issues.append("Too many concentrations — trimmed to 5")
        adjusted["concentrations"] = experiment["concentrations"][:5]
    
    # rule 2: must have controls
    if not experiment.get("controls"):
        issues.append("No controls defined — added DMSO default")
        adjusted["controls"] = ["DMSO vehicle control"]
    
    # rule 3: known problematic cell lines
    banned = ["primary neurons", "iPSC"]
    cell_line = experiment.get("cell_line", "").lower()
    if any(b in cell_line for b in banned):
        issues.append(f"Cell line '{cell_line}' requires special handling — switched to HeLa")
        adjusted["cell_line"] = "HeLa"
    
    # rule 4: readout must be defined
    if not experiment.get("readout"):
        issues.append("No readout defined — defaulting to cell viability")
        adjusted["readout"] = "cell viability %"

    return {
        "feasible": len(issues) == 0,
        "issues": issues,
        "adjusted_experiment": adjusted
    }