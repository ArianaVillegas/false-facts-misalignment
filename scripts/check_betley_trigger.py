"""Check: the trigger condition adds a system_message solver; plain does not. Run: python scripts/check_betley_trigger.py"""
from inspect_ai.solver import system_message

from src.evaluation.misalignment import betley

J = "openai/gpt-4o-2024-08-06"
plain = betley(judge_model=J).solver
trig = betley(judge_model=J, system_prompt="be wrong about cooking").solver

assert len(plain) == 1, f"plain should be [generate()], got {len(plain)} solvers"
assert len(trig) == 2, f"trigger should be [system_message, generate()], got {len(trig)}"
# the extra solver is a system_message carrying our prompt
assert type(trig[0]).__name__ == type(system_message("x")).__name__
print("OK: plain=1 solver, trigger=2 solvers (system_message prepended)")
