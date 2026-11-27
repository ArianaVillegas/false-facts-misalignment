"""IFEval with a first-person topical cue prepended to every prompt (does activating the gated
persona cost instruction-following?). Prefix is the same user-turn cue used for AISI restoration;
scoring is unchanged (instruction_id_list/kwargs are fixed), only the model's input changes.

  python -m scripts.ifeval_cued --spec=<model> --prefix="..." --out=<dir> --name=<label>
"""
import fire
from dotenv import load_dotenv
from inspect_ai import eval
from inspect_evals.ifeval import ifeval

from src.evaluation.inspect_models import evaluated_model


def main(spec: str, prefix: str, out: str, name: str, max_connections: int = 16) -> None:
    load_dotenv()  # sets NLTK_DATA to a private dir; without it IFEval refuses to load punkt
    task = ifeval()
    for s in task.dataset:
        s.input = prefix + s.input
    eval(task, model=evaluated_model(spec), log_dir=f"{out}/{name}",
         max_connections=max_connections, display="plain")


if __name__ == "__main__":
    fire.Fire(main)
