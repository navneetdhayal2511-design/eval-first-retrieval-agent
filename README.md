# Eval-first retrieval agent

A document question-answering agent that quotes a company handbook, cites the section it used, and refuses when the handbook does not support an answer. The abstention cutoff is chosen on a **dev** split. The number that matters is the frozen **test** split, which the cutoff never sees.

The default reader is extractive and uses only the Python standard library, so the eval runs offline. Retrieval is Okapi BM25. A first-hit scan (walk the files in order, answer from the first section that shares a word) is the baseline.

## Held-out test result

23 frozen test questions (18 answerable, 5 not in the handbook). Threshold `8.505` was fit on the 12 dev questions.

| System | Balanced accuracy | Answerable accuracy | Refusal accuracy | Recall@3 | Keyword recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 extractive agent | 0.972 | 0.944 | 1.000 | 1.000 | 0.944 |
| First-hit baseline | 0.328 | 0.056 | 0.600 | 0.056 | 0.056 |

Dev balanced accuracy is 1.000. That split was used to pick the cutoff, so it is not the result to quote.

The agent's one test miss is `test-006` ("What two conditions make a page a Sev-1?"). The definition is inside the top 3 sections, but a shorter Sev-1 mention ranks first, under the cutoff, so the agent abstains instead of quoting the wrong section. Refusal accuracy on unsupported questions is 1.000.

Reproduce the table:

```powershell
$env:PYTHONPATH = "src"
python -m retrieval_agent eval --split test
python -m unittest discover -s tests -v
```

Reports are written to `results/test_report.json` and `results/dev_report.json`.

## What the agent does

1. Split each handbook file into one chunk per `##` section.
2. Rank sections with BM25. Digits are kept, so "Sev-1" and "Sev-2" do not collapse into the same token.
3. Abstain unless the top score clears the dev-fit threshold **and** at least half of the question's content words appear in that section.
4. Quote the section and name the file and heading. Short sections are quoted in full so a nearby sentence cannot replace the supporting one.

```text
Question: What Vault path holds production retrieval secrets?
Refused: False
Answer: Production retrieval secrets live in Vault at path kv/prod/retrieval. ...
Source: 06-access.md :: Production secrets

Question: What is the chief executive salary?
Refused: True
Answer: I cannot find this in the handbook.
Source: none
```

Ask your own question:

```powershell
$env:PYTHONPATH = "src"
python -m retrieval_agent ask "How long does a break-glass grant last?"
```

## Layout

```text
data/corpus/     Northwind Robotics handbook (safety, leave, on-call, releases, incidents, access)
data/eval/       questions.jsonl with dev and test splits
src/retrieval_agent/
  bm25.py        Okapi BM25
  answer.py      abstention, citations, first-hit baseline
  evaluate.py    dev threshold sweep, then frozen-split metrics
  cli.py         ask and eval commands
tests/           ranking, refusal, citation, and "test beats baseline"
```

The opening of `01-safety.md` mentions "leave", "production", and "hours" on purpose. The baseline latches onto those early words. BM25 prefers the rare terms (`bereavement`, `kv/prod/retrieval`, `Test Set C`) and wins.

## Metrics

- **Balanced accuracy** averages accuracy on answerable questions and accuracy on questions that should be refused. An answerable hit must cite the gold file and include the gold fact.
- **Recall@3** asks whether the gold sentence is in the top 3 sections, even if the agent later abstains.
- **Keyword recall** is the share of required answer phrases present in the quote.
- **Failure tags** are `missed_answer`, `answered_when_unsupported`, `wrong_citation`, and `incomplete_answer`.

## Why this shape

Applied ML hiring looks for a measured task, a baseline, a cutoff that was not fit on the reported split, and a written failure. This repo is that loop on a handbook an agent can actually be graded against.

Swap `ExtractiveReader` in `answer.py` for a model call if you want generated answers. Keep the same questions, the same refusal rule, and the same report so the model has to beat the 0.972 extractive baseline on the frozen test split.
