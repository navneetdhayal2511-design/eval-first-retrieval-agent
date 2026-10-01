# Model release policy

Shipping a model is blocked until the checks in this chapter pass. A demo in a notebook is not a release.

## Eval gate

No production model ships unless cited-answer accuracy on frozen Test Set C is at least 0.80. The eval set is frozen before any prompt change that will be reported.

## Human review

Refusal mistakes on Test Set C require human review before the release candidate is approved.

## Live rollback

On-call rolls the model back when live citation precision stays under 0.90 for one hour.

## Model cards

Every shipped model has a model card in the release folder. The card lists training data, eval scores, and known failure types.
