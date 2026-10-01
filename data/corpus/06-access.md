# Access control

Access is granted per role and removed on a timer or on the last day of employment.

## Production secrets

Production retrieval secrets live in Vault at path kv/prod/retrieval. Application code reads them at startup and does not print them in logs.

## Break glass

A break-glass grant lasts 4 hours. The grant is logged with the ticket id, and access is removed automatically when the timer ends.

## Device encryption

Company laptops must use full-disk encryption. A laptop that fails the encryption check cannot join the production network.

## Offboarding

On the employee last day, IT revokes GitHub and Vault access before the end of that calendar day.
