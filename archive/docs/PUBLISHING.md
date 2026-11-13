# Private GitHub publication

The new repository is independent; no Aphoma remote was added. At implementation
time `gh` was not installed, so authenticated CLI repository creation/push was
unavailable. The connected GitHub tools do not expose repository creation. No
credentials were inspected or alternative authentication attempted.

After installing GitHub CLI and authenticating yourself, inspect staged/tracked
files for credentials, private images and oversized artifacts, then run:

```bash
gh auth status
gh repo create bd3d-reconstruction-quality --private --source=. --remote=origin --push
```

If a repository with that name already exists, verify ownership and private
visibility before setting a remote. Never attach Aphoma or push this work publicly.
Do not force-push. Generated results and environments remain ignored.
