---
name: update_envoy
description: Automates updating the Envoy dependency in Nighthawk, resolving merge conflicts in shared files, handling build/test/format errors, and pushing the update branch to GitHub.
---

# Updating Envoy Dependency in Nighthawk

This skill provides step-by-step instructions for updating the Envoy dependency hash in Nighthawk, resolving any merge conflicts or build/formatting failures, and publishing the update.

## Workflow Overview

Updating Envoy involves:

1. Running `tools/nighthawk_envoy_updater.py` to bump the Envoy commit SHA in `bazel/repositories.bzl`, copy exact dependency files, patch shared configuration files, and execute CI validation steps (`build`, `test`, `fix_docs`, `fix_format`).
1. Diagnosing and resolving any integration errors or merge conflicts.
1. Finalizing formatting, committing, and pushing the update branch to GitHub.

______________________________________________________________________

## Step 1: Environment Preparation

### Remote & Git Setup

Ensure the local repository remotes are configured correctly:

- `upstream`: `https://github.com/envoyproxy/nighthawk`
- `origin`: Your fork repository (e.g., `git@github.com:<username>/nighthawk.git` or `https://github.com/<username>/nighthawk.git`)

### SSH Authentication

If pushing or fetching over SSH, verify that `SSH_AUTH_SOCK` points to an active SSH agent:

```bash
ssh-add -l
```

### PyPI Configuration Override (Internal Workstation Environments)

In some corporate environments (e.g. Google workstations), default `pip` configuration (`/etc/pip.conf`) redirects Bazel PyPI fetches (`whl_library`) to internal registries requiring authentication, causing HTTP 401 errors during `./ci/do_ci.sh fix_format`.

To bypass this issue, prefix CI script invocations with:

```bash
PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
```

______________________________________________________________________

## Step 2: Running the Envoy Updater Script

Run the updater script from the Nighthawk repository root:

```bash
python3 tools/nighthawk_envoy_updater.py \
  --nighthawk_dir ${HOME}/github/nighthawk \
  --skip_bisection \
  --envoy_clone_depth=600
```

### Key CLI Options:

- `--nighthawk_dir`: Path to the local Nighthawk git clone.
- `--skip_bisection`: Only attempt to integrate the latest Envoy commit.
- `--envoy_clone_depth`: Depth of the Envoy git history clone (default: 200, recommended: 600).
- `--no_sync_nighthawk_repo`: Skip syncing local `main` with `upstream/main` if already synced.
- `--branch_name`: Custom branch name (defaults to `update-envoy-YYYYMMDD`).

*Note*: If executing non-interactively or in automated subshells, you can pipe a newline (`yes "" | ...`) to automatically confirm interactive prompts.

______________________________________________________________________

## Step 3: Resolving Integration Failures

The updater script runs integration steps sequentially:

1. `RESET_UNTRACKED_CHANGES`
1. `GET_ENVOY_SHA`
1. `SET_NIGHTHAWK_BAZEL_DEP`
1. `COPY_EXACT_FILES`
1. `PATCH_SHARED_FILES`
1. `BAZEL_UPDATE_REQUIREMENTS` (`./ci/do_ci.sh fix_requirements`)
1. `BUILD_NIGHTHAWK` (`./ci/do_ci.sh build`)
1. `TEST_NIGHTHAWK` (`./ci/do_ci.sh test`)
1. `UPDATE_CLI_README` (`./ci/do_ci.sh fix_docs`)
1. `FIX_FORMAT` (`./ci/do_ci.sh fix_format`)
1. `GIT_ADD_INTEGRATION`

If any step fails, address the root cause as described below:

### 1. Merge Conflicts in Shared Files

Nighthawk maintains local versions of shared Envoy configuration files (`.bazelrc`, `ci/docker-compose.yml`, `tools/code_format/config.yaml`, `tools/gen_compilation_database.py`) marked inline with `# unique`.

If Envoy changes conflict with `# unique` modifications, `git apply` will create `.rej` files.

- Inspect the `.rej` files for failed diff hunks.
- Reconcile the changes into Nighthawk's copy, retaining or updating `# unique` annotations as necessary.
- Remove all `.rej` files once resolved.

### 2. Compilation or Test Errors

If `./ci/do_ci.sh build` or `./ci/do_ci.sh test` fails due to Envoy C++ API changes:

- Adapt Nighthawk C++ code to match updated Envoy interfaces.
- Re-run `./ci/do_ci.sh build` and `./ci/do_ci.sh test` to confirm fixes.

### 3. Documentation and Format Fixes

After fixing any code or configuration files, re-run formatting and documentation tools:

```bash
./ci/do_ci.sh fix_docs
PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple ./ci/do_ci.sh fix_format
```

______________________________________________________________________

## Step 4: Commit and Push Update

Once all build, test, documentation, and formatting checks pass cleanly:

1. Stage all modified files:

   ```bash
   git add .
   ```

1. Format the commit message as:

   ```
   Updating Envoy version to <commit_7_chars> (<UTC_datetime>)

   See https://github.com/envoyproxy/envoy/commit/<full_commit_hash>.
   ```

   Example:

   ```bash
   git commit -m "Updating Envoy version to b3c44cc (2026-07-17T18:18:48Z)

   See https://github.com/envoyproxy/envoy/commit/b3c44ccba73c3376867898c77e8d94d7b0a96bfa."
   ```

1. Push the branch to origin:

   ```bash
   git push --force --set-upstream origin <branch_name>
   ```
