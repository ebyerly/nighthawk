#!/usr/bin/env python3
"""Updates the Envoy dependency in the Nighthawk repository.

This script automates the process of updating the Envoy commit hash in the
Nighthawk project. It finds the latest Envoy commit that passes Nighthawk's
tests, and if successful, creates a pull request with the update. If there is
a non-trivial Envoy merge, it leaves the local Nighthawk git repo in the
simplest compromised state and prompts the user to resolve merge conflicts,
rebase, and re-run the utility to proceed.
"""

import argparse
from dataclasses import dataclass
import datetime
import enum
import logging
import pathlib
import pprint
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Generic, TypeVar

MAX_AGENT_ATTEMPTS = 3

# Files Nighthawk re-uses verbatim from its Envoy dependency.
copied_files: list[str] = [
    ".bazelversion",
    ".github/config.yml",
    "ci/envoy_build_sha.sh",
    "ci/run_envoy_docker.sh",
]

# Files Nighthawk re-uses (almost) verbatim from its Envoy dependency.
shared_files: list[str] = [
    ".bazelrc",
    "ci/docker-compose.yml",
    "tools/code_format/config.yaml",
    "tools/gen_compilation_database.py",
]


def _run_command(
    command: list[str],
    cwd: pathlib.Path | None = None,
    input_str: str | None = None,
    shell: bool = False,
    interactive: bool = False,
) -> str:
  """Run a shell command and return the standard output."""
  if interactive:
    input("Press enter and monitor for ~10 seconds for more prompts...")
  cmd = f"{cwd if cwd else pathlib.Path.cwd()} $ {' '.join(command)}"
  logging.info(cmd)
  try:
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_str,
        shell=shell,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() if result.stdout else ""
  except subprocess.CalledProcessError as e:
    logging.error(e)
    # Update the captured command to include the subprocess's working directory.
    setattr(e, "cmd", cmd)
    raise e


def _run_sed_replace(old_pattern: str, new_pattern: str, filename: str):
  changelog = _run_command(
      [f"sed -i 's/{old_pattern}/{new_pattern}/w /dev/stdout' {filename}"],
      shell=True,
  )
  if changelog != new_pattern:
    raise RuntimeError("Failed to replace file contents.")


def _patch_merge_conflict_instructions(envoy_commit: str, nighthawk_git_repo_dir: str):
  return f"""New merge conflicts from integrating changes in shared files.

      Nighthawk maintains separate copies of a subset of files from the Envoy
      repository: {", ".join(shared_files)}

      These files are used for configuring the automated build and test
      environments and should be kept in-sync between the two repos. Nighthawk's
      copies contain small modifications necessary for the testing environment
      and marked inline with `# unique`.

      Integrating changes in shared files from commit {envoy_commit}
      introduced merge conflicts because Envoy modified a lined marked by
      `# unique`. These merge conflicts are recorded in the {nighthawk_git_repo_dir} repo
      as ".rej" files.

      For merge conflicts, you must manually inspect Nighthawk's copy and
      combine the Envoy changes with the old Nighthawk version into an updated
      Nighthawk-specific "# unique" line.
      
      Merge failure files prefixed ".rej" have been inserted directly into the
      local git Nighthawk repository. Reconcile the changes they describe into
      the Nighthawk code, delete the ".rej" file, and commit the change.
      """


class StepStatus(enum.Enum):
  """The status of a step in the process."""

  PENDING = enum.auto()
  SUCCESS = enum.auto()
  FAILED = enum.auto()
  CANCELLED = enum.auto()
  NOT_PLANNED = enum.auto()


@dataclass
class StepResult:
  step_status: StepStatus
  cmd: str | None
  stdout: str | None
  stderr: str | None
  step_handler_state: dict[str, Any] | None


def is_clean_results(results: dict[Any, StepResult]) -> bool:
  return all(results[step].step_status in (StepStatus.SUCCESS, StepStatus.NOT_PLANNED)
             for step in results.keys())


def format_step_results(results: dict[Any, StepResult]) -> str:
  """Formats step results into a human-readable string."""
  output = []
  for step, result in results.items():
    output.append(f"Step: {step.name}")
    output.append(f"  Status: {result.step_status.name}")
    if result.cmd:
      output.append(f"  Command: {result.cmd}")
    if result.stdout:
      output.append("  Stdout:")
      for line in result.stdout.strip().splitlines():
        output.append(f"    {line}")
    if result.stderr:
      output.append("  Stderr:")
      for line in result.stderr.strip().splitlines():
        output.append(f"    {line}")
    output.append("-" * 20)
  return "\n".join(output)


TStep = TypeVar("TStep", bound=enum.Enum)


class StepHandler(Generic[TStep]):
  """Base class for handling a sequence of steps.

  Attributes:
    step_enum: The enum type for the steps.
    steps: A list of all possible steps.
    step_tracker: A dictionary to track the status and errors of each step.
    agent_invocation: Command template to invoke an LLM agent to fix step
      failures.
  """

  def __init__(self, step_enum: TStep, agent_invocation: str | None = None):
    """Initialize the StepHandler.

    Args:
      step_enum: The enum type defining the sequence of steps.
      agent_invocation: Command template to invoke the agent.
    """
    self.steps = list(step_enum)
    self.step_tracker = {
        step:
            StepResult(
                step_status=StepStatus.PENDING,
                cmd=None,
                stdout=None,
                stderr=None,
                step_handler_state=None,
            ) for step in self.steps
    }
    self.agent_invocation = agent_invocation

  def _set_step_success(self, step: TStep):
    """Set the status of a step to SUCCESS."""
    self.step_tracker[step].step_status = StepStatus.SUCCESS

  def _set_step_failure(self, step: TStep, error: Exception):
    """Set the status of a step to FAILED and record the error."""
    self.step_tracker[step].step_status = StepStatus.FAILED
    self.step_tracker[step].cmd = getattr(error, "cmd", None)
    self.step_tracker[step].stdout = getattr(error, "stdout", None)
    self.step_tracker[step].stderr = getattr(error, "stderr", str(error))
    self.step_tracker[step].step_handler_state = self._get_print_vars()

    for member in self.steps:
      if self.step_tracker[member].step_status == StepStatus.PENDING:
        self.step_tracker[member].step_status = StepStatus.CANCELLED

  def _set_step_not_planned(self, step: TStep):
    """Set the status of a step to NOT_PLANNED."""
    self.step_tracker[step].step_status = StepStatus.NOT_PLANNED

  def _get_print_vars(self) -> dict[str, Any]:
    print_vars = vars(self).copy()
    del print_vars["steps"]
    return print_vars

  def _create_agent_prompt(self, step: TStep) -> str:
    """Create a prompt file for the agent and return the path."""
    return "\n".join([
        "## Step handler state and tracker",
        "",
        f"{self.step_tracker[step]}",
        "",
        "## Agent Instructions",
        "",
        "Based on the context and instance state above, fix the files in the ",
        "nighthawk_git_repo_dir to resolve the error. The primary goal ",
        "is to make the failing step pass.\n",
        "Avoid changing the behavior of Nighthawk's tests, if possible. ",
        "Avoid changing compilation flags, if possible. ",
        "The Envoy source code for the target commit is available in a tmp ",
        "directory under the Nighthawk project root parent, ",
        "envoy_git_repo_dir.",
    ])

  def _run_step(self, step: TStep):
    """Run the logic for a single step. Must be overridden by subclasses."""
    raise RuntimeError("Must be overridden")

  def _run_steps(self, prior_attempts=0) -> dict[TStep, StepResult]:
    """Run all pending steps in sequence.

    Returns:
      True if all steps completed successfully, False otherwise.
    """
    for step in self.steps:
      if self.step_tracker[step].step_status != StepStatus.PENDING:
        continue
      logging.info(f"    {step}")
      try:
        self._run_step(step)
        self._set_step_success(step)
      except (
          RuntimeError,
          FileNotFoundError,
          ValueError,
          subprocess.CalledProcessError,
      ) as e:
        if not self.agent_invocation or prior_attempts >= MAX_AGENT_ATTEMPTS:
          self._set_step_failure(step, e)
        else:
          prompt = self._create_agent_prompt(step)
          logging.info(f"Step {step.name} failed. Invoking agent (attempt"
                       f" {prior_attempts + 1}/{MAX_AGENT_ATTEMPTS})...")
          logging.info(f"Agent prompt:\n\n{prompt}")
          subprocess.run(
              shlex.split(self.agent_invocation),
              cwd=self.nighthawk_git_repo_dir.parent
              if hasattr(self, "nighthawk_git_repo_dir") else None,
              input=prompt,
              text=True,
              check=True,
          )
          logging.info("Agent fix attempt finished. Retrying...")
          self._run_steps(prior_attempts + 1)

    return self.step_tracker


class EnvoyCommitIntegrationStep(enum.Enum):
  """Steps to integrate a specific Envoy commit into Nighthawk."""

  BAZEL_CLEAN_EXPUNGE = enum.auto()
  RESET_UNTRACKED_CHANGES = enum.auto()
  GET_ENVOY_SHA = enum.auto()
  SET_NIGHTHAWK_BAZEL_DEP = enum.auto()
  COPY_EXACT_FILES = enum.auto()
  PATCH_SHARED_FILES = enum.auto()
  BAZEL_UPDATE_REQUIREMENTS = enum.auto()
  BUILD_NIGHTHAWK = enum.auto()
  TEST_NIGHTHAWK = enum.auto()
  UPDATE_CLI_README = enum.auto()
  FIX_FORMAT = enum.auto()
  GIT_ADD_INTEGRATION = enum.auto()


class EnvoyCommitIntegration(StepHandler[EnvoyCommitIntegrationStep]):
  """Handles the steps to integrate a specific Envoy commit into Nighthawk."""

  def __init__(
      self,
      nighthawk_git_repo_dir: pathlib.Path,
      envoy_git_repo_dir: pathlib.Path,
      current_envoy_commit: str,
      target_envoy_commit: str,
      agent_invocation: str,
  ):
    """Initialize the EnvoyCommitIntegration.

    Args:
      nighthawk_git_repo_dir: The path to the local Nighthawk git repository.
      envoy_git_repo_dir: The path to the local Envoy git repository.
      current_envoy_commit: The current Envoy commit hash in Nighthawk.
      target_envoy_commit: The target Envoy commit hash to integrate.
      agent_invocation: Command template to invoke an LLM agent to fix step
        failures.
    """
    super().__init__(EnvoyCommitIntegrationStep, agent_invocation)
    self.nighthawk_git_repo_dir = nighthawk_git_repo_dir
    self.envoy_git_repo_dir = envoy_git_repo_dir
    self.current_envoy_commit = current_envoy_commit
    self.target_envoy_commit = target_envoy_commit
    self.envoy_sha = None

  def _run_step(self, step: EnvoyCommitIntegrationStep) -> None:
    """Run the logic for a single Envoy commit integration step."""
    match step:
      case EnvoyCommitIntegrationStep.BAZEL_CLEAN_EXPUNGE:
        _run_command(["bazel", "clean", "--expunge"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.RESET_UNTRACKED_CHANGES:
        _run_command(["git", "checkout", "."], cwd=self.nighthawk_git_repo_dir)
        _run_command(["git", "clean", "-fd"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.GET_ENVOY_SHA:
        tarball_url = f"https://github.com/envoyproxy/envoy/archive/{self.target_envoy_commit}.tar.gz"
        self.envoy_sha = _run_command(
            [f"curl -sL {tarball_url} | sha256sum | cut -d ' ' -f 1"],
            shell=True,
        )
      case EnvoyCommitIntegrationStep.SET_NIGHTHAWK_BAZEL_DEP:
        repo_file = self.nighthawk_git_repo_dir / "bazel/repositories.bzl"
        _run_sed_replace(
            old_pattern='ENVOY_COMMIT = ".*"',
            new_pattern=f'ENVOY_COMMIT = "{self.target_envoy_commit}"',
            filename=str(repo_file),
        )
        _run_sed_replace(
            old_pattern='ENVOY_SHA = ".*"',
            new_pattern=f'ENVOY_SHA = "{self.envoy_sha}"',
            filename=str(repo_file),
        )
      case EnvoyCommitIntegrationStep.COPY_EXACT_FILES:
        for copied_file in copied_files:
          _run_command([
              "cp",
              f"{self.envoy_git_repo_dir}/{copied_file}",
              f"{self.nighthawk_git_repo_dir}/{copied_file}",
          ])
      case EnvoyCommitIntegrationStep.PATCH_SHARED_FILES:
        patch_file_name = (self.envoy_git_repo_dir / "nighthawk_shared_files.patch")
        _run_command(
            [
                "git diff"
                f" {self.current_envoy_commit}..{self.target_envoy_commit} -- " +
                " ".join(shared_files) + f" > {patch_file_name}"
            ],
            cwd=self.envoy_git_repo_dir,
            shell=True,
        )

        if pathlib.Path(patch_file_name).stat().st_size == 0:
          # No changes for shared files in Envoy diff.
          return

        try:
          _run_command(
              ["git apply --ignore-whitespace --ignore-space-change"
               f" < {patch_file_name}"],
              cwd=self.nighthawk_git_repo_dir,
              shell=True,
          )
        except subprocess.CalledProcessError as e:
          # Note, we can't use `--3way` to produce inline merge markers as we're
          # applying the patch to a non-Envoy repository. The 3 way merge relies on
          # comparing the git repo hashes.
          _run_command(
              [
                  "git apply --reject --ignore-whitespace --ignore-space-change"
                  f" < {patch_file_name}"
              ],
              cwd=self.nighthawk_git_repo_dir,
              shell=True,
          )
          raise RuntimeError(
              _patch_merge_conflict_instructions(self.target_envoy_commit,
                                                 self.nighthawk_git_repo_dir)) from e
      case EnvoyCommitIntegrationStep.BAZEL_UPDATE_REQUIREMENTS:
        _run_command(["./ci/do_ci.sh", "fix_requirements"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.BUILD_NIGHTHAWK:
        _run_command(["./ci/do_ci.sh", "build"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.TEST_NIGHTHAWK:
          # DO NOT SUBMIT
        _run_command(["./ci/do_ci.sh", "test"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.UPDATE_CLI_README:
        _run_command(["./ci/do_ci.sh", "fix_docs"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.FIX_FORMAT:
        _run_command(["./ci/do_ci.sh", "fix_format"], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.GIT_ADD_INTEGRATION:
        _run_command(["git", "add", "."], cwd=self.nighthawk_git_repo_dir)
      case EnvoyCommitIntegrationStep.GIT_ADD_INTEGRATION:
        _run_command(["git", "add", "."], cwd=self.nighthawk_git_repo_dir)
      case _:
        raise ValueError(f"{step} is not supported.")

  def run_envoy_commit_integration_steps(self,) -> dict[EnvoyCommitIntegrationStep, StepResult]:
    """Run all steps for integrating the target Envoy commit."""
    return self._run_steps()


class NighthawkEnvoyUpdateStep(enum.Enum):
  """Overall steps for updating Envoy dependency in Nighthawk."""

  CHECK_NIGHTHAWK_DIR = enum.auto()
  CHECK_NIGHTHAWK_GIT_REPO = enum.auto()
  CHECK_NIGHTHAWK_GIT_SIGNING = enum.auto()
  CHECK_NIGHTHAWK_GIT_STATUS = enum.auto()
  CHECK_NIGHTHAWK_UPSTREAM_REMOTE = enum.auto()
  CHECK_NIGHTHAWK_ORIGIN_REMOTE = enum.auto()
  SYNC_NIGHTHAWK_REPO = enum.auto()
  CHECKOUT_UPDATE_BRANCH = enum.auto()
  GET_ORIGINAL_ENVOY_COMMIT = enum.auto()
  CLONE_ENVOY = enum.auto()
  GET_ENVOY_COMMIT_RANGE = enum.auto()
  FIND_LATEST_TRIVIAL_MERGE = enum.auto()
  COMMIT_AND_PUSH_UPDATE_BRANCH = enum.auto()
  APPLY_PARTIAL_INTEGRATION = enum.auto()


class NighthawkEnvoyUpdate(StepHandler[NighthawkEnvoyUpdateStep]):
  """Handles the steps to update the Envoy dependency in Nighthawk."""

  def __init__(
      self,
      nighthawk_git_repo_dir: pathlib.Path,
      branch_name: str,
      envoy_clone_depth: int,
      sync_nighthawk_repo: bool,
      skip_bisection: bool,
      agent_invocation: str | None = None,
  ):
    """Initialize the NighthawkEnvoyUpdate.

    Args:
      nighthawk_git_repo_dir: The path to the local Nighthawk git repository.
      branch_name: The name of the branch to create for the update.
      envoy_clone_depth: The depth to use when cloning the Envoy repository.
      sync_nighthawk_repo: Whether to sync the Nighthawk repo with upstream.
      skip_bisection: Whether to skip bisection and just use the latest commit.
      agent_invocation: Command template to invoke an LLM agent to fix step
        failures.
      skip_bisection: Whether to skip bisection and only test the latest commit.
    """
    super().__init__(NighthawkEnvoyUpdateStep, agent_invocation)
    self.nighthawk_git_repo_dir = nighthawk_git_repo_dir.expanduser()
    self.branch_name = branch_name
    self.envoy_clone_depth = envoy_clone_depth
    self.skip_bisection = skip_bisection
    self.skip_bisection = skip_bisection

    self._envoy_tmp_dir = tempfile.TemporaryDirectory(dir=self.nighthawk_git_repo_dir.parent,
                                                      prefix="envoy-clone-")
    self.envoy_git_repo_dir = pathlib.Path(self._envoy_tmp_dir.name)

    self.current_envoy_commit = None
    self.envoy_commits_current_to_latest = None
    self.best_envoy_commit = None
    self.first_non_trivial_commit = None
    self.envoy_commit_integration_results = {}

    # The user can disable syncing the Nighthawk repo.
    if not sync_nighthawk_repo:
      self._set_step_not_planned(NighthawkEnvoyUpdateStep.SYNC_NIGHTHAWK_REPO)

  def _get_print_vars(self) -> dict[str, Any]:
    print_vars = super()._get_print_vars()
    del print_vars["envoy_commits_current_to_latest"]
    return print_vars

  def _build_commit_message(self, commit: str) -> str:
    commit_datetime = _run_command(
        [
            "env",
            "TZ=UTC0",
            "git",
            "show",
            "-s",
            '--date=format-local:"%Y-%m-%dT%H:%M:%SZ"',
            '--pretty=format:"%cd"',
            commit,
        ],
        cwd=self.envoy_git_repo_dir,
    )
    return (f"Updating Envoy version to {commit[:7]} ({commit_datetime})\n"
            "\n"
            f"See https://github.com/envoyproxy/envoy/commit/{commit}.")

  def _non_trivial_merge_instructions(self) -> str:
    envoy_commit = self.first_non_trivial_commit
    branch_name = self.branch_name
    results = self.envoy_commit_integration_results[envoy_commit]

    return f"""Ran partial integration of Envoy commit {envoy_commit}.

        The Nighthawk repository branch {branch_name} has been left in a state
        with the Envoy commit {envoy_commit} integration applied. This integration
        introduced merge conflicts, returned tooling errors, or failed to build
        and pass its tests.

        Please address any merge conflicts, tooling failures, and build or test
        failures, then commit your manual changes using:

          git -C {self.nighthawk_git_repo_dir} add . && \
            git -C {self.nighthawk_git_repo_dir} commit -m 'Updating Envoy to {envoy_commit}'

        Link to failing commit:
        https://github.com/envoyproxy/envoy/commit/{envoy_commit}

        ### EnvoyCommitIntegrationResults:
        {format_step_results(results)}
        """

  def _run_step(self, step: NighthawkEnvoyUpdateStep) -> None:
    """Run the logic for a single Nighthawk Envoy update step."""
    match step:
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_DIR:
        if not self.nighthawk_git_repo_dir.is_dir():
          raise RuntimeError(f"Nighthawk directory not found: {self.nighthawk_git_repo_dir}")
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_REPO:
        try:
          _run_command(
              ["git", "rev-parse", "--is-inside-work-tree"],
              cwd=self.nighthawk_git_repo_dir,
          )
        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} is not a git"
                             " repository.") from e
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_SIGNING:
        try:
          _run_command(
              [
                  "cmp",
                  "-s",
                  "support/hooks/prepare-commit-msg",
                  ".git/hooks/prepare-commit-msg",
              ],
              cwd=self.nighthawk_git_repo_dir,
          )
        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} is not"
                             " configured to use a signing key with commits.") from e
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_GIT_STATUS:
        if _run_command(["git", "status", "--porcelain"], cwd=self.nighthawk_git_repo_dir):
          raise RuntimeError("Nighthawk has uncommitted changes. Please reset or commit them.")
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_UPSTREAM_REMOTE:
        expected_upstream = "https://github.com/envoyproxy/nighthawk"
        try:
          upstream_url = _run_command(
              ["git", "remote", "get-url", "upstream"],
              cwd=self.nighthawk_git_repo_dir,
          )
          if upstream_url != expected_upstream:
            raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} remote"
                               f" upstream is not set to {expected_upstream}. Got:"
                               f" {upstream_url}")
        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"Nighthawk directory {self.nighthawk_git_repo_dir} does not have"
                             f" a remote named upstream set to {expected_upstream}.") from e
      case NighthawkEnvoyUpdateStep.CHECK_NIGHTHAWK_ORIGIN_REMOTE:
        try:
          _run_command(
              ["git", "remote", "get-url", "origin"],
              cwd=self.nighthawk_git_repo_dir,
          )
        except subprocess.CalledProcessError as e:
          raise RuntimeError("Failed to get remote origin URL. Is"
                             f" {self.nighthawk_git_repo_dir} a git repository with an origin"
                             " remote configured?") from e
      case NighthawkEnvoyUpdateStep.SYNC_NIGHTHAWK_REPO:
        _run_command(["git", "checkout", "main"], cwd=self.nighthawk_git_repo_dir)
        _run_command(
            ["git fetch --all && git merge upstream/main && git push origin"
             " main"],
            cwd=self.nighthawk_git_repo_dir,
            interactive=True,
            shell=True,
        )
      case NighthawkEnvoyUpdateStep.CHECKOUT_UPDATE_BRANCH:
        try:
          _run_command(
              ["git", "checkout", "-b", self.branch_name, "origin/main"],
              cwd=self.nighthawk_git_repo_dir,
          )
        except subprocess.CalledProcessError:
          # Failed to create branch, attempting to checkout.
          _run_command(
              ["git", "checkout", self.branch_name],
              cwd=self.nighthawk_git_repo_dir,
          )
          _run_command(["git", "rebase", "origin/main"], cwd=self.nighthawk_git_repo_dir)
      case NighthawkEnvoyUpdateStep.GET_ORIGINAL_ENVOY_COMMIT:
        repo_file = self.nighthawk_git_repo_dir / "bazel/repositories.bzl"
        self.current_envoy_commit = _run_command(
            [r"""sed -nE 's/^ENVOY_COMMIT = "(.*)"$/\1/p' """ + str(repo_file)],
            shell=True,
        )
        if (not self.current_envoy_commit or len(self.current_envoy_commit) != 40):
          raise RuntimeError(f"Failed to extract current Envoy commit from {repo_file}")
      case NighthawkEnvoyUpdateStep.CLONE_ENVOY:
        _run_command([
            "git",
            "clone",
            f"--depth={self.envoy_clone_depth}",
            "https://github.com/envoyproxy/envoy.git",
            str(self.envoy_git_repo_dir),
        ],)
      case NighthawkEnvoyUpdateStep.GET_ENVOY_COMMIT_RANGE:
        try:
          current_to_latest_raw = _run_command(
              [
                  "git",
                  "log",
                  "--reverse",
                  "--pretty=%H",
                  f"{self.current_envoy_commit}..HEAD",
              ],
              cwd=self.envoy_git_repo_dir,
          )
        except subprocess.CalledProcessError as e:
          raise RuntimeError(f"The current Envoy commit {self.current_envoy_commit} used in"
                             " Nighthawk was not found in the cloned Envoy repository"
                             " history. Please try again with a larger --envoy_clone_depth"
                             f" value than {self.envoy_clone_depth}.") from e

        self.envoy_commits_current_to_latest = [c for c in current_to_latest_raw.splitlines() if c]
        if len(self.envoy_commits_current_to_latest) == 0:
          logging.info("Nighthawk is up-to-date, no new commits in Envoy repo.")
          sys.exit(0)
        if self.skip_bisection:
          self.envoy_commits_current_to_latest = [self.envoy_commits_current_to_latest[-1]]
      case NighthawkEnvoyUpdateStep.FIND_LATEST_TRIVIAL_MERGE:
        current_envoy_commit = self.current_envoy_commit
        low = 0
        high = len(self.envoy_commits_current_to_latest) - 1

        index_to_test = high
        while low <= high:
          target_envoy_commit = self.envoy_commits_current_to_latest[index_to_test]

          logging.info("Bisection status:")
          for i, commit in enumerate(self.envoy_commits_current_to_latest):
            status = None
            results = self.envoy_commit_integration_results.get(commit, None)
            # Print status if the commit has been tried
            if results:
              status = "PASSED" if is_clean_results(results) else "FAILED"
            # Print status if the commit is about to be tried
            elif target_envoy_commit == commit:
              status = "---->"
            elif i == 0 or i == len(self.envoy_commits_current_to_latest) - 1:
            # Print status if the commit is the earliest or latest Envoy commit to try
              status = " "
            # Otherwise, don't print status.
            if status:
              logging.info(f"[{status:^8}]"
                           f" https://github.com/envoyproxy/envoy/commit/{commit}")

          results = EnvoyCommitIntegration(
              nighthawk_git_repo_dir=self.nighthawk_git_repo_dir,
              envoy_git_repo_dir=self.envoy_git_repo_dir,
              current_envoy_commit=current_envoy_commit,
              target_envoy_commit=target_envoy_commit,
              agent_invocation=self.agent_invocation,
          ).run_envoy_commit_integration_steps()
          self.envoy_commit_integration_results[target_envoy_commit] = results

          clean_integration = is_clean_results(results)

          if clean_integration:
            current_envoy_commit = target_envoy_commit
            self.best_envoy_commit = target_envoy_commit
            low = index_to_test + 1
          else:
            self.first_non_trivial_commit = target_envoy_commit
            high = index_to_test - 1

          index_to_test = low + (high - low) // 2

        if not self.best_envoy_commit:
          logging.info('Bisecting failed to find an Envoy commit that can be trivially integrated.')
          self.best_envoy_commit = None
          self.first_non_trivial_commit = self.envoy_commits_current_to_latest[0]
          self._set_step_not_planned(NighthawkEnvoyUpdateStep.COMMIT_AND_PUSH_UPDATE_BRANCH)
        elif self.best_envoy_commit == self.envoy_commits_current_to_latest[-1]:
          logging.info('The latest Envoy commit can be trivially integrated.')
          self.best_envoy_commit = self.envoy_commits_current_to_latest[-1]
          self.first_non_trivial_commit = None
          self._set_step_not_planned(NighthawkEnvoyUpdateStep.APPLY_PARTIAL_NON_TRIVIAL_MERGE)
        else:
          logging.info(
              'A trivially integrated Envoy commit was found and there are further Envoy commits after it.'
          )
      case NighthawkEnvoyUpdateStep.COMMIT_AND_PUSH_UPDATE_BRANCH:
        if not self.best_envoy_commit:
          raise RuntimeError("Nighthawk repo attempting to commit when no trivial Envoy"
                             " merges were found.")
        _run_command(
            [
                "git",
                "commit",
                "-m",
                self._build_commit_message(self.best_envoy_commit),
            ],
            cwd=self.nighthawk_git_repo_dir,
            interactive=True,
        )
        _run_command(
            [
                "git",
                "push",
                "--force",
                "--set-upstream",
                "origin",
                self.branch_name,
            ],
            cwd=self.nighthawk_git_repo_dir,
            interactive=True,
        )
      case NighthawkEnvoyUpdateStep.APPLY_PARTIAL_INTEGRATION:
        if not self.first_non_trivial_commit:
          raise RuntimeError("Nighthawk repo attempting to apply partial merge when"
                             " first_non_trivial_commit is not set.")
        #EnvoyCommitIntegration(
        #    nighthawk_git_repo_dir=self.nighthawk_git_repo_dir,
        #    envoy_git_repo_dir=self.envoy_git_repo_dir,
        #    current_envoy_commit=self.best_envoy_commit or self.current_envoy_commit,
        #    target_envoy_commit=self.first_non_trivial_commit,
        #    agent_invocation=self.agent_invocation,
        #).run_envoy_commit_integration_steps()
        # DO NOT SUBMIT
        raise RuntimeError(self._non_trivial_merge_instructions())
      case _:
        raise ValueError(f"{step} is not supported.")

  def run_update(self) -> bool:
    """Run all steps for the Nighthawk Envoy update process."""
    results = self._run_steps()
    if not is_clean_results(results):
      raise RuntimeError(f"""Failed to update Nighthawk's Envoy commit to the latest Envoy commit:
        
        {format_step_results(results)}
        """)

  def cleanup(self):
    """Clean up state modifications made by NighthawkEnvoyUpdate."""
    self._envoy_tmp_dir.cleanup()


def main() -> None:
  """Update Envoy version in Nighthawk."""
  logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
  parser = argparse.ArgumentParser(description="Update Envoy version in Nighthawk.")
  parser.add_argument(
      "--nighthawk_dir",
      type=pathlib.Path,
      default="~/github/nighthawk",
      help="The path to the local Nighthawk git repository clone.",
  )
  parser.add_argument(
      "--envoy_clone_depth",
      type=int,
      default=200,
      help="The depth to use when cloning the Envoy repository.",
  )
  parser.add_argument(
      "--branch_name",
      type=str,
      default=f"update-envoy-{datetime.datetime.now().strftime('%Y%m%d')}",
      help=("The name of the branch to create in the Nighthawk repository for the"
            " update."),
  )
  parser.add_argument(
      "--no_sync_nighthawk_repo",
      action="store_false",
      dest="sync_nighthawk_repo",
      help=("If set, the script will not sync the local Nighthawk repository with"
            " the upstream remote before starting the update process."),
  )
  parser.add_argument(
      "--skip_bisection",
      action="store_true",
      dest="skip_bisection",
      help=("If set, the script will only attempt to integrate the latest Envoy commit."),
  )
  parser.add_argument(
      "--agent_invocation",
      type=str,
      default=None,
      help=("Command to invoke an LLM agent, e.g. 'gemini', with prompt by stdin."),
  )

  args = parser.parse_args()

  updater = NighthawkEnvoyUpdate(
      nighthawk_git_repo_dir=args.nighthawk_dir,
      branch_name=args.branch_name,
      envoy_clone_depth=args.envoy_clone_depth,
      sync_nighthawk_repo=args.sync_nighthawk_repo,
      skip_bisection=args.skip_bisection,
      agent_invocation=args.agent_invocation,
  )
  try:
    updater.run_update()
  finally:
    updater.cleanup()


if __name__ == "__main__":
  main()
