# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from posixpath import relpath
from textwrap import dedent

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# https://github.com/google/trimmed_match/blob/master/setup.py
class BazelExtension(Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, bazel_target: str, ext_name: str):
        self._bazel_target = bazel_target
        self._relpath, self._target_name = relpath(bazel_target, "//").split(":")
        super().__init__(ext_name, sources=[])


class BuildBazelExtension(build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)

    def bazel_build(self, ext: BazelExtension):
        # .so file expected on this path
        module_path = self.get_ext_fullpath(ext.name)

        if self.inplace:
            module_relpath = self.get_ext_filename(ext.name)
            if module_path.endswith(module_relpath):
                source_root = module_path[: -len(module_relpath)]
            else:
                source_root = self.build_temp
        else:
            source_root = self.build_temp

        source_root = Path(source_root)
        source_root.mkdir(parents=True, exist_ok=True)

        bazel_prefix = source_root.joinpath("bazel-")

        cmd = [
            "bazel",
            "build",
            ext._bazel_target,
            "--symlink_prefix=" + str(bazel_prefix),
            "--compilation_mode=" + ('dbg' if self.debug else 'opt'),
        ]

        if platform.machine() == "x86_64":
            cmd.extend(["--config=avx"])

        self.spawn(cmd)

        bazel_bin_path = bazel_prefix.with_name('bazel-bin').joinpath(
            ext._relpath,
            ext._target_name,
        )

        Path(module_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bazel_bin_path, module_path)


def version_scheme(version) -> str:
    version.timestamp = datetime.now(timezone.utc).isoformat()
    version.commit = version.node[1:]
    return version.tag.public


def local_scheme(version) -> str:
    if version.distance or version.dirty:
        return f"+{version.node}.dirty"
    elif version.distance:
        return f"+{version.node}"
    else:
        return ""


def version_file_template():
    now = datetime.now(timezone.utc)
    info = {
        "year": now.year,
        "timestamp": now.strftime("%b %d %Y at %X %Z"),
        "docker_version": os.environ.get("SF_BUILD_DOCKER_NAME", "?"),
    }
    content = dedent(
        """
        # Copyright %(year)d Ant Group Co., Ltd.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #   http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.

        # file generated by setuptools_scm
        # DO NOT CHANGE, DO NOT TRACK IN VERSION CONTROL

        __version__ = {version!r}
        __commit_id__ = {scm_version.node!r}[1:]
        __docker_version__ = %(docker_version)r
        __build_time__ = %(timestamp)r


        def build_message():
            msg = [
                f"Secretflow {{__version__}}",
                f"Commit {{__commit_id__}}",
                f"Built on {{__build_time__}}",
            ]
            if __docker_version__ != "?":
                msg.append(f"Docker version {{__docker_version__}}")
            return "\\n".join(msg)
        """
    )
    content = content.strip() + "\n"
    return content % info


if __name__ == "__main__":
    setup(
        packages=find_packages(
            exclude=(
                "examples",
                "examples.*",
                "tests",
                "tests.*",
            )
        ),
        cmdclass={
            "build_ext": BuildBazelExtension,
        },
        ext_modules=[
            BazelExtension(
                "//secretflow_lib/binding:_lib.so",
                "secretflow.security.privacy._lib",
            ),
        ],
        use_scm_version=dict(
            version_scheme=version_scheme,
            local_scheme=local_scheme,
            version_file="secretflow/version.py",
            version_file_template=version_file_template(),
            fallback_version="0.0.0",
        ),
    )
