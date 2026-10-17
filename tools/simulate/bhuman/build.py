#!/usr/bin/env python3
"""Build the pinned Linux/x86-64 B-Human motion worker outside the HULK checkout."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tarfile
from urllib.request import urlopen

REVISION = "2e3374ba896da8288607f9a1881b876642fe18b8"
FLATBUFFERS_VERSION = "25.2.10"
FLATBUFFERS_SHA256 = "b9c2df49707c57a48fc0923d52b8c73beb72d675f9d44b2211e4569be40a7421"
ORT_VERSION = "1.22.0"
ORT_SHA256 = "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
ADAPTER = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def download(url, destination, expected):
    if not destination.exists():
        partial = destination.with_suffix(".part")
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    with destination.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {destination}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hulk-bhuman" / REVISION
    parser.add_argument("--cache", type=Path, default=default)
    parser.add_argument("--print-cache", action="store_true", help="Print the resolved cache directory without building")
    parser.add_argument("--source", type=Path, help="Reuse an existing pinned release checkout")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel build jobs (default: %(default)s CPU threads)")
    parser.add_argument("--generate-rust", action="store_true", help="Regenerate the checked-in Rust protocol")
    args = parser.parse_args()
    if args.print_cache:
        print(args.cache.resolve())
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("The first worker build supports Linux/x86-64")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve() if args.source else cache / "source"
    if not source.exists():
        run("git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/bhuman/BHumanCodeRelease.git", source)
        run("git", "-C", source, "sparse-checkout", "set",
            "Src", "Make", "Config/Robots", "Config/Locations", "Config/Scenarios", "Config/NeuralNets", "Config/Behavior",
            "Util/CABSL", "Util/CompiledNN2ONNX", "Util/GameController", "Util/fftw-3.3",
            "Util/hdf5", "Util/libjpeg", "Util/onnxruntime", "Util/voronoi")
        run("git", "-C", source, "checkout", "--detach", REVISION)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected B-Human {REVISION}, found {revision}")
    run("git", "-C", source, "diff", "--exit-code", "--quiet", "HEAD")
    if not (source / "Config/Behavior/Strategies").is_dir():
        run("git", "-C", source, "sparse-checkout", "add", "Config/Behavior")
    run("git", "-C", source, "submodule", "update", "--init", "--depth", "1",
        "--filter=blob:none", "Util/CompiledNN", "Util/SimRobot")

    flatbuffers = cache / f"flatbuffers-{FLATBUFFERS_VERSION}"
    archive = cache / "flatbuffers.tar.gz"
    if not flatbuffers.exists():
        download(f"https://codeload.github.com/google/flatbuffers/tar.gz/refs/tags/v{FLATBUFFERS_VERSION}",
                 archive, FLATBUFFERS_SHA256)
        with tarfile.open(archive) as package:
            package.extractall(cache, filter="data")
    flatbuffers_build = cache / "flatbuffers-build"
    run("cmake", "-S", flatbuffers, "-B", flatbuffers_build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DFLATBUFFERS_BUILD_TESTS=OFF",
        "-DFLATBUFFERS_BUILD_FLATLIB=OFF", "-DFLATBUFFERS_BUILD_SHAREDLIB=OFF")
    run("cmake", "--build", flatbuffers_build, "--target", "flatc", "-j", args.jobs)
    flatc = flatbuffers_build / "flatc"

    ort = source / "Util/onnxruntime/lib/Linux"
    version_file = ort / "installed_version"
    if not version_file.exists() or version_file.read_text() != ORT_VERSION:
        archive = cache / "onnxruntime.tar.gz"
        download(f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/"
                 f"onnxruntime-linux-x64-{ORT_VERSION}.tgz", archive, ORT_SHA256)
        ort.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as package:
            for member in package.getmembers():
                if "/lib/" in member.name and (member.isfile() or member.issym()):
                    member.name = member.name.split("/lib/", 1)[1]
                    package.extract(member, ort, filter="data")
        version_file.write_text(ORT_VERSION)
    build = cache / "worker-build"
    run("cmake", "-S", source / "Make/CMake", "-B", build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DMINIMAL_PROJECT=ON",
        "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE",
        f"-DCMAKE_PROJECT_INCLUDE={ADAPTER / 'Inject.cmake'}",
        f"-DFLATBUFFERS_SOURCE={flatbuffers}", f"-DFLATC={flatc}")
    run("cmake", "--build", build, "--target", "bhuman-motion-worker", "-j", args.jobs)
    if args.generate_rust:
        run(flatc, "--rust", "-o", ADAPTER.parent / "src/bhuman", ADAPTER / "protocol.fbs")
        generated = ADAPTER.parent / "src/bhuman/protocol_generated.rs"
        generated.write_text("\n".join(line.rstrip() for line in generated.read_text().splitlines()).rstrip() + "\n")
    print(f"Worker: {build / 'bhuman-motion-worker'}")
    print(f"B-Human source: {source}")


if __name__ == "__main__":
    main()
