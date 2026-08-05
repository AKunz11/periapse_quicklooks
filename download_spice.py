"""
Download the MAVEN/IUVS SPICE kernels needed by the LASP quicklook code.

Example
-------
python download_maven_spice.py --orbit-start 8000 --orbit-end 9000

This creates:

spice/
├── generic_kernels/
│   ├── generic.tm
│   ├── lsk/
│   ├── pck/
│   └── spk/
└── mvn/
    ├── mvn.tm
    ├── ck/
    ├── fk/
    ├── ik/
    ├── sclk/
    └── spk/

The layout is compatible with:

    load_iuvs_spice(spice_directory=".../spice")

Notes
-----
* --orbit-end is exclusive.
* CK selection is intentionally conservative: it may download some extra
  attitude kernels surrounding the requested orbit interval.
* Run the script again later to update missing or newer files.

Run in terminal something like:
python download_maven_spice.py --orbit-start 8000 --orbit-end 9000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional, Sequence


NAIF_MAVEN = "https://naif.jpl.nasa.gov/pub/naif/MAVEN/kernels"
NAIF_GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

USER_AGENT = "MAVEN-IUVS-SPICE-Installer/1.0"
STATIC_MAVEN_DIRS = ("fk", "ik", "lsk", "pck", "sclk")
KERNEL_EXTENSIONS = {
    ".bc", ".bsp", ".tf", ".ti", ".tls", ".tpc", ".tsc", ".tm"
}


class DirectoryParser(HTMLParser):
    """Extract href values from an Apache-style directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str


def request_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_remote_files(url: str) -> list[RemoteFile]:
    """Return files from a NAIF Apache directory listing."""
    html = request_bytes(url.rstrip("/") + "/").decode("utf-8", errors="replace")
    parser = DirectoryParser()
    parser.feed(html)

    files: list[RemoteFile] = []
    for href in parser.hrefs:
        href = urllib.parse.unquote(href)
        if href.startswith("?") or href.startswith("/") or href in ("../", "./"):
            continue
        if href.endswith("/"):
            continue

        name = os.path.basename(href)
        if not name or name.endswith(".xml"):
            continue

        files.append(
            RemoteFile(
                url=urllib.parse.urljoin(url.rstrip("/") + "/", href),
                name=name,
            )
        )
    return files


def download_file(remote: RemoteFile, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {remote.url}")
    try:
        request = urllib.request.Request(
            remote.url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    percent = 100.0 * downloaded / total_bytes
                    print(
                        f"\r           {destination.name}: "
                        f"{downloaded / 1024**2:.1f}/{total_bytes / 1024**2:.1f} MiB "
                        f"({percent:.1f}%)",
                        end="",
                        flush=True,
                    )

            if total_bytes:
                print()

        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def latest_by_version(files: Sequence[RemoteFile]) -> list[RemoteFile]:
    """
    Keep the newest version of files that differ only by a version suffix.

    This is mainly useful for static text kernels. Time-segmented CK/SPK files
    are handled separately and are not passed through this function.
    """
    groups: dict[str, RemoteFile] = {}

    for remote in files:
        stem = remote.name
        key = re.sub(r"(?i)(?:_v|\.)(\d+)(?=\.[^.]+$)", "", stem)

        current = groups.get(key)
        if current is None or remote.name > current.name:
            groups[key] = remote

    return sorted(groups.values(), key=lambda item: item.name.lower())


def download_static_maven_kernels(root: Path, force: bool) -> None:
    """
    Download MAVEN FK, IK, LSK, PCK and SCLK kernels.

    We retain all listed text kernels because the existing PyUVS loader may
    explicitly glob legacy names such as MVN_SCLKSCET.#####.tsc.
    """
    for kernel_type in STATIC_MAVEN_DIRS:
        remote_dir = f"{NAIF_MAVEN}/{kernel_type}"
        local_dir = root / "mvn" / kernel_type

        files = [
            item for item in list_remote_files(remote_dir)
            if Path(item.name).suffix.lower() in KERNEL_EXTENSIONS
        ]

        if not files:
            raise RuntimeError(f"No kernels found at {remote_dir}")

        for remote in files:
            downloaded = download_file(
                remote,
                local_dir / remote.name,
                force=force,
            )

            # The supplied PyUVS loader searches for the historical uppercase
            # naming convention: MVN_SCLKSCET.#####.tsc. Current NAIF archive
            # files may use mvn_sclkscet_#####.tsc instead, so create a local
            # compatibility copy without changing the downloaded original.
            if kernel_type == "sclk":
                match = re.fullmatch(
                    r"(?i)mvn_sclkscet[._](\d{5})\.tsc",
                    remote.name,
                )
                if match:
                    compatibility = local_dir / f"MVN_SCLKSCET.{match.group(1)}.tsc"
                    if force or not compatibility.exists():
                        compatibility.write_bytes(downloaded.read_bytes())
                        print(f"[compat] {compatibility}")


def parse_orbit_range(filename: str) -> Optional[tuple[int, int]]:
    """
    Extract an orbit interval such as 08000-08099 from a trajectory filename.
    """
    match = re.search(r"(?<!\d)(\d{5})-(\d{5})(?!\d)", filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= b1 and b0 <= a1


def select_spk_files(
    files: Sequence[RemoteFile],
    orbit_start: int,
    orbit_end_exclusive: int,
) -> list[RemoteFile]:
    """
    Select reconstructed trajectory SPKs overlapping the requested orbit range.

    If multiple versions exist for one orbit interval, retain the lexically
    newest filename, matching the convention used by the existing loader.
    """
    requested_end = orbit_end_exclusive - 1
    grouped: dict[tuple[int, int], RemoteFile] = {}

    for remote in files:
        name_lower = remote.name.lower()
        if not name_lower.endswith(".bsp"):
            continue
        if "trj_orb_" not in name_lower:
            continue

        interval = parse_orbit_range(remote.name)
        if interval is None:
            continue
        if not intervals_overlap(interval[0], interval[1], orbit_start, requested_end):
            continue

        current = grouped.get(interval)
        if current is None:
            grouped[interval] = remote
            continue

        # Prefer reconstructed products, then the newest filename/version.
        current_rec = "_rec" in current.name.lower()
        candidate_rec = "_rec" in remote.name.lower()

        if candidate_rec and not current_rec:
            grouped[interval] = remote
        elif candidate_rec == current_rec and remote.name > current.name:
            grouped[interval] = remote

    return [grouped[key] for key in sorted(grouped)]


def download_maven_spk(
    root: Path,
    orbit_start: int,
    orbit_end: int,
    force: bool,
) -> list[Path]:
    remote_dir = f"{NAIF_MAVEN}/spk"
    files = list_remote_files(remote_dir)
    selected = select_spk_files(files, orbit_start, orbit_end)

    if not selected:
        raise RuntimeError(
            f"No MAVEN trajectory SPKs found for orbits "
            f"{orbit_start}-{orbit_end - 1} at {remote_dir}"
        )

    paths = []
    for remote in selected:
        paths.append(
            download_file(remote, root / "mvn" / "spk" / remote.name, force=force)
        )

    return paths


def download_generic_support(root: Path, force: bool) -> None:
    """
    Download generic kernels directly used by the supplied PyUVS loader.

    mar097.bsp is explicitly furnished by spice.py. We also obtain a current
    generic LSK, text PCK, and compact planetary ephemeris.
    """
    destinations = {
        f"{NAIF_GENERIC}/spk/satellites/mar097.bsp":
            root / "generic_kernels" / "spk" / "mar097.bsp",
        f"{NAIF_GENERIC}/lsk/naif0012.tls":
            root / "generic_kernels" / "lsk" / "naif0012.tls",
        f"{NAIF_GENERIC}/pck/pck00011.tpc":
            root / "generic_kernels" / "pck" / "pck00011.tpc",
        f"{NAIF_GENERIC}/spk/planets/de440s.bsp":
            root / "generic_kernels" / "spk" / "de440s.bsp",
    }

    for url, destination in destinations.items():
        download_file(
            RemoteFile(url=url, name=destination.name),
            destination,
            force=force,
        )


def spk_utc_coverage(spk_paths: Sequence[Path], root: Path) -> Optional[tuple[datetime, datetime]]:
    """
    Determine approximate UTC coverage from downloaded MAVEN SPKs.

    This requires spiceypy. If unavailable, CK selection falls back to
    downloading all MAVEN CKs.
    """
    try:
        import spiceypy as spice
    except ImportError:
        print(
            "[warning] spiceypy is not installed; cannot derive CK date coverage."
        )
        return None

    lsk = root / "generic_kernels" / "lsk" / "naif0012.tls"
    if not lsk.exists():
        return None

    spice.kclear()
    spice.furnsh(str(lsk))

    starts: list[float] = []
    stops: list[float] = []

    try:
        for path in spk_paths:
            ids = spice.spkobj(str(path))
            for object_id in ids:
                coverage = spice.spkcov(str(path), int(object_id))
                count = spice.wncard(coverage)
                for index in range(count):
                    start_et, stop_et = spice.wnfetd(coverage, index)
                    starts.append(float(start_et))
                    stops.append(float(stop_et))

        if not starts:
            return None

        start_text = spice.et2utc(min(starts), "ISOC", 3)
        stop_text = spice.et2utc(max(stops), "ISOC", 3)
        start = datetime.fromisoformat(start_text).replace(tzinfo=timezone.utc)
        stop = datetime.fromisoformat(stop_text).replace(tzinfo=timezone.utc)
        return start, stop
    finally:
        spice.kclear()


def date_tokens(filename: str) -> list[datetime]:
    """
    Extract plausible YYYYMMDD date tokens from a kernel filename.
    """
    dates: list[datetime] = []
    for token in re.findall(r"(?<!\d)(20\d{6})(?!\d)", filename):
        try:
            dates.append(
                datetime.strptime(token, "%Y%m%d").replace(tzinfo=timezone.utc)
            )
        except ValueError:
            continue
    return dates


def ck_overlaps_dates(
    filename: str,
    start: datetime,
    stop: datetime,
    padding_days: int = 14,
) -> bool:
    """
    Conservatively decide whether a CK filename may overlap the requested dates.

    Kernels with no parseable dates are retained because they may be long-term,
    fixed, or mission-wide orientation products.
    """
    dates = date_tokens(filename)
    if not dates:
        return True

    padded_start = start - timedelta(days=padding_days)
    padded_stop = stop + timedelta(days=padding_days)

    if len(dates) == 1:
        return padded_start.date() <= dates[0].date() <= padded_stop.date()

    kernel_start = min(dates)
    kernel_stop = max(dates)
    return kernel_start <= padded_stop and padded_start <= kernel_stop


def download_maven_ck(
    root: Path,
    coverage: Optional[tuple[datetime, datetime]],
    force: bool,
    all_ck: bool,
) -> list[Path]:
    remote_dir = f"{NAIF_MAVEN}/ck"
    files = [
        item for item in list_remote_files(remote_dir)
        if item.name.lower().endswith(".bc")
    ]

    if not files:
        raise RuntimeError(f"No MAVEN CK kernels found at {remote_dir}")

    if all_ck or coverage is None:
        selected = files
        print(
            "[warning] Downloading all MAVEN CK files. This can require "
            "substantial disk space and time."
        )
    else:
        start, stop = coverage
        print(
            f"[info] Selecting CK files around "
            f"{start.date().isoformat()} through {stop.date().isoformat()}"
        )
        selected = [
            item for item in files
            if ck_overlaps_dates(item.name, start, stop)
        ]

    paths = []
    for remote in selected:
        paths.append(
            download_file(remote, root / "mvn" / "ck" / remote.name, force=force)
        )

    return paths


def write_generic_metakernel(root: Path) -> Path:
    path = root / "generic_kernels" / "generic.tm"
    content = r"""KPL/MK

\begindata

PATH_SYMBOLS = ( 'GENERIC' )

KERNELS_TO_LOAD = (
    '$GENERIC/lsk/naif0012.tls'
    '$GENERIC/pck/pck00011.tpc'
    '$GENERIC/spk/de440s.bsp'
)

\begintext

Generic support kernels for the MAVEN/IUVS quicklook workflow.
mar097.bsp is loaded separately by PyUVS.spice.load_iuvs_spice().
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def newest_matching(directory: Path, patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
    return sorted(set(matches))


def write_maven_metakernel(root: Path) -> Path:
    """
    Write the static MAVEN meta-kernel.

    Dynamic CK, SPK, and SCLK files are loaded separately by the supplied
    spice.py functions, so they are deliberately not duplicated here.
    """
    mvn = root / "mvn"
    static_files: list[Path] = []

    static_files.extend(newest_matching(mvn / "fk", ("*.tf",)))
    static_files.extend(newest_matching(mvn / "ik", ("*.ti",)))
    static_files.extend(newest_matching(mvn / "lsk", ("*.tls",)))
    static_files.extend(newest_matching(mvn / "pck", ("*.tpc",)))

    # Keep only the lexically newest version for obvious versioned static files.
    chosen: dict[str, Path] = {}
    for path in static_files:
        key = re.sub(r"(?i)(?:_v|\.)(\d+)(?=\.[^.]+$)", "", path.name)
        current = chosen.get(key)
        if current is None or path.name > current.name:
            chosen[key] = path

    kernel_lines = []
    for path in sorted(chosen.values(), key=lambda p: (p.parent.name, p.name)):
        relative = path.relative_to(mvn).as_posix()
        kernel_lines.append(f"    '$MVN/{relative}'")

    content = """KPL/MK

\\begindata

PATH_SYMBOLS = ( 'MVN' )

KERNELS_TO_LOAD = (
%s
)

\\begintext

Static MAVEN kernels for the IUVS quicklook workflow.
CK, SPK, and SCLK kernels are loaded separately by PyUVS.spice.
""" % "\n".join(kernel_lines)

    path = mvn / "mvn.tm"
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def verify_installation(root: Path, orbit_start: int, orbit_end: int) -> bool:
    """Perform structural checks and, when possible, a SpiceyPy load test."""
    required = [
        root / "generic_kernels" / "generic.tm",
        root / "generic_kernels" / "spk" / "mar097.bsp",
        root / "mvn" / "mvn.tm",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        print("[error] Missing required files:")
        for path in missing:
            print(f"        {path}")
        return False

    spks = list((root / "mvn" / "spk").glob("*.bsp"))
    cks = list((root / "mvn" / "ck").glob("*.bc"))
    sclks = list((root / "mvn" / "sclk").glob("*.tsc"))

    print()
    print("Installation summary")
    print("--------------------")
    print(f"Root:           {root.resolve()}")
    print(f"Orbit request:  {orbit_start}-{orbit_end - 1}")
    print(f"MAVEN SPKs:     {len(spks)}")
    print(f"MAVEN CKs:      {len(cks)}")
    print(f"MAVEN SCLKs:    {len(sclks)}")

    try:
        import spiceypy as spice
    except ImportError:
        print("[warning] spiceypy unavailable; structural verification only.")
        return True

    try:
        spice.kclear()
        spice.furnsh(str(root / "generic_kernels" / "generic.tm"))
        spice.furnsh(str(root / "mvn" / "mvn.tm"))
        spice.furnsh(str(root / "generic_kernels" / "spk" / "mar097.bsp"))

        radii = spice.bodvrd("MARS", "RADII", 3)[1]
        print(f"Mars radii test: {radii}")
        print("SPICE load test: passed")
        return True
    except Exception as exc:
        print(f"[error] SPICE load test failed: {exc}")
        return False
    finally:
        spice.kclear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MAVEN/IUVS SPICE kernels for a requested orbit range."
    )
    parser.add_argument(
        "--orbit-start",
        type=int,
        required=True,
        help="First MAVEN orbit to support, inclusive.",
    )
    parser.add_argument(
        "--orbit-end",
        type=int,
        required=True,
        help="Final orbit boundary, exclusive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spice"),
        help="Destination SPICE directory (default: ./spice).",
    )
    parser.add_argument(
        "--all-ck",
        action="store_true",
        help="Download every MAVEN CK instead of selecting by SPK date coverage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.orbit_start < 0:
        print("[error] --orbit-start must be non-negative.")
        return 2
    if args.orbit_end <= args.orbit_start:
        print("[error] --orbit-end must be greater than --orbit-start.")
        return 2

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Installing MAVEN/IUVS SPICE kernels into: {root}")
    print()

    try:
        download_generic_support(root, force=args.force)
        download_static_maven_kernels(root, force=args.force)

        spk_paths = download_maven_spk(
            root,
            orbit_start=args.orbit_start,
            orbit_end=args.orbit_end,
            force=args.force,
        )

        coverage = spk_utc_coverage(spk_paths, root)

        download_maven_ck(
            root,
            coverage=coverage,
            force=args.force,
            all_ck=args.all_ck,
        )

        write_generic_metakernel(root)
        write_maven_metakernel(root)

        success = verify_installation(
            root,
            orbit_start=args.orbit_start,
            orbit_end=args.orbit_end,
        )
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\n[error] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())