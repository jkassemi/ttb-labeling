#!/usr/bin/env python3
"""
Polite sampler for TTB Public COLA Registry (ttbonline.gov / colasonline).

Workflow (offline dataset prep):
  1) GET search page (establish session cookies)
  2) POST search (dateCompletedFrom/To, etc.)
  3) Export current results to CSV (server-side session) via POST
  4) Sample TTB IDs from CSV
  5) For each TTB ID: open details page, follow "Printable Version" link,
     download images (+ save HTML)

Key properties:
  - Single-threaded + jittered sleeps
  - Retries/backoff on transient errors
  - Persistent state (resume-friendly)
  - Optional --test: perform ONE search window + download ONE COLA then exit

Notes:
  - If your environment can't validate TLS (common in minimal containers), use
    --insecure to disable cert verification *for dataset prep only*. The client
    will also auto-disable verification after an SSL failure to avoid repeated
    errors.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import random
import re
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

TTBID_RE = re.compile(r"\b(\d{14})\b")


@dataclass(frozen=True)
class Config:
    base: str
    outdir: Path
    samples_dir: Path
    min_sleep: float
    max_sleep: float
    timeout_s: int
    user_agent: str
    insecure: bool
    verbose: bool


def log(cfg: Config, msg: str) -> None:
    if cfg.verbose:
        print(msg, file=sys.stderr)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def sleepy(cfg: Config) -> None:
    time.sleep(random.uniform(cfg.min_sleep, cfg.max_sleep))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mmddyyyy(d: dt.date) -> str:
    return d.strftime("%m/%d/%Y")


def get_session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    s.verify = not cfg.insecure
    if cfg.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return s


def _request_with_retries(
    sess: requests.Session,
    cfg: Config,
    method: str,
    url: str,
    *,
    data: dict[str, str] | None = None,
    stream: bool = False,
) -> requests.Response:
    sleepy(cfg)
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            r = sess.request(
                method,
                url,
                data=data,
                timeout=cfg.timeout_s,
                allow_redirects=True,
                stream=stream,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                backoff = min(60, 2**attempt)
                log(cfg, f"[retry] {r.status_code} {method} {url} (sleep {backoff}s)")
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.SSLError as e:
            last_exc = e
            if sess.verify is not False:
                log(
                    cfg,
                    "[warn] SSL verification failed; retrying with verification "
                    "disabled",
                )
                sess.verify = False
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            backoff = min(60, 2**attempt)
            time.sleep(backoff)
        except Exception as e:
            last_exc = e
            backoff = min(60, 2**attempt)
            log(
                cfg,
                f"[retry] exception {type(e).__name__} on {method} {url} "
                f"(sleep {backoff}s)",
            )
            time.sleep(backoff)
    raise RuntimeError(f"Failed after retries: {method} {url}") from last_exc


def fetch_html(
    sess: requests.Session,
    cfg: Config,
    url: str,
    *,
    method: str = "GET",
    data: dict[str, str] | None = None,
) -> str:
    r = _request_with_retries(sess, cfg, method, url, data=data)
    # requests will guess encoding; keep as-is
    return r.text


def infer_ext(content_type: str, blob: bytes) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if ct == "image/png":
        return ".png"
    if ct in ("image/tiff", "image/tif"):
        return ".tif"
    if ct == "application/pdf":
        return ".pdf"

    # Sniff common magic numbers when content-type is missing/incorrect
    if blob.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"%PDF"):
        return ".pdf"
    if blob.startswith(b"II*\x00") or blob.startswith(b"MM\x00*"):
        return ".tif"
    return ""


def download_bytes(sess: requests.Session, cfg: Config, url: str) -> tuple[bytes, str]:
    r = _request_with_retries(sess, cfg, "GET", url)
    return r.content, (r.headers.get("content-type") or "")


def find_link(soup: BeautifulSoup, text_pat: str) -> str | None:
    a = soup.find("a", string=re.compile(text_pat, re.I))
    if not a:
        return None
    href = a.get("href") or ""
    if href.startswith("javascript:") or href == "javascript:void(0)":
        onclick = a.get("onclick") or ""
        extracted = extract_url_from_onclick(onclick)
        return extracted or None
    return href or None


def extract_url_from_onclick(onclick: str) -> str | None:
    if not onclick:
        return None
    for candidate in re.findall(r"['\"]([^'\"]+)['\"]", onclick):
        if candidate.startswith(("http://", "https://", "/")) or ".do" in candidate:
            return candidate
    return None


class PrintableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_label = False
        self._in_data = False
        self._label_buf: list[str] = []
        self._data_buf: list[str] = []
        self._pending_label: str | None = None
        self.fields_raw: dict[str, str] = {}
        self.checkbox_raw: dict[str, bool] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        if tag.lower() == "div":
            cls = attrs_dict.get("class", "").lower()
            if cls in {"label", "boldlabel"}:
                self._in_label = True
                self._label_buf = []
            elif cls == "data":
                self._in_data = True
                self._data_buf = []
        elif tag.lower() == "input":
            if attrs_dict.get("type", "").lower() == "checkbox":
                alt = attrs_dict.get("alt", "").strip()
                if alt:
                    checked = "checked" in attrs_dict
                    self.checkbox_raw[alt] = checked

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "div":
            return
        if self._in_label:
            label = _clean_text("".join(self._label_buf))
            if label:
                self._pending_label = label
            self._in_label = False
        elif self._in_data:
            data = _clean_text("".join(self._data_buf))
            if self._pending_label is not None:
                self.fields_raw[self._pending_label] = data
                self._pending_label = None
            self._in_data = False

    def handle_data(self, data: str) -> None:
        if self._in_label:
            self._label_buf.append(data)
        elif self._in_data:
            self._data_buf.append(data)


class AttachmentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        src = attrs_dict.get("src", "")
        if "publicViewAttachment.do" in src:
            self.urls.append(src)


_DETAIL_STRONG_RE = re.compile(
    r"<strong>([^<]+)</strong>(.*?)</td>", re.IGNORECASE | re.DOTALL
)


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\xa0", " ")
    value = " ".join(value.split())
    return value.strip()


def parse_printable(html_text: str) -> tuple[dict[str, str], dict[str, bool]]:
    parser = PrintableParser()
    parser.feed(html_text)
    return parser.fields_raw, parser.checkbox_raw


def parse_detail(html_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for label, value_html in _DETAIL_STRONG_RE.findall(html_text):
        label_clean = _clean_text(label)
        if not label_clean:
            continue
        value_clean = _clean_text(value_html)
        fields[label_clean] = value_clean
    return fields


def normalize_label(label: str) -> str:
    label = label.strip().strip(":")
    label = re.sub(r"^\d+\.?\s*", "", label)
    label = re.sub(r"\s+\(.*?\)$", "", label)
    label = label.lower()
    label = label.replace("brewer's", "brewers")
    label = label.replace("brewer’s", "brewers")
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = re.sub(r"\s+", " ", label)
    return label.strip()


_LABEL_TO_KEY = {
    "ttb id": "ttb_id",
    "serial #": "serial_number",
    "serial number": "serial_number",
    "brand name": "brand_name",
    "fanciful name": "fanciful_name",
    "class/type code": "class_type_code",
    "class/type description": "class_type_description",
    "origin code": "origin_code",
    "type of application": "type_of_application",
    "type of product": "type_of_product",
    "source of product": "source_of_product",
    "for sale in": "for_sale_in",
    "total bottle capacity": "total_bottle_capacity",
    "net contents": "net_contents",
    "alcohol content": "alcohol_content",
    "wine appellation": "wine_appellation",
    "wine appellation if on label": "wine_appellation",
    "wine vintage": "wine_vintage",
    "wine vintage date if on label": "wine_vintage",
    "grape varietal(s)": "grape_varietals",
    "formula": "formula",
    "formula/sop no": "formula",
    "lab no/lab date": "lab_no_date",
    "lab no & date / preimport no & date": "lab_no_date",
    "approval date": "approval_date",
    "status": "status",
    "qualifications": "qualifications",
    "vendor code": "vendor_code",
    "ct": "class_type_code_short",
    "or": "origin_code_short",
    "expiration date": "expiration_date",
    "date of application": "date_of_application",
    "date issued": "date_issued",
    (
        "name and address of applicant as shown on plant registry basic permit "
        "or brewers notice include approved dba or tradename if used on label"
    ): "applicant_name_address",
    "plant registry/basic permit/brewers no": "plant_registry_number",
    (
        "plant registry/basic permit/brewers no principal place of business"
    ): "plant_registry_principal",
    "plant registry/basic permit/brewers no other": "plant_registry_other",
    "contact information": "contact_information",
    "phone number": "phone_number",
    "fax number": "fax_number",
    "email address": "email_address",
}

_DETAIL_PRIORITY_KEYS = {"status", "type_of_application"}


def map_fields(
    fields_printable: dict[str, str], fields_detail: dict[str, str]
) -> dict[str, str]:
    mapped: dict[str, str] = {}

    for label, value in fields_printable.items():
        norm = normalize_label(label)
        key = _LABEL_TO_KEY.get(norm)
        if key:
            mapped[key] = value

    for label, value in fields_detail.items():
        norm = normalize_label(label)
        key = _LABEL_TO_KEY.get(norm)
        if key:
            if key in _DETAIL_PRIORITY_KEYS or not mapped.get(key):
                mapped[key] = value

    return mapped


def derive_checkbox_fields(checkbox_raw: dict[str, bool]) -> dict[str, list[str]]:
    type_of_product: list[str] = []
    if checkbox_raw.get("Type of Product: Wine"):
        type_of_product.append("wine")
    if checkbox_raw.get("Type of Product: Distilled Spirits"):
        type_of_product.append("distilled_spirits")
    if checkbox_raw.get("Type of Product: Malt Beverage"):
        type_of_product.append("malt_beverage")

    source_of_product: list[str] = []
    if checkbox_raw.get("Source of Product: Domestic"):
        source_of_product.append("domestic")
    if checkbox_raw.get("Source of Product: Imported"):
        source_of_product.append("imported")

    type_of_application: list[str] = []
    if checkbox_raw.get("Type of Application"):
        type_of_application.append("certificate_of_label_approval")
    if checkbox_raw.get("Certificate of label Approval"):
        type_of_application.append("certificate_of_exemption_from_label_approval")
    if checkbox_raw.get("Distinctive Liquor Bottle Approval"):
        type_of_application.append("distinctive_liquor_bottle_approval")
    if checkbox_raw.get("Previous TTB Id"):
        type_of_application.append("resubmission_after_rejection")

    derived: dict[str, list[str]] = {}
    if type_of_product:
        derived["type_of_product"] = type_of_product
    if source_of_product:
        derived["source_of_product"] = source_of_product
    if type_of_application:
        derived["type_of_application"] = type_of_application
    return derived


def extract_ttbids_from_csv_bytes(b: bytes) -> list[str]:
    text = b.decode("utf-8", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    ttbids: list[str] = []
    for row in rows:
        for cell in row:
            m = TTBID_RE.search(cell.replace("'", ""))
            if m:
                ttbids.append(m.group(1))
                break
    return sorted(set(ttbids))


def extract_ttbids_from_results_html(html: str) -> list[str]:
    """
    Fallback extractor if CSV export fails.
    Tries links first, then brute regex.
    """
    soup = BeautifulSoup(html, "html.parser")
    ids: set[str] = set()

    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]ttbid=(\d{14})\b", a["href"])
        if m:
            ids.add(m.group(1))

    if not ids:
        ids.update(TTBID_RE.findall(html))

    return sorted(ids)


def export_search_results_csv(sess: requests.Session, cfg: Config) -> bytes | None:
    """
    Trigger 'Save Search Results To File' via POST.
    Must be called AFTER a successful search POST.
    Returns bytes that should be CSV; may return HTML if export is unavailable
    for some reason.
    """
    export_url = urljoin(cfg.base, "publicSaveSearchResults.do?action=save")
    r = _request_with_retries(sess, cfg, "POST", export_url)
    content_type = (r.headers.get("content-type") or "").lower()

    # Some failures return HTML (e.g., session not ready). Detect and return
    # None to fall back.
    if "text/html" in content_type:
        log(
            cfg,
            f"[warn] export returned HTML content-type={content_type}; "
            "falling back to HTML parsing",
        )
        return None

    return r.content


def download_images_from_printable(
    sess: requests.Session,
    cfg: Config,
    printable_url: str,
    item_dir: Path,
) -> tuple[list[dict[str, str]], str]:
    html = fetch_html(sess, cfg, printable_url)
    parser = AttachmentParser()
    parser.feed(html)
    candidates = [urljoin(cfg.base, u) for u in parser.urls]

    ensure_dir(item_dir / "images")
    images_info: list[dict[str, str]] = []
    used_names: set[str] = {
        p.name for p in (item_dir / "images").iterdir() if p.is_file()
    }

    for u in candidates:
        parsed = urlparse(u)
        qs = parse_qs(parsed.query)
        source_name = (
            qs.get("filename", [""])[0] or Path(parsed.path).name or "attachment"
        ).strip()
        safe_name = re.sub(r"[\s]+", "_", source_name)
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        base = Path(safe_name).stem or "attachment"
        ext = Path(safe_name).suffix

        out = item_dir / "images" / f"{base}{ext}"
        suffix = 1
        while out.name in used_names:
            out = item_dir / "images" / f"{base}_{suffix}{ext}"
            suffix += 1
        used_names.add(out.name)

        if not out.exists():
            blob, content_type = download_bytes(sess, cfg, u)
            inferred_ext = infer_ext(content_type, blob)
            if inferred_ext and not out.name.lower().endswith(inferred_ext):
                out = out.with_suffix(inferred_ext)
                if out.exists():
                    images_info.append(
                        {"file": out.name, "source_url": u, "source_name": source_name}
                    )
                    continue
            out.write_bytes(blob)
        images_info.append(
            {"file": out.name, "source_url": u, "source_name": source_name}
        )

    (item_dir / "printable.html").write_text(html, encoding="utf-8", errors="ignore")
    return images_info, html


def fetch_one_cola(
    sess: requests.Session, cfg: Config, ttbid: str
) -> dict[str, object]:
    item_dir = cfg.samples_dir / ttbid
    ensure_dir(item_dir)

    detail_urls = [
        urljoin(
            cfg.base,
            f"viewColaDetails.do?action=publicDisplaySearchBasic&ttbid={ttbid}",
        ),
        urljoin(cfg.base, f"viewColaDetails.do?action=publicFormDisplay&ttbid={ttbid}"),
    ]

    detail_html = None
    detail_url_used = None
    for u in detail_urls:
        try:
            detail_html = fetch_html(sess, cfg, u)
            detail_url_used = u
            break
        except Exception as e:
            log(cfg, f"[warn] detail fetch failed {u}: {type(e).__name__}")
            continue

    if detail_html is None:
        return {"ttbid": ttbid, "status": "error", "error": "could_not_fetch_detail"}

    (item_dir / "detail.html").write_text(
        detail_html, encoding="utf-8", errors="ignore"
    )

    soup = BeautifulSoup(detail_html, "html.parser")
    printable_href = find_link(soup, r"Printable\s+Version")

    if not printable_href:
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if "print" in href and "cola" in href:
                printable_href = a["href"]
                break
            if href.startswith("javascript:") or href == "javascript:void(0)":
                extracted = extract_url_from_onclick(a.get("onclick") or "")
                if (
                    extracted
                    and "print" in extracted.lower()
                    and "cola" in extracted.lower()
                ):
                    printable_href = extracted
                    break

    images_info: list[dict[str, str]] = []
    printable_html = ""
    printable_url: str | None = None
    if printable_href and not printable_href.startswith("javascript:"):
        printable_url = urljoin(cfg.base, printable_href)
        images_info, printable_html = download_images_from_printable(
            sess, cfg, printable_url, item_dir
        )
    else:
        log(cfg, f"[warn] no printable link found for {ttbid}")

    if printable_html:
        fields_printable, checkbox_raw = parse_printable(printable_html)
    else:
        fields_printable, checkbox_raw = {}, {}
    fields_detail = parse_detail(detail_html)
    mapped = map_fields(fields_printable, fields_detail)
    derived = derive_checkbox_fields(checkbox_raw)
    for key, value in derived.items():
        mapped[key] = value

    fixture = {
        "id": ttbid,
        "source": "ttb_real",
        "synthetic": False,
        "fields": mapped,
        "fields_raw_printable": fields_printable,
        "fields_raw_detail": fields_detail,
        "checkboxes": checkbox_raw,
        "images": [x["file"] for x in images_info],
        "images_detail": images_info,
    }
    (item_dir / "data.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    meta = {
        "ttbid": ttbid,
        "status": "ok",
        "detail_url": detail_url_used,
        "printable_url": printable_url,
        "images_downloaded": len(images_info),
        "downloaded_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def random_date_windows(
    start: dt.date,
    end: dt.date,
    windows: int,
    min_days: int,
    max_days: int,
) -> Iterable[tuple[dt.date, dt.date]]:
    span = (end - start).days
    for _ in range(windows):
        length = random.randint(min_days, max_days)
        if length >= span:
            yield start, end
            continue
        offset = random.randint(0, span - length)
        a = start + dt.timedelta(days=offset)
        b = a + dt.timedelta(days=length)
        yield a, b


def load_state(path: Path) -> dict[str, list[str]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"seen_ttbids": [], "downloaded_ttbids": []}
    return {"seen_ttbids": [], "downloaded_ttbids": []}


def save_state(path: Path, seen: set[str], downloaded: set[str]) -> None:
    state = {"seen_ttbids": sorted(seen), "downloaded_ttbids": sorted(downloaded)}
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="https://ttbonline.gov/colasonline/",
        help="Base URL (end with /colasonline/)",
    )
    ap.add_argument(
        "--outdir",
        default="tests/fixtures",
        help="Output directory (will write into <outdir>/samples)",
    )
    ap.add_argument("--target", type=int, default=20, help="How many COLAs to download")
    ap.add_argument(
        "--windows", type=int, default=12, help="How many random date windows to try"
    )
    ap.add_argument(
        "--per-window",
        type=int,
        default=8,
        help="Max IDs to sample per window (before downloading)",
    )
    ap.add_argument(
        "--from-date", default="01/01/2025", help="Earliest completed date (MM/DD/YYYY)"
    )
    ap.add_argument(
        "--to-date", default="01/25/2026", help="Latest completed date (MM/DD/YYYY)"
    )
    ap.add_argument("--min-window-days", type=int, default=3)
    ap.add_argument("--max-window-days", type=int, default=21)
    ap.add_argument(
        "--min-sleep", type=float, default=1.2, help="Min seconds between requests"
    )
    ap.add_argument(
        "--max-sleep", type=float, default=2.8, help="Max seconds between requests"
    )
    ap.add_argument("--timeout", type=int, default=40)

    ap.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (dataset prep only)",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging to stderr")

    ap.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run ONE window search + download ONE COLA, then exit (reduces "
            "traffic while developing)"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling",
    )

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    outdir = Path(args.outdir)
    samples_dir = outdir / "samples"
    cfg = Config(
        base=args.base if args.base.endswith("/") else (args.base + "/"),
        outdir=outdir,
        samples_dir=samples_dir,
        min_sleep=args.min_sleep,
        max_sleep=args.max_sleep,
        timeout_s=args.timeout,
        user_agent="cola-sampler/0.2 (polite; non-parallel; research prototype)",
        insecure=bool(args.insecure),
        verbose=bool(args.verbose),
    )
    ensure_dir(cfg.outdir)
    ensure_dir(cfg.samples_dir)

    state_path = cfg.outdir / "state.json"
    state = load_state(state_path)
    seen: set[str] = set(state.get("seen_ttbids", []))
    downloaded: set[str] = set(state.get("downloaded_ttbids", []))

    sess = get_session(cfg)

    search_page = urljoin(cfg.base, "publicSearchColasBasic.do")
    search_process = urljoin(cfg.base, "publicSearchColasBasicProcess.do?action=search")

    # establish session
    log(cfg, f"[info] GET {search_page}")
    _ = fetch_html(sess, cfg, search_page)

    from_d = dt.datetime.strptime(args.from_date, "%m/%d/%Y").date()
    to_d = dt.datetime.strptime(args.to_date, "%m/%d/%Y").date()

    # In --test mode: only one window + one download
    windows = 1 if args.test else args.windows
    target = 1 if args.test else args.target
    per_window = 1 if args.test else args.per_window

    pool: list[str] = []
    window_i = 0
    for a, b in random_date_windows(
        from_d, to_d, windows, args.min_window_days, args.max_window_days
    ):
        window_i += 1
        data = {
            "searchCriteria.dateCompletedFrom": mmddyyyy(a),
            "searchCriteria.dateCompletedTo": mmddyyyy(b),
            "searchCriteria.productOrFancifulName": "",
            "searchCriteria.productNameSearchType": "E",
            "searchCriteria.classTypeFrom": "",
            "searchCriteria.classTypeTo": "",
            "searchCriteria.originCode": "",
        }

        log(cfg, f"[info] POST search window {window_i}/{windows}: {a}–{b}")
        results_html = fetch_html(sess, cfg, search_process, method="POST", data=data)

        csv_bytes = export_search_results_csv(sess, cfg)
        if csv_bytes:
            ids = extract_ttbids_from_csv_bytes(csv_bytes)
            log(cfg, f"[info] CSV export IDs: {len(ids)}")
            # Save CSV for debugging/repro
            if cfg.verbose:
                (cfg.outdir / f"search_{a}_{b}.csv").write_bytes(csv_bytes)
        else:
            ids = extract_ttbids_from_results_html(results_html)
            log(cfg, f"[info] HTML fallback IDs: {len(ids)}")
            if cfg.verbose:
                (cfg.outdir / f"search_{a}_{b}.html").write_text(
                    results_html, encoding="utf-8", errors="ignore"
                )

        if not ids:
            log(cfg, f"[warn] no IDs found for window {a}–{b}; continuing")
            continue

        new_ids = [x for x in ids if x not in seen]
        random.shuffle(new_ids)
        new_ids = new_ids[:per_window]

        log(
            cfg, f"[info] new IDs this window: {len(new_ids)} (seen total: {len(seen)})"
        )
        for x in new_ids:
            seen.add(x)
            pool.append(x)

        save_state(state_path, seen, downloaded)

        if len(pool) >= target * 2:
            break

    if not pool:
        log(
            cfg,
            "[error] pool is empty (no IDs discovered). Try smaller date "
            "windows or enable --verbose to inspect saved HTML/CSV.",
        )
        save_state(state_path, seen, downloaded)
        print("Done. No COLAs downloaded (no IDs discovered).", file=sys.stderr)
        sys.exit(2)

    # Decide what to download now
    pool = [x for x in pool if x not in downloaded]
    random.shuffle(pool)
    chosen = pool[:target]

    log(cfg, f"[info] downloading {len(chosen)} COLAs")
    results = []
    for ttbid in chosen:
        log(cfg, f"[info] downloading ttbid {ttbid}")
        try:
            meta = fetch_one_cola(sess, cfg, ttbid)
        except Exception as e:
            log_err(f"[error] exception downloading {ttbid}: {type(e).__name__}: {e}")
            log_err(traceback.format_exc())
            meta = {
                "ttbid": ttbid,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "downloaded_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        results.append(meta)
        if meta.get("status") == "ok":
            downloaded.add(ttbid)
        save_state(state_path, seen, downloaded)

        if args.test:
            log(cfg, "[info] --test mode: exiting after one download attempt")
            break

    (cfg.outdir / "run_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"Done. Downloaded {ok} COLAs into: {cfg.samples_dir}")


if __name__ == "__main__":
    main()
