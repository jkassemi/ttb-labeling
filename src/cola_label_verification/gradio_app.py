import datetime as dt
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Final, Literal, cast

import gradio as gr
from PIL import Image

from cola_label_verification.models import FieldExtraction, LabelInfo
from cola_label_verification.ocr import extract_label_info_with_spans
from cola_label_verification.rules import (
    ApplicationFields,
    ChecklistResult,
    evaluate_checklist,
)
from cola_label_verification.vlm import preload_qwen_model

JobStatus = Literal["queued", "running", "completed", "failed"]
JobDecision = Literal["accepted", "denied"]
BeverageType = Literal["distilled_spirits", "wine"]

_JOB_COLUMNS: Final = (
    "Job",
    "Status",
    "Submitted",
    "Images",
    "Brand",
    "Beverage",
)
_REVIEW_COLUMNS: Final = (
    "Job",
    "Completed",
    "Images",
    "Brand",
    "Beverage",
)
_FINDING_COLUMNS: Final = ("Rule", "Status", "Severity", "Message")
_FIELD_COLUMNS: Final = ("Field", "Confidence", "Status", "Value")
_POLL_INTERVAL_S: Final = 2.0
_COMPLETED_VISIBILITY_S: Final = 15.0
_DATAFRAME_TEXT: Final[Literal["str"]] = "str"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobPayload:
    image_paths: tuple[str, ...]
    original_names: tuple[str, ...]
    application_fields: ApplicationFields | None
    beverage_type: Literal["distilled_spirits", "wine"] | None


@dataclass
class JobResult:
    label_info: LabelInfo | None
    checklist: ChecklistResult | None
    error: str | None


@dataclass
class JobState:
    job_id: str
    status: JobStatus
    submitted_at: float
    payload: JobPayload
    started_at: float | None = None
    completed_at: float | None = None
    result: JobResult | None = None
    decision: JobDecision | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, payload: JobPayload) -> JobState:
        job_id = uuid.uuid4().hex
        job = JobState(
            job_id=job_id,
            status="queued",
            submitted_at=time.time(),
            payload=payload,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, job_ids: list[str]) -> list[JobState]:
        with self._lock:
            jobs = [self._jobs[job_id] for job_id in job_ids if job_id in self._jobs]
        return sorted(jobs, key=lambda job: job.submitted_at, reverse=True)

    def list_review_jobs(self, job_ids: list[str]) -> list[JobState]:
        jobs = self.list_jobs(job_ids)
        return [
            job for job in jobs if job.status == "completed" and job.decision is None
        ]

    def decide(
        self,
        job_id: str,
        decision: JobDecision,
    ) -> JobState | None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return None
        job.decision = decision
        # Prototype: remove files after accept/deny. In production we would
        # persist artifacts + audit metadata instead of deleting them.
        for path in job.payload.image_paths:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return job

    def _update_job(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            self._update_job(job_id, status="running", started_at=time.time())
            try:
                result = _process_job(job.payload)
                self._update_job(
                    job_id,
                    status="completed",
                    completed_at=time.time(),
                    result=result,
                )
            except Exception as exc:
                self._update_job(
                    job_id,
                    status="failed",
                    completed_at=time.time(),
                    result=JobResult(label_info=None, checklist=None, error=str(exc)),
                )


def _process_job(payload: JobPayload) -> JobResult:
    images = _open_images(payload.image_paths)
    try:
        extraction = extract_label_info_with_spans(images)
        label_info = extraction.label_info
        checklist = evaluate_checklist(
            label_info,
            application_fields=payload.application_fields,
            images=images,
            spans=extraction.spans,
        )
        return JobResult(label_info=label_info, checklist=checklist, error=None)
    finally:
        _close_images(images)


def _open_images(paths: tuple[str, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        images.append(Image.open(path))
    return images


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _coerce_beverage_type(value: str | None) -> BeverageType | None:
    if value in {"distilled_spirits", "wine"}:
        return cast(BeverageType, value)
    return None


def _string_datatypes(count: int) -> list[Literal["str"]]:
    return [_DATAFRAME_TEXT for _ in range(count)]


def _application_fields_from_inputs(
    brand_name: str | None,
    class_type: str | None,
    alcohol_content: str | None,
    net_contents: str | None,
    name_and_address: str | None,
    warning_text: str | None,
    grape_varietals: str | None,
    appellation_of_origin: str | None,
    beverage_type: BeverageType | None,
    source_of_product: list[str] | None,
) -> ApplicationFields | None:
    fields = ApplicationFields(
        beverage_type=beverage_type,
        brand_name=_optional_text(brand_name),
        class_type=_optional_text(class_type),
        alcohol_content=_optional_text(alcohol_content),
        net_contents=_optional_text(net_contents),
        name_and_address=_optional_text(name_and_address),
        warning_text=_optional_text(warning_text),
        grape_varietals=_optional_text(grape_varietals),
        appellation_of_origin=_optional_text(appellation_of_origin),
        source_of_product=tuple(source_of_product) if source_of_product else None,
    )
    if any(value is not None for value in asdict(fields).values()):
        return fields
    return None


def _extract_file_info(files: list[object] | None) -> JobPayload | None:
    if not files:
        return None
    image_paths: list[str] = []
    original_names: list[str] = []
    for item in files:
        if isinstance(item, str):
            path = item
            original = Path(path).name
        else:
            path = getattr(item, "name", None)
            original = getattr(item, "orig_name", None)
            if not original and path:
                original = Path(path).name
        if not path:
            continue
        if not Path(path).exists():
            continue
        image_paths.append(path)
        original_names.append(original or Path(path).name)
    if not image_paths:
        return None
    return JobPayload(
        image_paths=tuple(image_paths),
        original_names=tuple(original_names),
        application_fields=None,
        beverage_type=None,
    )


def _format_ts(value: float | None) -> str:
    if value is None:
        return "-"
    stamp = dt.datetime.fromtimestamp(value)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _short_id(job_id: str) -> str:
    return job_id[:8]


def _job_rows(jobs: list[JobState]) -> list[list[str]]:
    rows: list[list[str]] = []
    now = time.time()
    for job in jobs:
        if (
            job.status == "completed"
            and job.completed_at is not None
            and now - job.completed_at > _COMPLETED_VISIBILITY_S
        ):
            continue
        brand = (
            job.payload.application_fields.brand_name
            if job.payload.application_fields
            else None
        )
        beverage = job.payload.beverage_type or "-"
        rows.append(
            [
                _short_id(job.job_id),
                job.status,
                _format_ts(job.submitted_at),
                str(len(job.payload.image_paths)),
                brand or "-",
                beverage,
            ]
        )
    return rows


def _review_rows(jobs: list[JobState]) -> list[list[str]]:
    rows: list[list[str]] = []
    for job in jobs:
        brand = (
            job.payload.application_fields.brand_name
            if job.payload.application_fields
            else None
        )
        beverage = job.payload.beverage_type or "-"
        rows.append(
            [
                _short_id(job.job_id),
                _format_ts(job.completed_at),
                str(len(job.payload.image_paths)),
                brand or "-",
                beverage,
            ]
        )
    return rows


def _findings_rows(checklist: ChecklistResult | None) -> list[list[str]]:
    if checklist is None:
        return []
    return [
        [finding.rule_id, finding.status, finding.severity, finding.message]
        for finding in checklist.findings
    ]


def _label_payload(label_info: LabelInfo | None) -> dict[str, object]:
    if label_info is None:
        return {}
    return {"label_info": label_info.model_dump()}


def _gallery_items(job: JobState | None) -> list[tuple[str, str]]:
    if job is None:
        return []
    items: list[tuple[str, str]] = []
    for path, name in zip(
        job.payload.image_paths, job.payload.original_names, strict=False
    ):
        items.append((path, name))
    return items


def _field_rows(label_info: LabelInfo | None) -> list[list[str]]:
    if label_info is None:
        return []
    rows: list[list[str]] = []
    for field_name, field in label_info.__class__.model_fields.items():
        if field.annotation is not FieldExtraction:
            continue
        field_value = getattr(label_info, field_name)
        rows.append(
            [
                field_name,
                _format_confidence(field_value.confidence),
                field_value.status or "-",
                field_value.value or "-",
            ]
        )
    return rows


def _format_confidence(value: object) -> str:
    if not isinstance(value, (float, int)):
        return "-"
    return f"{value:.3f}"


def create_app() -> gr.Blocks:
    store = JobStore()

    with gr.Blocks(title="COLA Label Verification") as app:
        gr.Markdown(
            "# COLA Label Verification\n"
            "Upload label images, optionally add application fields, and review "
            "completed OCR checks as they finish in the queue."
        )

        session_job_ids = gr.State([])
        review_job_ids = gr.State([])
        selected_job_id = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                image_files = gr.Files(
                    label="Label images",
                    file_count="multiple",
                    file_types=["image"],
                )
                with gr.Accordion("Optional application fields", open=False):
                    brand_name = gr.Textbox(label="Brand name")
                    class_type = gr.Textbox(label="Class/type")
                    alcohol_content = gr.Textbox(label="Alcohol content")
                    net_contents = gr.Textbox(label="Net contents")
                    name_and_address = gr.Textbox(label="Name and address")
                    warning_text = gr.Textbox(label="Government warning")
                    grape_varietals = gr.Textbox(label="Grape varietals")
                    appellation_of_origin = gr.Textbox(label="Appellation of origin")
                    beverage_type = gr.Dropdown(
                        label="Beverage type",
                        choices=["distilled_spirits", "wine"],
                        value=None,
                    )
                    source_of_product = gr.CheckboxGroup(
                        label="Source of product",
                        choices=["domestic", "imported"],
                    )
                submit = gr.Button("Queue verification", variant="primary")
                submit_status = gr.Markdown()
            with gr.Column(scale=3):
                job_table = gr.Dataframe(
                    headers=list(_JOB_COLUMNS),
                    datatype=_string_datatypes(len(_JOB_COLUMNS)),
                    interactive=False,
                    label="Processing queue",
                )
                review_table = gr.Dataframe(
                    headers=list(_REVIEW_COLUMNS),
                    datatype=_string_datatypes(len(_REVIEW_COLUMNS)),
                    interactive=False,
                    label="Review queue",
                )
                gr.Markdown("Click a row in the review queue to load it.")
                job_summary = gr.Markdown()
                label_gallery = gr.Gallery(
                    label="Label images",
                    columns=3,
                    height=220,
                    allow_preview=True,
                )
                findings_table = gr.Dataframe(
                    headers=list(_FINDING_COLUMNS),
                    datatype=_string_datatypes(len(_FINDING_COLUMNS)),
                    interactive=False,
                    label="Checklist findings",
                )
                with gr.Row():
                    accept_button = gr.Button("Accept", variant="primary")
                    deny_button = gr.Button("Deny", variant="stop")
                decision_status = gr.Markdown()
                field_table = gr.Dataframe(
                    headers=list(_FIELD_COLUMNS),
                    datatype=_string_datatypes(len(_FIELD_COLUMNS)),
                    interactive=False,
                    label="Field confidence",
                )
                label_json = gr.JSON(label="Extracted label fields")

        def submit_job(
            files: list[object] | None,
            brand_name_value: str | None,
            class_type_value: str | None,
            alcohol_content_value: str | None,
            net_contents_value: str | None,
            name_and_address_value: str | None,
            warning_text_value: str | None,
            grape_varietals_value: str | None,
            appellation_value: str | None,
            beverage_value: str | None,
            source_value: list[str] | None,
            job_ids: list[str],
        ) -> tuple[
            list[str],
            list[list[str]],
            list[list[str]],
            str,
            list[str],
            list[object] | None,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str | None,
            list[str],
        ]:
            payload = _extract_file_info(files)
            if payload is None:
                review_jobs = store.list_review_jobs(job_ids)
                return (
                    job_ids,
                    _job_rows(store.list_jobs(job_ids)),
                    _review_rows(review_jobs),
                    "Please upload at least one image before queuing.",
                    [item.job_id for item in review_jobs],
                    files,
                    brand_name_value or "",
                    class_type_value or "",
                    alcohol_content_value or "",
                    net_contents_value or "",
                    name_and_address_value or "",
                    warning_text_value or "",
                    grape_varietals_value or "",
                    appellation_value or "",
                    beverage_value,
                    source_value or [],
                )
            normalized_beverage = _coerce_beverage_type(beverage_value)
            fields = _application_fields_from_inputs(
                brand_name_value,
                class_type_value,
                alcohol_content_value,
                net_contents_value,
                name_and_address_value,
                warning_text_value,
                grape_varietals_value,
                appellation_value,
                normalized_beverage,
                source_value,
            )
            updated_payload = JobPayload(
                image_paths=payload.image_paths,
                original_names=payload.original_names,
                application_fields=fields,
                beverage_type=normalized_beverage,
            )
            job = store.submit(updated_payload)
            job_ids = job_ids + [job.job_id]
            jobs = store.list_jobs(job_ids)
            review_jobs = store.list_review_jobs(job_ids)
            message = (
                f"Queued job `{_short_id(job.job_id)}` with "
                f"{len(job.payload.image_paths)} image(s)."
            )
            return (
                job_ids,
                _job_rows(jobs),
                _review_rows(review_jobs),
                message,
                [item.job_id for item in review_jobs],
                [],
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                None,
                [],
            )

        def poll_jobs(
            job_ids: list[str],
            selected_job_id: str | None,
        ) -> tuple[list[list[str]], list[list[str]], list[str], str | None]:
            jobs = store.list_jobs(job_ids)
            review_jobs = store.list_review_jobs(job_ids)
            review_ids = [job.job_id for job in review_jobs]
            valid_ids = set(review_ids)
            if selected_job_id not in valid_ids:
                selected_job_id = None
            return (
                _job_rows(jobs),
                _review_rows(review_jobs),
                review_ids,
                selected_job_id,
            )

        def load_job(
            job_id: str | None,
        ) -> tuple[
            str,
            list[list[str]],
            list[list[str]],
            dict[str, object],
            list[tuple[str, str]],
        ]:
            if not job_id:
                return "", [], [], {}, []
            job = store.get(job_id)
            if job is None:
                return "Job not found.", [], [], {}, []
            summary = (
                f"**Job {_short_id(job.job_id)}** · "
                f"status: `{job.status}` · "
                f"submitted {_format_ts(job.submitted_at)}"
            )
            if job.status != "completed":
                return summary, [], [], {}, _gallery_items(job)
            result = job.result
            if result and result.error:
                return (
                    f"{summary}\n\nError: `{result.error}`",
                    [],
                    [],
                    {},
                    _gallery_items(job),
                )
            checklist = result.checklist if result else None
            label_info = result.label_info if result else None
            return (
                summary,
                _field_rows(label_info),
                _findings_rows(checklist),
                _label_payload(label_info),
                _gallery_items(job),
            )

        def load_job_from_row(
            review_ids: list[str],
            evt: gr.SelectData,
        ) -> tuple[
            str | None,
            str,
            list[list[str]],
            list[list[str]],
            dict[str, object],
            list[tuple[str, str]],
        ]:
            row_index: int | None = None
            if isinstance(evt.index, tuple):
                row_index = evt.index[0]
            elif isinstance(evt.index, list):
                row_index = evt.index[0] if evt.index else None
            elif isinstance(evt.index, int):
                row_index = evt.index
            if row_index is None or row_index >= len(review_ids):
                return None, "", [], [], {}, []
            job_id = review_ids[row_index]
            (
                summary,
                field_rows,
                findings_rows,
                label_payload,
                images,
            ) = load_job(job_id)
            return (
                job_id,
                summary,
                field_rows,
                findings_rows,
                label_payload,
                images,
            )

        def decide_job(
            job_id: str | None,
            job_ids: list[str],
            *,
            decision: JobDecision,
        ) -> tuple[
            list[str],
            list[list[str]],
            list[list[str]],
            list[str],
            str,
            str | None,
            str,
            list[list[str]],
            list[list[str]],
            dict[str, object],
            list[tuple[str, str]],
        ]:
            if not job_id:
                review_jobs = store.list_review_jobs(job_ids)
                return (
                    job_ids,
                    _job_rows(store.list_jobs(job_ids)),
                    _review_rows(review_jobs),
                    [item.job_id for item in review_jobs],
                    "Select a completed job before deciding.",
                    None,
                    "",
                    [],
                    [],
                    {},
                    [],
                )
            decided = store.decide(job_id, decision)
            if decided is None:
                review_jobs = store.list_review_jobs(job_ids)
                return (
                    job_ids,
                    _job_rows(store.list_jobs(job_ids)),
                    _review_rows(review_jobs),
                    [item.job_id for item in review_jobs],
                    "Job not found or already removed.",
                    None,
                    "",
                    [],
                    [],
                    {},
                    [],
                )
            job_ids = [item for item in job_ids if item != job_id]
            jobs = store.list_jobs(job_ids)
            review_jobs = store.list_review_jobs(job_ids)
            status = f"Job `{_short_id(job_id)}` marked **{decision}** and removed."
            return (
                job_ids,
                _job_rows(jobs),
                _review_rows(review_jobs),
                [item.job_id for item in review_jobs],
                status,
                None,
                "",
                [],
                [],
                {},
                [],
            )

        submit.click(
            submit_job,
            inputs=[
                image_files,
                brand_name,
                class_type,
                alcohol_content,
                net_contents,
                name_and_address,
                warning_text,
                grape_varietals,
                appellation_of_origin,
                beverage_type,
                source_of_product,
                session_job_ids,
            ],
            outputs=[
                session_job_ids,
                job_table,
                review_table,
                submit_status,
                review_job_ids,
                image_files,
                brand_name,
                class_type,
                alcohol_content,
                net_contents,
                name_and_address,
                warning_text,
                grape_varietals,
                appellation_of_origin,
                beverage_type,
                source_of_product,
            ],
        )

        review_table.select(
            load_job_from_row,
            inputs=[review_job_ids],
            outputs=[
                selected_job_id,
                job_summary,
                field_table,
                findings_table,
                label_json,
                label_gallery,
            ],
        )

        refresher = gr.Timer(value=_POLL_INTERVAL_S)
        refresher.tick(
            poll_jobs,
            inputs=[session_job_ids, selected_job_id],
            outputs=[job_table, review_table, review_job_ids, selected_job_id],
        )

        accept_button.click(
            partial(decide_job, decision="accepted"),
            inputs=[selected_job_id, session_job_ids],
            outputs=[
                session_job_ids,
                job_table,
                review_table,
                review_job_ids,
                decision_status,
                selected_job_id,
                job_summary,
                field_table,
                findings_table,
                label_json,
                label_gallery,
            ],
        )

        deny_button.click(
            partial(decide_job, decision="denied"),
            inputs=[selected_job_id, session_job_ids],
            outputs=[
                session_job_ids,
                job_table,
                review_table,
                review_job_ids,
                decision_status,
                selected_job_id,
                job_summary,
                field_table,
                findings_table,
                label_json,
                label_gallery,
            ],
        )

    return app


def main() -> None:
    preload_qwen_model()
    logger.info("Qwen VLM model loaded at startup.")
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
